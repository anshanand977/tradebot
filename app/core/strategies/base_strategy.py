"""
Base Strategy Abstract Class
==============================
All strategies must inherit from BaseStrategy and implement analyze().
The framework provides a standardized StrategySignal output that the
VotingEngine uses to aggregate decisions.

Industry-Standard Defaults
---------------------------
- Stop loss: 2×ATR from entry (enough room for normal candle variance)
- Target 1: 2R (minimum 1:2 R:R — industry standard minimum)
- Target 2: 3R
- Target 3: 5R
- Entry Zone: ±0.8% band around optimal entry for pending zone orders
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
import pandas as pd
from loguru import logger
from app.config import settings


@dataclass
class StrategySignal:
    """
    Typed output from a strategy analysis.
    Every field is required — no faking numbers.
    """
    strategy_name: str
    direction: Literal["BUY", "SELL", "HOLD", "NO_TRADE"]
    confidence: float                  # 0.0 - 1.0  (signal-internal estimate)
    entry_price: float
    stop_loss: float
    targets: List[float]               # [T1, T2, T3]
    risk_reward: float                 # positive value, e.g. 2.5 means 1:2.5
    contributing_signals: List[str]    # Human-readable list of reasons
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "1d"
    ticker: str = ""
    pattern_context: str = ""          # Any pattern that triggered the signal

    # Entry zone: defines a price band where the trade should be entered.
    # The auto-trader will only execute when live price falls inside this zone.
    entry_zone_low: float = 0.0        # Lower bound of the optimal entry zone
    entry_zone_high: float = 0.0       # Upper bound of the optimal entry zone

    @property
    def is_actionable(self) -> bool:
        """
        A signal is actionable only when:
        1. Direction is BUY or SELL
        2. Confidence >= 55% (strategy-level threshold)
        3. R:R >= MIN_RISK_REWARD
        """
        return (
            self.direction in ("BUY", "SELL") and
            self.confidence >= 0.55 and
            self.risk_reward >= settings.MIN_RISK_REWARD
        )

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        return abs(self.targets[0] - self.entry_price) if self.targets else 0.0

    def has_valid_entry_zone(self) -> bool:
        """Returns True if the entry zone is defined and non-trivial."""
        return (
            self.entry_zone_low > 0 and
            self.entry_zone_high > 0 and
            self.entry_zone_low < self.entry_zone_high
        )

    def price_in_entry_zone(self, price: float) -> bool:
        """Returns True if the given price falls within the entry zone."""
        if not self.has_valid_entry_zone():
            return False
        return self.entry_zone_low <= price <= self.entry_zone_high


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses implement:
      - analyze(df, indicators, ticker, timeframe) → StrategySignal
      - name (property)
      - description (property)
      - category (property)
    """

    def __init__(self, weight: float = 1.0):
        self._weight = weight
        self._enabled = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the strategy logic."""
        ...

    @property
    def category(self) -> str:
        """Strategy category: TREND / MOMENTUM / REVERSAL / BREAKOUT / MEAN_REVERSION"""
        return "GENERIC"

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = max(0.1, min(3.0, value))  # Clamp weight

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        ticker: str = "",
        timeframe: str = "1d",
    ) -> StrategySignal:
        """
        Core analysis method. Must return a StrategySignal.
        Never raise exceptions — catch internally and return NO_TRADE.
        """
        ...

    def _no_trade(self, ticker: str, timeframe: str, reason: str = "") -> StrategySignal:
        """Helper to return a standardized NO_TRADE signal."""
        close = 0.0
        return StrategySignal(
            strategy_name=self.name,
            direction="NO_TRADE",
            confidence=0.0,
            entry_price=close,
            stop_loss=close,
            targets=[],
            risk_reward=0.0,
            contributing_signals=[reason] if reason else ["No qualifying setup"],
            ticker=ticker,
            timeframe=timeframe,
        )

    def _calculate_rr(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk:reward ratio (expressed as reward multiple)."""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def _atr_stop(self, indicators: Dict, entry: float, direction: str,
                  multiplier: float = 2.0) -> float:
        """
        ATR-based stop loss.
        Default multiplier is 2.0 (industry standard for daily timeframes).
        This gives the trade enough room to breathe through normal candle variance
        without being stopped out on legitimate pullbacks.
        """
        atr = indicators.get("atr_14", entry * 0.015)  # Fallback: 1.5% of price
        if direction == "BUY":
            return round(entry - multiplier * atr, 2)
        return round(entry + multiplier * atr, 2)

    def _atr_targets(self, indicators: Dict, entry: float, stop: float,
                     direction: str, ratios: List[float] = None) -> List[float]:
        """
        Generate targets based on R:R ratios from entry.
        Default ratios: [2.0, 3.0, 5.0]
          T1 = 2R (industry minimum 1:2)
          T2 = 3R (ideal risk:reward)
          T3 = 5R (swing target)
        """
        if ratios is None:
            ratios = [2.0, 3.0, 5.0]
        risk = abs(entry - stop)
        targets = []
        for r in ratios:
            if direction == "BUY":
                targets.append(round(entry + risk * r, 2))
            else:
                targets.append(round(entry - risk * r, 2))
        return targets

    def _calculate_entry_zone(
        self,
        entry: float,
        direction: str,
        indicators: Dict,
        zone_type: str = "PULLBACK",
    ) -> Tuple[float, float]:
        """
        Calculate the optimal entry zone (low, high) for a pending zone order.

        Zone types:
          PULLBACK    — Wait for a slight dip/rally to a better price
          BREAKOUT    — Enter just above/below the breakout level (tight zone)
          REVERSAL    — Enter at the extreme (already near a low/high)
          SUPPORT     — Buy zone is near the support level
          RESISTANCE  — Sell zone is near the resistance level

        Returns: (zone_low, zone_high)
        """
        from app.config import settings
        buf = settings.ENTRY_ZONE_BUFFER_PCT / 100  # e.g., 0.008

        atr = indicators.get("atr_14", entry * 0.015)
        ema20 = indicators.get("ema_20", entry)
        vwap = indicators.get("vwap", entry)

        if direction == "BUY":
            if zone_type == "PULLBACK":
                # For trend/pullback: buy zone is between EMA20 and 0.5% above current
                optimal_entry = max(ema20, entry * 0.995)  # Near EMA or slight dip
                zone_low  = round(optimal_entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf * 0.5), 2)  # Slight upside for momentum
            elif zone_type == "BREAKOUT":
                # For breakouts: enter very close to the breakout level
                zone_low  = round(entry * (1 - buf * 0.3), 2)  # Very tight below
                zone_high = round(entry * (1 + buf), 2)
            elif zone_type == "REVERSAL":
                # For reversals: already at extreme, buy immediately in tight zone
                zone_low  = round(entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf), 2)
            else:
                zone_low  = round(entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf), 2)
        else:  # SELL
            if zone_type == "PULLBACK":
                optimal_entry = min(ema20, entry * 1.005)
                zone_low  = round(entry * (1 - buf * 0.5), 2)
                zone_high = round(optimal_entry * (1 + buf), 2)
            elif zone_type == "BREAKOUT":
                zone_low  = round(entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf * 0.3), 2)
            elif zone_type == "REVERSAL":
                zone_low  = round(entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf), 2)
            else:
                zone_low  = round(entry * (1 - buf), 2)
                zone_high = round(entry * (1 + buf), 2)

        return zone_low, zone_high

"""
Base Strategy Abstract Class
==============================
All strategies must inherit from BaseStrategy and implement analyze().
The framework provides a standardized StrategySignal output that the
VotingEngine uses to aggregate decisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import pandas as pd
from loguru import logger


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

    @property
    def is_actionable(self) -> bool:
        return self.direction in ("BUY", "SELL") and self.confidence >= 0.55

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        return abs(self.targets[0] - self.entry_price) if self.targets else 0.0


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
                  multiplier: float = 1.5) -> float:
        """ATR-based stop loss."""
        atr = indicators.get("atr_14", entry * 0.01)
        if direction == "BUY":
            return round(entry - multiplier * atr, 2)
        return round(entry + multiplier * atr, 2)

    def _atr_targets(self, indicators: Dict, entry: float, stop: float,
                     direction: str, ratios: List[float] = None) -> List[float]:
        """Generate targets based on R:R ratios."""
        if ratios is None:
            ratios = [1.5, 2.5, 4.0]
        risk = abs(entry - stop)
        targets = []
        for r in ratios:
            if direction == "BUY":
                targets.append(round(entry + risk * r, 2))
            else:
                targets.append(round(entry - risk * r, 2))
        return targets

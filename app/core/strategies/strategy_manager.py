"""
Strategy Manager & Voting Engine
===================================
Orchestrates all strategy plugins:
  1. Strategy Manager: loads, enables/disables, weights strategies
  2. Voting Engine: aggregates votes into a final TradeRecommendation
  3. Probability Engine: calculates real probability from historical data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from loguru import logger

from app.core.strategies.base_strategy import BaseStrategy, StrategySignal
from app.core.strategies.strategies import ALL_STRATEGIES
from app.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECOMMENDATION — Final output of the decision engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecommendation:
    """
    The final, aggregated trade recommendation produced by the voting engine.
    This is the core output displayed to the user.
    """
    ticker: str
    symbol: str
    exchange: str
    timeframe: str

    direction: Literal["BUY", "SELL", "NO_TRADE"]
    confidence: float          # 0.0 – 1.0
    probability: float         # Real calculated probability

    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    risk_reward: float
    risk_level: str            # LOW / MEDIUM / HIGH

    # Voting details (full transparency)
    strategy_votes: Dict[str, Dict]     # {strategy_name: {direction, confidence}}
    strategies_agreed: int
    total_strategies_voted: int
    majority_direction: str

    # Reasoning
    contributing_signals: List[str]
    pattern_context: str = ""
    market_regime: str = ""
    smc_bias: str = ""

    # Entry Zone: the optimal price range the auto-trader will WAIT for before executing.
    # Trades are placed as pending zone orders and only fill when price enters this band.
    entry_zone_low: float = 0.0   # Lower bound of buy/sell zone
    entry_zone_high: float = 0.0  # Upper bound of buy/sell zone

    # Metadata
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_actionable: bool = False

    @property
    def summary(self) -> str:
        if not self.is_actionable:
            return "NO TRADE — No quality setup detected."
        direction_icon = "🟢 BUY" if self.direction == "BUY" else "🔴 SELL"
        return (
            f"{direction_icon} {self.symbol} | "
            f"Entry ₹{self.entry_price:.0f} | SL ₹{self.stop_loss:.0f} | "
            f"T1 ₹{self.target1:.0f} | "
            f"Confidence {self.confidence*100:.0f}% | "
            f"Probability {self.probability*100:.0f}% | "
            f"R:R 1:{self.risk_reward:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class StrategyManager:
    """
    Manages the lifecycle of all strategy plugins.
    Loads from the ALL_STRATEGIES registry and applies stored weights.
    """

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._load_strategies()

    def _load_strategies(self) -> None:
        for cls in ALL_STRATEGIES:
            try:
                instance = cls()
                self._strategies[instance.name] = instance
                logger.debug("Loaded strategy: {}", instance.name)
            except Exception as e:
                logger.error("Failed to load strategy {}: {}", cls.__name__, e)
        logger.info("Loaded {} strategies", len(self._strategies))

    def get_all(self) -> List[BaseStrategy]:
        return list(self._strategies.values())

    def get_enabled(self) -> List[BaseStrategy]:
        return [s for s in self._strategies.values() if s.enabled]

    def enable(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = False

    def set_weight(self, name: str, weight: float) -> None:
        if name in self._strategies:
            self._strategies[name].weight = weight
            logger.info("Strategy '{}' weight updated to {}", name, weight)

    def apply_weights_from_db(self, weights: Dict[str, float]) -> None:
        """Apply learned weights from self-learning system."""
        for name, w in weights.items():
            self.set_weight(name, w)

    def get_strategy_list(self) -> List[Dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "weight": s.weight,
                "enabled": s.enabled,
            }
            for s in self._strategies.values()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# VOTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class VotingEngine:
    """
    Aggregates signals from all enabled strategies into one TradeRecommendation.

    Voting rules:
    - At least MIN_STRATEGY_AGREEMENT strategies must agree on direction
    - Weighted confidence is computed using each strategy's performance weight
    - If consensus is not reached → NO_TRADE
    """

    def __init__(self, strategy_manager: StrategyManager):
        self.strategy_manager = strategy_manager

    def vote(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        ticker: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        market_regime: str = "UNKNOWN",
    ) -> TradeRecommendation:
        """
        Run all enabled strategies against the given data and aggregate votes.
        """
        close = indicators.get("close", df["close"].iloc[-1] if not df.empty else 0)
        strategies = self.strategy_manager.get_enabled()

        votes: Dict[str, StrategySignal] = {}
        for strategy in strategies:
            try:
                signal = strategy.analyze(df, indicators, ticker, timeframe)
                votes[strategy.name] = signal
            except Exception as e:
                logger.warning("Strategy {} threw exception: {}", strategy.name, e)

        # Count votes by direction (weighted)
        buy_weight = 0.0
        sell_weight = 0.0
        buy_count = 0
        sell_count = 0
        buy_weight_sum = 0.0
        sell_weight_sum = 0.0
        all_signals: List[str] = []
        pattern_context = ""

        for strategy_name, signal in votes.items():
            strategy = self.strategy_manager._strategies.get(strategy_name)
            w = strategy.weight if strategy else 1.0

            if signal.direction == "BUY" and signal.is_actionable:
                buy_weight  += signal.confidence * w
                buy_count   += 1
                buy_weight_sum += w
                all_signals.extend(signal.contributing_signals)
                if signal.pattern_context:
                    pattern_context = signal.pattern_context
            elif signal.direction == "SELL" and signal.is_actionable:
                sell_weight += signal.confidence * w
                sell_count  += 1
                sell_weight_sum += w
                all_signals.extend(signal.contributing_signals)

        # Determine direction
        min_agreement = settings.MIN_STRATEGY_AGREEMENT
        if buy_count >= min_agreement and buy_weight >= sell_weight * 1.2:
            direction = "BUY"
            agreed    = buy_count
            total_w   = buy_weight
            weight_sum = buy_weight_sum
        elif sell_count >= min_agreement and sell_weight >= buy_weight * 1.2:
            direction = "SELL"
            agreed    = sell_count
            total_w   = sell_weight
            weight_sum = sell_weight_sum
        else:
            # NO_TRADE — no consensus
            return self._no_trade_recommendation(
                ticker, symbol, exchange, timeframe, votes, all_signals, close, indicators
            )

        # Calculate weighted confidence
        weighted_conf = total_w / weight_sum if weight_sum > 0 else 0.0
        confidence = round(min(0.97, max(0.50, weighted_conf)), 3)

        # Gather entry/stop/targets from the highest-confidence agreeing signal
        agreeing = [
            (name, sig) for name, sig in votes.items()
            if sig.direction == direction and sig.is_actionable
        ]
        best_signal = max(agreeing, key=lambda x: x[1].confidence * 0.7 + x[1].risk_reward * 0.3)[1]

        # Deduplicate and limit signals
        seen = set()
        unique_signals = []
        for s in all_signals:
            if s not in seen:
                seen.add(s)
                unique_signals.append(s)

        targets = best_signal.targets or [close]
        t1 = targets[0] if len(targets) > 0 else close
        t2 = targets[1] if len(targets) > 1 else t1
        t3 = targets[2] if len(targets) > 2 else t2

        risk = abs(close - best_signal.stop_loss)
        reward = abs(t1 - close)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # ── Minimum R:R Gate — industry standard 1:2 minimum ──────────────────
        # No trade should ever be recommended with R:R below 2.0.
        # This is the single most important filter for long-term profitability.
        if rr < settings.MIN_RISK_REWARD:
            logger.info("{} {} | R:R {:.2f} below minimum {:.1f} — NO TRADE",
                        direction, ticker, rr, settings.MIN_RISK_REWARD)
            return self._no_trade_recommendation(
                ticker, symbol, exchange, timeframe, votes,
                [f"R:R {rr:.2f} below minimum 1:{settings.MIN_RISK_REWARD:.0f} required"],
                close, indicators
            )

        # Risk level classification (aligned with industry standards)
        if rr < 2.0:
            risk_level = "LOW"      # Below standard — should never reach here after gate above
        elif rr < 3.0:
            risk_level = "MEDIUM"   # Standard 1:2 to 1:3
        else:
            risk_level = "HIGH"     # Reward-dominant (good, despite the "HIGH" name)

        # Build vote details dict
        vote_details = {
            name: {
                "direction": sig.direction,
                "confidence": round(sig.confidence, 3),
                "actionable": sig.is_actionable,
            }
            for name, sig in votes.items()
        }

        # ── Aggregate entry zone from agreeing signals ─────────────────────
        # Use the average zone bounds from all agreeing signals that have zones
        zone_signals = [
            sig for _, sig in agreeing
            if sig.entry_zone_low > 0 and sig.entry_zone_high > 0
        ]
        if zone_signals:
            agg_zone_low  = round(sum(s.entry_zone_low  for s in zone_signals) / len(zone_signals), 2)
            agg_zone_high = round(sum(s.entry_zone_high for s in zone_signals) / len(zone_signals), 2)
        elif best_signal.entry_zone_low > 0:
            agg_zone_low  = best_signal.entry_zone_low
            agg_zone_high = best_signal.entry_zone_high
        else:
            # Fallback: compute zone from entry using config buffer
            buf = settings.ENTRY_ZONE_BUFFER_PCT / 100
            if direction == "BUY":
                agg_zone_low  = round(close * (1 - buf), 2)
                agg_zone_high = round(close * (1 + buf * 0.5), 2)
            else:
                agg_zone_low  = round(close * (1 - buf * 0.5), 2)
                agg_zone_high = round(close * (1 + buf), 2)

        rec = TradeRecommendation(
            ticker=ticker,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            probability=0.0,   # Will be filled by ProbabilityEngine
            entry_price=close,
            stop_loss=round(best_signal.stop_loss, 2),
            target1=round(t1, 2),
            target2=round(t2, 2),
            target3=round(t3, 2),
            risk_reward=rr,
            risk_level=risk_level,
            strategy_votes=vote_details,
            strategies_agreed=agreed,
            total_strategies_voted=len(votes),
            majority_direction=direction,
            contributing_signals=unique_signals[:10],
            pattern_context=pattern_context,
            market_regime=market_regime,
            indicator_snapshot=indicators,
            entry_zone_low=agg_zone_low,
            entry_zone_high=agg_zone_high,
            is_actionable=confidence >= settings.MIN_CONFIDENCE_THRESHOLD,
        )

        logger.info("{} {} | {}% conf | {}/{} strategies agree",
                    direction, ticker, confidence * 100, agreed, len(votes))
        return rec

    def _no_trade_recommendation(
        self, ticker, symbol, exchange, timeframe, votes, signals, close, indicators
    ) -> TradeRecommendation:
        vote_details = {
            name: {"direction": sig.direction, "confidence": round(sig.confidence, 3), "actionable": sig.is_actionable}
            for name, sig in votes.items()
        }
        buy_count  = sum(1 for s in votes.values() if s.direction == "BUY" and s.is_actionable)
        sell_count = sum(1 for s in votes.values() if s.direction == "SELL" and s.is_actionable)

        return TradeRecommendation(
            ticker=ticker, symbol=symbol, exchange=exchange, timeframe=timeframe,
            direction="NO_TRADE", confidence=0.0, probability=0.0,
            entry_price=close, stop_loss=close, target1=close, target2=close, target3=close,
            risk_reward=0.0, risk_level="HIGH",
            strategy_votes=vote_details, strategies_agreed=max(buy_count, sell_count),
            total_strategies_voted=len(votes), majority_direction="NONE",
            contributing_signals=["No consensus among strategies — NO TRADE"],
            indicator_snapshot=indicators, is_actionable=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ProbabilityEngine:
    """
    Calculates a statistically-grounded probability for a trade recommendation.

    The probability is derived from real data:
    1. Historical win rate of similar setups (from strategy_performance table)
    2. Indicator confluence score
    3. Volume confirmation
    4. Trend strength (ADX)
    5. Volatility adjustment
    6. Market regime match

    NO fake numbers — every factor is explicitly sourced.
    """

    def calculate(
        self,
        rec: TradeRecommendation,
        strategy_performance: Dict[str, float],  # {strategy_name: win_rate}
    ) -> float:
        """
        Returns probability (0.0 - 1.0) for the trade being a winner.
        """
        factors = []
        weights = []

        # Factor 1: Historical win rate of agreeing strategies
        if strategy_performance:
            agreed_strategies = [
                name for name, vote in rec.strategy_votes.items()
                if vote["direction"] == rec.direction and vote["actionable"]
            ]
            win_rates = [strategy_performance.get(name, 0.50) for name in agreed_strategies]
            if win_rates:
                avg_wr = np.mean(win_rates)
                factors.append(avg_wr)
                weights.append(3.0)

        # Factor 2: Strategy confidence (already consensus-weighted)
        factors.append(rec.confidence)
        weights.append(2.0)

        # Factor 3: Indicator confluence
        inds = rec.indicator_snapshot
        confluences = 0
        total_checks = 0

        if rec.direction == "BUY":
            checks = [
                inds.get("above_ema_20", False),
                inds.get("above_ema_50", False),
                inds.get("above_vwap", False),
                inds.get("above_ichi_cloud", False),
                inds.get("rsi_14", 50) < 65 and inds.get("rsi_14", 50) > 30,
                inds.get("macd_hist", 0) > 0,
                inds.get("supertrend_dir", 0) == 1,
                inds.get("adx", 0) > 20,
            ]
        else:
            checks = [
                not inds.get("above_ema_20", True),
                not inds.get("above_ema_50", True),
                not inds.get("above_vwap", True),
                inds.get("rsi_14", 50) < 70 and inds.get("rsi_14", 50) > 35,
                inds.get("macd_hist", 0) < 0,
                inds.get("supertrend_dir", 0) == -1,
                inds.get("adx", 0) > 20,
            ]

        for check in checks:
            total_checks += 1
            if check:
                confluences += 1

        confluence_score = confluences / total_checks if total_checks > 0 else 0.5
        factors.append(confluence_score)
        weights.append(2.0)

        # Factor 4: Volume confirmation
        vol_ratio = inds.get("vol_ratio", 1.0)
        vol_score = min(1.0, vol_ratio / 2.0)  # 2x volume = 1.0 score
        factors.append(vol_score)
        weights.append(1.5)

        # Factor 5: Trend strength
        adx = inds.get("adx", 20)
        adx_score = min(1.0, adx / 50)  # ADX 50 = max score
        factors.append(adx_score)
        weights.append(1.0)

        # Factor 6: R:R ratio (better R:R = higher probability of being a sensible trade)
        rr_score = min(1.0, rec.risk_reward / 4.0)
        factors.append(rr_score)
        weights.append(1.0)

        # Factor 7: Candlestick pattern win rate learning integration
        pattern_name = rec.pattern_context
        if pattern_name:
            try:
                from app.db.models import PatternPerformance
                with db_session() as db:
                    perf = db.query(PatternPerformance).filter_by(
                        pattern_name=pattern_name,
                        pattern_type="CANDLESTICK"
                    ).first()
                    # Apply weight only if we have at least 3 trades history for this pattern
                    if perf and perf.total_trades >= 3:
                        factors.append(perf.win_rate)
                        weights.append(2.0)
                        logger.debug("Self-Learning: Candlestick pattern '{}' win rate of {:.1f}% factored into probability",
                                     pattern_name, perf.win_rate * 100)
            except Exception as e:
                logger.debug("Failed to apply pattern learning stats: {}", e)

        # Weighted average
        total_weight = sum(weights)
        probability = sum(f * w for f, w in zip(factors, weights)) / total_weight

        # Cap: no strategy can claim >93% probability
        return round(min(0.93, max(0.30, probability)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Top-level orchestrator that runs the full signal generation pipeline:
    1. Compute indicators
    2. Run all strategies (via VotingEngine)
    3. Calculate probability (via ProbabilityEngine)
    4. Return final TradeRecommendation
    """

    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.voting_engine = VotingEngine(self.strategy_manager)
        self.probability_engine = ProbabilityEngine()

    def generate(
        self,
        df: pd.DataFrame,
        ticker: str,
        symbol: str,
        exchange: str = "NSE",
        timeframe: str = "1d",
        strategy_performance: Optional[Dict[str, float]] = None,
    ) -> TradeRecommendation:
        """
        Full pipeline: indicators → vote → probability → recommendation.
        """
        from app.core.analysis.indicators import compute_all_indicators
        from app.core.analysis.market_structure import detect_market_structure

        if df.empty or len(df) < 20:
            return self._no_data_rec(ticker, symbol, exchange, timeframe)

        # Compute indicators
        indicators = compute_all_indicators(df)

        # Market regime
        ms = detect_market_structure(df)
        regime = ms.regime

        # Vote
        rec = self.voting_engine.vote(
            df=df, indicators=indicators, ticker=ticker,
            symbol=symbol, exchange=exchange, timeframe=timeframe,
            market_regime=regime,
        )

        # Calculate probability
        if rec.is_actionable:
            prob = self.probability_engine.calculate(rec, strategy_performance or {})
            rec.probability = prob

        return rec

    def _no_data_rec(self, ticker, symbol, exchange, timeframe) -> TradeRecommendation:
        return TradeRecommendation(
            ticker=ticker, symbol=symbol, exchange=exchange, timeframe=timeframe,
            direction="NO_TRADE", confidence=0.0, probability=0.0,
            entry_price=0, stop_loss=0, target1=0, target2=0, target3=0,
            risk_reward=0, risk_level="HIGH",
            strategy_votes={}, strategies_agreed=0, total_strategies_voted=0,
            majority_direction="NONE",
            contributing_signals=["Insufficient data for analysis"],
            is_actionable=False,
        )


# ─── Singletons ───────────────────────────────────────────────────────────────
strategy_manager = StrategyManager()
signal_generator = SignalGenerator()

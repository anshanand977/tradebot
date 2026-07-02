"""
Self-Learning Engine
======================
After every completed simulated trade, adjusts strategy weights
based on historical performance. No autonomous code rewriting.
Only adjusts voting weights using rolling win-rate calculation.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from app.db.database import db_session
from app.db.models import Trade, StrategyPerformance, LearningLog
from app.config import settings


class SelfLearningEngine:
    """
    Adjusts strategy weights based on accumulated trade history.
    Runs after every trade closes and on a scheduled basis.

    Rules:
    - Minimum 20 trades per strategy before adjusting weights
    - Weight decay applied to prevent overfit to recent data
    - Weights clamped to [0.1, 3.0]
    - Learning is transparent and logged to DB
    """

    def __init__(self, strategy_manager=None):
        self.strategy_manager = strategy_manager

    def process_trade_result(self, trade_id: int) -> None:
        """Called after a trade closes. Updates the relevant strategy's performance."""
        try:
            with db_session() as db:
                trade = db.query(Trade).filter_by(id=trade_id).first()
                if not trade or not trade.strategy_used:
                    return

                strategy_name = trade.strategy_used
                market_regime = trade.market_regime or "UNKNOWN"
                timeframe = "1d"  # Default
                outcome = "WIN" if trade.net_pnl > 0 else "LOSS"
                pnl_pct = trade.pnl_pct or 0.0

                # Get or create performance record
                perf = db.query(StrategyPerformance).filter_by(
                    strategy_name=strategy_name,
                    market_regime=market_regime,
                    timeframe=timeframe,
                ).first()

                if not perf:
                    perf = StrategyPerformance(
                        strategy_name=strategy_name,
                        market_regime=market_regime,
                        timeframe=timeframe,
                    )
                    db.add(perf)

                # Update counts
                perf.total_trades += 1
                if outcome == "WIN":
                    perf.wins += 1
                    perf.avg_profit_pct = (
                        (perf.avg_profit_pct * (perf.wins - 1) + pnl_pct) / perf.wins
                    )
                else:
                    perf.losses += 1
                    perf.avg_loss_pct = (
                        (perf.avg_loss_pct * (perf.losses - 1) + abs(pnl_pct)) / perf.losses
                    )

                perf.win_rate = perf.wins / perf.total_trades if perf.total_trades > 0 else 0
                perf.profit_factor = (
                    (perf.avg_profit_pct * perf.wins) /
                    (abs(perf.avg_loss_pct) * perf.losses + 1e-10)
                    if perf.losses > 0 else 99.0
                )

                db.flush()

                # Adjust weight if enough data
                if perf.total_trades >= settings.STRATEGY_MIN_TRADES_FOR_LEARNING:
                    old_weight = perf.current_weight
                    new_weight = self._calculate_new_weight(perf)

                    # Smooth transition
                    smoothed = old_weight * settings.STRATEGY_WEIGHT_DECAY + new_weight * (1 - settings.STRATEGY_WEIGHT_DECAY)
                    perf.current_weight = round(max(0.1, min(3.0, smoothed)), 3)

                    # Log the change
                    if abs(perf.current_weight - old_weight) > 0.01:
                        log = LearningLog(
                            trade_id=trade_id,
                            strategy_name=strategy_name,
                            old_weight=old_weight,
                            new_weight=perf.current_weight,
                            reason=f"Win rate: {perf.win_rate*100:.1f}% over {perf.total_trades} trades",
                            market_regime=market_regime,
                            trade_outcome=outcome,
                            pnl_pct=round(pnl_pct, 2),
                        )
                        db.add(log)

                    # Apply to live strategy manager
                    if self.strategy_manager:
                        self.strategy_manager.set_weight(strategy_name, perf.current_weight)

                    logger.info(
                        "📚 Self-Learn: {} weight {} → {} (Win: {:.1f}%)",
                        strategy_name, old_weight, perf.current_weight, perf.win_rate * 100
                    )

        except Exception as e:
            logger.error("Self-learning error: {}", e)

    def _calculate_new_weight(self, perf: StrategyPerformance) -> float:
        """
        Calculate desired weight from performance metrics.
        - Base: 1.0
        - Win rate > 60%: increase weight
        - Win rate < 40%: decrease weight
        - Profit factor > 2.0: bonus
        """
        win_rate = perf.win_rate
        pf = min(perf.profit_factor, 5.0)  # Cap at 5

        # Win rate contribution (0.1 to 2.0)
        if win_rate >= 0.65:
            wr_weight = 1.5 + (win_rate - 0.65) * 5
        elif win_rate >= 0.50:
            wr_weight = 1.0 + (win_rate - 0.50) * 3.3
        elif win_rate >= 0.40:
            wr_weight = 0.7 + (win_rate - 0.40) * 3
        else:
            wr_weight = max(0.1, win_rate * 1.5)

        # Profit factor contribution
        pf_bonus = min(0.5, (pf - 1.0) * 0.1) if pf > 1.0 else 0

        return round(min(3.0, max(0.1, wr_weight + pf_bonus)), 3)

    def get_strategy_weights(self) -> Dict[str, float]:
        """Returns current weights for all strategies from DB."""
        weights = {}
        try:
            with db_session() as db:
                perfs = db.query(StrategyPerformance).all()
                for p in perfs:
                    if p.total_trades >= settings.STRATEGY_MIN_TRADES_FOR_LEARNING:
                        weights[p.strategy_name] = p.current_weight
        except Exception as e:
            logger.debug("Get strategy weights error: {}", e)
        return weights

    def get_learning_report(self) -> List[Dict]:
        """Returns a detailed performance report for all strategies."""
        report = []
        try:
            with db_session() as db:
                perfs = db.query(StrategyPerformance).order_by(
                    StrategyPerformance.win_rate.desc()
                ).all()
                for p in perfs:
                    report.append({
                        "strategy": p.strategy_name,
                        "market_regime": p.market_regime,
                        "total_trades": p.total_trades,
                        "win_rate": round(p.win_rate * 100, 1),
                        "profit_factor": round(p.profit_factor, 2),
                        "avg_profit_pct": round(p.avg_profit_pct, 2),
                        "avg_loss_pct": round(p.avg_loss_pct, 2),
                        "current_weight": p.current_weight,
                        "reliable": p.total_trades >= settings.STRATEGY_MIN_TRADES_FOR_LEARNING,
                    })
        except Exception as e:
            logger.error("Learning report error: {}", e)
        return report

    def get_best_strategies(self, top_n: int = 5) -> List[Dict]:
        report = self.get_learning_report()
        reliable = [r for r in report if r["reliable"] and r["total_trades"] >= 10]
        return sorted(reliable, key=lambda x: x["win_rate"], reverse=True)[:top_n]

    def get_worst_strategies(self, bottom_n: int = 3) -> List[Dict]:
        report = self.get_learning_report()
        reliable = [r for r in report if r["reliable"]]
        return sorted(reliable, key=lambda x: x["win_rate"])[:bottom_n]


self_learning = SelfLearningEngine()

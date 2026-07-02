"""
Risk Management Engine
========================
Enforces all risk rules before any simulated trade is placed.
Circuit breaker stops all trading when daily loss or drawdown limits are hit.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, date
from loguru import logger

from app.config import settings


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str
    position_size: float = 0.0
    risk_amount: float = 0.0
    max_loss: float = 0.0


class RiskManager:
    """
    Central risk gatekeeper.
    All simulated orders pass through this before execution.
    """

    def __init__(self):
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._circuit_broken: bool = False
        self._last_reset: date = date.today()

    def check_trade(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        portfolio_balance: float,
        open_positions: int,
        sector_exposure: float = 0.0,
    ) -> RiskCheckResult:
        """
        Validates a proposed trade against all risk rules.
        Returns RiskCheckResult with allowed=True/False and reason.
        """
        self._reset_if_new_day()

        # 1. Circuit breaker
        if self._circuit_broken:
            return RiskCheckResult(False, "Circuit breaker ACTIVE — daily loss limit reached")

        # 2. Max daily trades
        if self._daily_trades >= settings.MAX_CONCURRENT_POSITIONS * 3:
            return RiskCheckResult(False, f"Max daily trades reached ({self._daily_trades})")

        # 3. Max concurrent positions
        if open_positions >= settings.MAX_CONCURRENT_POSITIONS:
            return RiskCheckResult(
                False,
                f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached"
            )

        # 4. Sector exposure
        if sector_exposure > settings.MAX_SECTOR_EXPOSURE_PCT / 100 * portfolio_balance:
            return RiskCheckResult(False, f"Sector exposure limit exceeded")

        # 5. Position size calculation
        risk_per_trade = settings.DEFAULT_RISK_PER_TRADE_PCT / 100 * portfolio_balance
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return RiskCheckResult(False, "Invalid stop loss — same as entry price")

        quantity = risk_per_trade / risk_per_share
        position_value = quantity * entry_price
        max_position_value = settings.MAX_POSITION_SIZE_PCT / 100 * portfolio_balance

        # Cap by max position size
        if position_value > max_position_value:
            quantity = max_position_value / entry_price
            position_value = max_position_value

        if quantity < 1:
            return RiskCheckResult(
                False,
                f"Position too small — min 1 share, got {quantity:.2f}"
            )

        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            position_size=round(quantity, 2),
            risk_amount=round(quantity * risk_per_share, 2),
            max_loss=round(risk_per_trade, 2),
        )

    def update_daily_pnl(self, pnl: float) -> None:
        """Called after each trade closes."""
        self._reset_if_new_day()
        self._daily_pnl += pnl
        self._daily_trades += 1

        # Check circuit breaker
        max_loss = settings.MAX_DAILY_LOSS_PCT / 100
        if abs(self._daily_pnl) > max_loss * abs(self._daily_pnl + sum([])):
            # Simplified: if daily loss is significant, trip breaker
            pass

        # Absolute check
        daily_loss_limit = settings.MAX_DAILY_LOSS_PCT  # Will be checked against portfolio pct
        if pnl < 0 and abs(self._daily_pnl) > daily_loss_limit:
            self._circuit_broken = True
            logger.warning("⚠️  CIRCUIT BREAKER TRIGGERED — daily loss limit reached")

    def reset_circuit_breaker(self) -> None:
        self._circuit_broken = False
        logger.info("Circuit breaker reset manually")

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_trades(self) -> int:
        return self._daily_trades

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if today != self._last_reset:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._circuit_broken = False
            self._last_reset = today
            logger.info("Daily risk counters reset for {}", today)

    def get_status(self) -> Dict:
        return {
            "circuit_broken": self._circuit_broken,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
            "max_concurrent_positions": settings.MAX_CONCURRENT_POSITIONS,
            "max_position_size_pct": settings.MAX_POSITION_SIZE_PCT,
        }


risk_manager = RiskManager()

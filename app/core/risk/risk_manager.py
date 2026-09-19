"""
Risk Management Engine
========================
Enforces position sizing and concurrency rules before any simulated trade is placed.
No automatic daily loss circuit breaker — trading is never halted by P&L.
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
    Daily loss limit has been removed — trading is never auto-halted.
    """

    def __init__(self):
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._circuit_broken: bool = False   # Manual override only
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
        Validates a proposed trade against position sizing rules.
        No daily loss limit — circuit breaker is manual-only.
        """
        self._reset_if_new_day()

        # 1. Manual circuit breaker (never triggered automatically)
        if self._circuit_broken:
            return RiskCheckResult(False, "Circuit breaker ACTIVE — reset manually via Settings to resume trading")

        # 2. Max concurrent positions
        if open_positions >= settings.MAX_CONCURRENT_POSITIONS:
            return RiskCheckResult(
                False,
                f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached"
            )

        # 3. Sector exposure
        if sector_exposure > settings.MAX_SECTOR_EXPOSURE_PCT / 100 * portfolio_balance:
            return RiskCheckResult(False, "Sector exposure limit exceeded")

        # 4. Position size calculation
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
        """Called after each trade closes. Tracks P&L for informational display only."""
        self._reset_if_new_day()
        self._daily_pnl += pnl
        self._daily_trades += 1
        # Daily loss limit removed — no circuit breaker auto-trigger

    def reset_circuit_breaker(self) -> None:
        """Manually clear the circuit breaker if it was set via Settings."""
        self._circuit_broken = False
        logger.info("Circuit breaker reset manually")

    def reset_daily_counters(self) -> None:
        """Resets daily P&L counters and clears manual circuit breaker."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._circuit_broken = False
        self._last_reset = date.today()
        logger.info("Daily risk counters reset manually")

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
            # Note: Do not reset manual circuit breaker on new day
            self._last_reset = today
            logger.info("Daily risk counters reset for {}", today)

    def get_status(self) -> Dict:
        return {
            "circuit_broken": self._circuit_broken,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "max_concurrent_positions": settings.MAX_CONCURRENT_POSITIONS,
            "max_position_size_pct": settings.MAX_POSITION_SIZE_PCT,
        }


risk_manager = RiskManager()

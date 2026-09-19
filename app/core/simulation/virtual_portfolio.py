"""
Virtual Portfolio & Paper Trading Engine
==========================================
Simulates order execution, P&L tracking, equity curve, and trade journal.
This is SIMULATION ONLY — never touches real money or broker APIs.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from loguru import logger

from app.config import settings
from app.db.database import db_session
from app.db.models import Trade, Portfolio, EquityCurvePoint
from app.core.risk.risk_manager import risk_manager


@dataclass
class SimulatedOrder:
    order_id: str
    ticker: str
    symbol: str
    direction: str         # BUY / SELL
    quantity: float
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    strategy_name: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeframe: str = "1d"
    exchange: str = "NSE"
    ai_reason: str = ""
    indicator_snapshot: Optional[Dict] = None


@dataclass
class OpenPosition:
    order_id: str
    ticker: str
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    strategy_name: str
    confidence: float
    entry_time: datetime
    timeframe: str = "1d"
    exchange: str = "NSE"
    ai_reason: str = ""
    unrealized_pnl: float = 0.0
    target1_hit: bool = False
    target2_hit: bool = False
    indicator_snapshot: Optional[Dict] = None

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.direction == "BUY":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

    @property
    def position_value(self) -> float:
        return self.quantity * self.current_price


class VirtualPortfolio:
    """
    Simulates a complete paper trading portfolio.
    Initial balance configurable (default 1,00,000 Coins = ₹1 each).
    """

    def __init__(self, portfolio_id: int = 1, initial_balance: float = None):
        self.portfolio_id = portfolio_id
        self.initial_balance = initial_balance or settings.VIRTUAL_INITIAL_BALANCE
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.open_positions: Dict[str, OpenPosition] = {}
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self._load_from_db()

    # ─── Order Placement ──────────────────────────────────────────────────────

    def place_order(self, order: SimulatedOrder) -> Dict:
        """
        Places a simulated order.
        Validates through risk manager first.
        """
        # Double check if already in a position for this ticker
        existing = [p for p in self.open_positions.values() if p.ticker == order.ticker]
        if existing:
            logger.info("Order rejected: Already in a position for {}", order.ticker)
            return {"status": "REJECTED", "reason": f"Already in a position for {order.ticker}"}

        # Risk check
        sector_exposure = self._get_sector_exposure(order.ticker)
        risk_check = risk_manager.check_trade(
            ticker=order.ticker,
            entry_price=order.entry_price,
            stop_loss=order.stop_loss,
            portfolio_balance=self.balance,
            open_positions=len(self.open_positions),
            sector_exposure=sector_exposure,
        )

        if not risk_check.allowed:
            logger.info("Order rejected: {}", risk_check.reason)
            return {"status": "REJECTED", "reason": risk_check.reason}

        # Cap quantity to available balance
        max_qty = self.balance * 0.95 / order.entry_price
        quantity = min(order.quantity, max_qty)

        if quantity < 1:
            return {"status": "REJECTED", "reason": "Insufficient balance"}

        # Apply slippage (simulated)
        fill_price = order.entry_price * (1 + settings.VIRTUAL_SLIPPAGE_PCT / 100)
        cost = fill_price * quantity + settings.VIRTUAL_BROKERAGE_PER_TRADE

        self.balance -= cost

        # Create position
        position = OpenPosition(
            order_id=order.order_id,
            ticker=order.ticker,
            symbol=order.symbol,
            direction=order.direction,
            quantity=quantity,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=order.stop_loss,
            target1=order.target1,
            target2=order.target2,
            target3=order.target3,
            strategy_name=order.strategy_name,
            confidence=order.confidence,
            entry_time=datetime.utcnow(),
            timeframe=order.timeframe,
            exchange=order.exchange,
            ai_reason=order.ai_reason,
            indicator_snapshot=order.indicator_snapshot,
        )

        self.open_positions[order.order_id] = position

        # Save to DB
        trade_id = self._save_trade_open(position, fill_price, quantity, order)
        if trade_id:
            # Swap order_id in position and open_positions to match DB primary key id
            old_id = order.order_id
            position.order_id = str(trade_id)
            self.open_positions.pop(old_id, None)
            self.open_positions[position.order_id] = position

        logger.info("📈 ORDER FILLED: {} {} {} @ ₹{:.2f} | Qty: {:.0f} | Balance: ₹{:.0f}",
                    order.direction, order.symbol, order.ticker, fill_price, quantity, self.balance)

        return {
            "status": "FILLED",
            "order_id": position.order_id,
            "fill_price": round(fill_price, 2),
            "quantity": round(quantity, 2),
            "cost": round(cost, 2),
        }

    # ─── Price Update (called by scanner) ─────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Updates current prices for all open positions.
        Auto-triggers SL and target exits.
        Returns list of any closed trades.
        """
        closed_trades = []
        to_close = []

        for order_id, pos in self.open_positions.items():
            if pos.ticker not in prices:
                continue

            current = prices[pos.ticker]
            pos.current_price = current

            if pos.direction == "BUY":
                pnl = (current - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current) * pos.quantity
            pos.unrealized_pnl = pnl

            # Check exits
            exit_reason = None
            exit_price = current

            # Stop Loss
            if pos.direction == "BUY" and current <= pos.stop_loss:
                exit_reason = "STOP_LOSS"
            elif pos.direction == "SELL" and current >= pos.stop_loss:
                exit_reason = "STOP_LOSS"

            # Target 3 (full exit)
            elif pos.direction == "BUY" and current >= pos.target3:
                exit_reason = "TARGET3"
            elif pos.direction == "SELL" and current <= pos.target3:
                exit_reason = "TARGET3"

            # Target 1 (partial: 50% exit)
            elif pos.direction == "BUY" and current >= pos.target1 and not pos.target1_hit:
                pos.target1_hit = True
                closed = self._partial_exit(pos, current, 0.5, "TARGET1")
                closed_trades.append(closed)
            elif pos.direction == "SELL" and current <= pos.target1 and not pos.target1_hit:
                pos.target1_hit = True
                closed = self._partial_exit(pos, current, 0.5, "TARGET1")
                closed_trades.append(closed)

            # Target 2 (partial: remaining 50%)
            elif pos.direction == "BUY" and current >= pos.target2 and pos.target1_hit and not pos.target2_hit:
                pos.target2_hit = True
                closed = self._partial_exit(pos, current, 0.5, "TARGET2")
                closed_trades.append(closed)
            elif pos.direction == "SELL" and current <= pos.target2 and pos.target1_hit and not pos.target2_hit:
                pos.target2_hit = True
                closed = self._partial_exit(pos, current, 0.5, "TARGET2")
                closed_trades.append(closed)

            if exit_reason:
                to_close.append((order_id, exit_price, exit_reason))

        for order_id, exit_price, reason in to_close:
            result = self.close_position(order_id, exit_price, reason)
            closed_trades.append(result)

        return closed_trades

    def close_position(self, order_id: str, exit_price: float,
                       reason: str = "MANUAL") -> Dict:
        """Close a position fully."""
        if order_id not in self.open_positions:
            return {"status": "ERROR", "reason": "Position not found"}

        pos = self.open_positions.pop(order_id)
        brokerage = settings.VIRTUAL_BROKERAGE_PER_TRADE

        if pos.direction == "BUY":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            proceeds = pos.quantity * exit_price - brokerage
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            proceeds = (2 * pos.entry_price - exit_price) * pos.quantity - brokerage

        net_pnl = gross_pnl - brokerage

        self.balance += proceeds
        self.total_pnl += net_pnl
        self.trade_count += 1

        if net_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Update risk manager
        risk_manager.update_daily_pnl(net_pnl)

        # Save to DB
        self._save_trade_close(pos, exit_price, net_pnl, gross_pnl, reason)
        self._save_equity_point()

        icon = "✅" if net_pnl > 0 else "❌"
        logger.info("{} CLOSED: {} @ ₹{:.2f} | PnL: ₹{:.2f} | Reason: {}",
                    icon, pos.ticker, exit_price, net_pnl, reason)

        return {
            "status": "CLOSED",
            "order_id": order_id,
            "ticker": pos.ticker,
            "exit_price": round(exit_price, 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "win": net_pnl > 0,
        }

    def _partial_exit(self, pos: OpenPosition, exit_price: float,
                      fraction: float, reason: str) -> Dict:
        """Exit a fraction of the position."""
        qty = pos.quantity * fraction
        pos.quantity -= qty
        brokerage = settings.VIRTUAL_BROKERAGE_PER_TRADE / 2

        if pos.direction == "BUY":
            gross_pnl = (exit_price - pos.entry_price) * qty
            proceeds = qty * exit_price - brokerage
        else:
            gross_pnl = (pos.entry_price - exit_price) * qty
            proceeds = (2 * pos.entry_price - exit_price) * qty - brokerage

        net_pnl = gross_pnl - brokerage
        self.balance += proceeds
        self.total_pnl += net_pnl

        logger.info("📊 PARTIAL EXIT: {} {} ({:.0f}%) @ ₹{:.2f} | PnL: ₹{:.2f}",
                    pos.symbol, reason, fraction * 100, exit_price, net_pnl)
        return {"status": "PARTIAL", "ticker": pos.ticker, "reason": reason, "net_pnl": round(net_pnl, 2)}

    # ─── Portfolio Stats ──────────────────────────────────────────────────────

    @property
    def total_value(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.open_positions.values())
        return self.balance + unrealized

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return round(self.win_count / self.trade_count * 100, 1)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return round((self.peak_balance - self.total_value) / self.peak_balance * 100, 2)

    @property
    def total_return_pct(self) -> float:
        return round((self.total_value - self.initial_balance) / self.initial_balance * 100, 2)

    def get_stats(self) -> Dict:
        return {
            "initial_balance": self.initial_balance,
            "current_balance": round(self.balance, 2),
            "total_value": round(self.total_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": self.total_return_pct,
            "total_trades": self.trade_count,
            "winning_trades": self.win_count,
            "losing_trades": self.loss_count,
            "win_rate": self.win_rate,
            "drawdown_pct": self.drawdown_pct,
            "open_positions": len(self.open_positions),
            "peak_balance": round(self.peak_balance, 2),
        }

    def get_open_positions(self) -> List[Dict]:
        return [
            {
                "order_id": p.order_id,
                "ticker": p.ticker,
                "symbol": p.symbol,
                "direction": p.direction,
                "quantity": round(p.quantity, 2),
                "entry_price": round(p.entry_price, 2),
                "current_price": round(p.current_price, 2),
                "stop_loss": round(p.stop_loss, 2),
                "target1": round(p.target1, 2),
                "target2": round(p.target2, 2),
                "target3": round(p.target3, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                "strategy": p.strategy_name,
                "entry_time": p.entry_time.isoformat(),
            }
            for p in self.open_positions.values()
        ]

    def _get_sector_exposure(self, ticker: str) -> float:
        """Simplified sector exposure check."""
        return 0.0

    def _load_from_db(self) -> None:
        try:
            with db_session() as db:
                portfolio = db.query(Portfolio).filter_by(id=self.portfolio_id).first()
                if not portfolio:
                    # Seed/create the default portfolio in the database to prevent foreign key errors
                    portfolio = Portfolio(
                        id=self.portfolio_id,
                        name="Virtual Portfolio",
                        initial_balance=self.initial_balance,
                        current_balance=self.balance,
                        total_pnl=self.total_pnl,
                        total_trades=self.trade_count,
                        winning_trades=self.win_count,
                        losing_trades=self.loss_count,
                        win_rate=0.0,
                        peak_balance=self.peak_balance
                    )
                    db.add(portfolio)
                    db.commit()
                    logger.info("Seeded default virtual portfolio (ID: {}) in database", self.portfolio_id)
                else:
                    self.balance = portfolio.current_balance
                    self.initial_balance = portfolio.initial_balance
                    self.peak_balance = portfolio.peak_balance
                    self.total_pnl = portfolio.total_pnl
                    self.trade_count = portfolio.total_trades
                    self.win_count = portfolio.winning_trades
                    self.loss_count = portfolio.losing_trades

                # Load existing open trades from database into memory
                open_trades = db.query(Trade).filter_by(portfolio_id=self.portfolio_id, status="OPEN").all()
                self.open_positions = {}
                for t in open_trades:
                    pos = OpenPosition(
                        order_id=str(t.id),
                        ticker=t.ticker,
                        symbol=t.symbol,
                        direction=t.direction,
                        quantity=t.quantity,
                        entry_price=t.entry_price,
                        current_price=t.entry_price,
                        stop_loss=t.stop_loss,
                        target1=t.target1,
                        target2=t.target2,
                        target3=t.target3,
                        strategy_name=t.strategy_used,
                        confidence=t.confidence_at_entry or 0.5,
                        entry_time=t.entry_time or datetime.utcnow(),
                        timeframe=t.timeframe or "1d",
                        exchange=t.exchange or "NSE",
                        ai_reason=t.ai_reason or "",
                        indicator_snapshot=t.indicator_snapshot,
                    )
                    self.open_positions[pos.order_id] = pos
                logger.info("Loaded {} open positions from database into memory", len(self.open_positions))
        except Exception as e:
            logger.debug("Portfolio DB load: {}", e)

    def _save_trade_open(self, pos, fill_price, quantity, order) -> Optional[int]:
        try:
            with db_session() as db:
                coins_used = fill_price * quantity + settings.VIRTUAL_BROKERAGE_PER_TRADE
                trade = Trade(
                    portfolio_id=self.portfolio_id,
                    ticker=pos.ticker,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    status="OPEN",
                    entry_price=fill_price,
                    quantity=quantity,
                    stop_loss=pos.stop_loss,
                    target1=pos.target1,
                    target2=pos.target2,
                    target3=pos.target3,
                    strategy_used=pos.strategy_name,
                    confidence_at_entry=pos.confidence,
                    entry_time=pos.entry_time,
                    timeframe=getattr(pos, "timeframe", "1d"),
                    exchange=getattr(pos, "exchange", "NSE"),
                    coins_used=round(coins_used, 2),
                    coins_remaining=round(self.balance, 2),
                    ai_reason=getattr(pos, "ai_reason", ""),
                    indicator_snapshot=getattr(pos, "indicator_snapshot", None),
                )
                db.add(trade)
                db.commit()
                return trade.id
        except Exception as e:
            logger.error("Save trade open error: {}", e)
            return None

    def _save_trade_close(self, pos, exit_price, net_pnl, gross_pnl, reason) -> None:
        try:
            with db_session() as db:
                trade = None
                try:
                    # Query trade by database primary key ID directly
                    trade_id = int(pos.order_id)
                    trade = db.query(Trade).filter_by(id=trade_id).first()
                except (ValueError, TypeError):
                    pass

                if not trade:
                    trade = db.query(Trade).filter_by(
                        ticker=pos.ticker, status="OPEN"
                    ).order_by(Trade.entry_time.desc()).first()

                if trade:
                    trade.status = "CLOSED"
                    trade.exit_price = exit_price
                    trade.exit_reason = reason
                    trade.gross_pnl = round(gross_pnl, 2)
                    trade.brokerage = settings.VIRTUAL_BROKERAGE_PER_TRADE
                    trade.net_pnl = round(net_pnl, 2)
                    trade.pnl_pct = round(net_pnl / (pos.entry_price * pos.quantity) * 100, 2)
                    trade.exit_time = datetime.utcnow()
                    duration = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
                    trade.holding_minutes = int(duration)
                    trade.coins_remaining = round(self.balance, 2)

                # Update portfolio
                portfolio = db.query(Portfolio).filter_by(id=self.portfolio_id).first()
                if portfolio:
                    portfolio.current_balance = round(self.balance, 2)
                    portfolio.total_pnl = round(self.total_pnl, 2)
                    portfolio.total_trades = self.trade_count
                    portfolio.winning_trades = self.win_count
                    portfolio.losing_trades = self.loss_count
                    portfolio.win_rate = self.win_rate
                    portfolio.peak_balance = round(self.peak_balance, 2)
        except Exception as e:
            logger.debug("Save trade close error: {}", e)

    def _save_equity_point(self) -> None:
        try:
            with db_session() as db:
                point = EquityCurvePoint(
                    portfolio_id=self.portfolio_id,
                    balance=round(self.total_value, 2),
                    drawdown_pct=self.drawdown_pct,
                    daily_pnl=round(risk_manager.daily_pnl, 2),
                )
                db.add(point)
        except Exception as e:
            logger.debug("Equity point save error: {}", e)


virtual_portfolio = VirtualPortfolio()

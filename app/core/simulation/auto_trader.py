"""
Autonomous Paper Trading & Smart Trade Manager
=================================================
Monitors market scan signals, places simulated paper orders,
and manages active positions with trailing stops and early exits.

Key Feature: PENDING ZONE ORDERS
---------------------------------
Instead of buying immediately at the alert price, the auto-trader
defines an optimal entry zone (e.g., pullback to EMA, support level).
The trade is only executed when the live price actually enters that zone.

This avoids chasing prices and significantly improves entry quality,
which is the single biggest factor for trade profitability.

Self-Learning Integration
--------------------------
After every closed trade the system:
1. Updates per-strategy win rates in the database
2. Analyses which entry zones produced winners vs losers
3. Adjusts strategy weights via SelfLearningEngine
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from loguru import logger
import pandas as pd

from app.config import settings
from app.db.database import db_session
from app.db.models import Trade, PatternPerformance
from app.core.simulation.virtual_portfolio import virtual_portfolio, SimulatedOrder
from app.data.historical_data import historical_data
from app.core.analysis.indicators import compute_all_indicators
from app.core.strategies.strategy_manager import signal_generator
from app.ai.self_learning import self_learning


# ─── Pending Zone Order ───────────────────────────────────────────────────────

@dataclass
class PendingZoneOrder:
    """
    Represents a pending order that waits for price to enter the entry zone.
    If the price never reaches the zone before expiry, the order is cancelled.
    """
    ticker: str
    symbol: str
    direction: str
    entry_zone_low: float    # Buy zone: price must fall IN this range
    entry_zone_high: float   # Buy zone: price must fall IN this range
    stop_loss: float
    target1: float
    target2: float
    target3: float
    strategy_name: str
    confidence: float
    timeframe: str
    exchange: str
    ai_reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=settings.ZONE_ORDER_EXPIRY_HOURS))
    alert_entry_price: float = 0.0   # The original alert price (before zone calculation)
    indicator_snapshot: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def time_remaining_min(self) -> float:
        delta = self.expires_at - datetime.utcnow()
        return round(delta.total_seconds() / 60, 1)

    def price_in_zone(self, price: float) -> bool:
        """Returns True if the given price is within the entry zone."""
        return self.entry_zone_low <= price <= self.entry_zone_high


class AutoTrader:
    """
    Autonomous trading agent that executes simulated orders in the background.
    Uses pending zone orders to enter only at optimal price levels.
    """

    def __init__(self):
        self._is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str, dict], None]] = []
        self._status = "IDLE"
        self._last_run_time: Optional[datetime] = None
        # Pending zone orders: {ticker: PendingZoneOrder}
        self._pending_zone_orders: Dict[str, PendingZoneOrder] = {}

    def register_callback(self, callback: Callable[[str, dict], None]) -> None:
        """Register a callback for websocket notifications."""
        self._callbacks.append(callback)

    def trigger_event(self, event_type: str, data: dict) -> None:
        """Helper to fire callbacks safely."""
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error("Error in auto trader callback: {}", e)

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_run_time(self) -> Optional[datetime]:
        return self._last_run_time

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def pending_zone_orders(self) -> Dict[str, dict]:
        """Returns pending zone orders as serializable dicts."""
        result = {}
        for ticker, pzo in self._pending_zone_orders.items():
            result[ticker] = {
                "ticker": pzo.ticker,
                "symbol": pzo.symbol,
                "direction": pzo.direction,
                "entry_zone_low": pzo.entry_zone_low,
                "entry_zone_high": pzo.entry_zone_high,
                "stop_loss": pzo.stop_loss,
                "target1": pzo.target1,
                "target2": pzo.target2,
                "target3": pzo.target3,
                "strategy": pzo.strategy_name,
                "confidence": round(pzo.confidence * 100, 1),
                "alert_price": pzo.alert_entry_price,
                "created_at": pzo.created_at.isoformat(),
                "expires_at": pzo.expires_at.isoformat(),
                "minutes_remaining": pzo.time_remaining_min,
            }
        return result

    def start(self) -> None:
        """Starts the autonomous auto-trader monitor loop."""
        if self._is_running:
            return

        self._is_running = True
        self._status = "RUNNING"
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🤖 Autonomous Paper Trader started successfully.")

    def stop(self) -> None:
        """Stops the loops."""
        self._is_running = False
        self._status = "STOPPED"
        logger.info("🤖 Autonomous Paper Trader stopped.")

    # ─── Signals Callback (connected to scanner) ──────────────────────────────

    def handle_scanner_signal(self, scan_result: dict) -> None:
        """
        Receives actionable setups from the scanner.
        Instead of placing an immediate order, creates a PENDING ZONE ORDER.
        The trade will only execute when the live price enters the entry zone.
        """
        if not self._is_running:
            return

        ticker = scan_result.get("ticker")
        direction = scan_result.get("direction")

        # Skip if already in an open position for this ticker
        existing = [p for p in virtual_portfolio.open_positions.values() if p.ticker == ticker]
        if existing:
            return

        # Skip if there's already a pending zone order for this ticker
        if ticker in self._pending_zone_orders:
            return

        alert_entry = float(scan_result.get("entry", 0.0))
        alert_sl    = float(scan_result.get("stop_loss", 0.0))
        alert_t1    = float(scan_result.get("target1", 0.0))
        alert_t2    = float(scan_result.get("target2", 0.0))
        alert_t3    = float(scan_result.get("target3", 0.0))

        # ── Validate R:R from scan result ───────────────────────────────────
        risk = abs(alert_entry - alert_sl)
        reward = abs(alert_t1 - alert_entry)
        rr = reward / risk if risk > 0 else 0.0

        if rr < settings.MIN_RISK_REWARD:
            logger.debug("🚫 AutoTrader: Skipping {} — R:R {:.2f} below minimum {:.1f}",
                         ticker, rr, settings.MIN_RISK_REWARD)
            return

        # ── Get entry zone from scan result, or derive it ──────────────────
        zone_low  = float(scan_result.get("entry_zone_low",  0.0))
        zone_high = float(scan_result.get("entry_zone_high", 0.0))

        if zone_low <= 0 or zone_high <= 0 or zone_low >= zone_high:
            # Fallback: derive zone from alert price using config buffer
            buf = settings.ENTRY_ZONE_BUFFER_PCT / 100
            if direction == "BUY":
                zone_low  = round(alert_entry * (1 - buf), 2)
                zone_high = round(alert_entry * (1 + buf * 0.5), 2)
            else:
                zone_low  = round(alert_entry * (1 - buf * 0.5), 2)
                zone_high = round(alert_entry * (1 + buf), 2)

        # ── Create pending zone order ────────────────────────────────────────
        pzo = PendingZoneOrder(
            ticker=ticker,
            symbol=scan_result.get("symbol", ticker.replace(".NS", "")),
            direction=direction,
            entry_zone_low=zone_low,
            entry_zone_high=zone_high,
            stop_loss=alert_sl,
            target1=alert_t1,
            target2=alert_t2,
            target3=alert_t3,
            strategy_name=scan_result.get("signals", ["Composite Consensus"])[0],
            confidence=float(scan_result.get("confidence", 0.0)) / 100.0,
            timeframe=scan_result.get("timeframe", "1d"),
            exchange=scan_result.get("exchange", "NSE"),
            ai_reason=f"Zone order: {', '.join(scan_result.get('signals', []))}. Confidence: {scan_result.get('confidence')}%.",
            alert_entry_price=alert_entry,
            indicator_snapshot={"candle_pattern": scan_result.get("pattern")},
        )

        self._pending_zone_orders[ticker] = pzo
        logger.info(
            "📋 AutoTrader: Zone Order created for {} {} | Zone: ₹{:.2f}–₹{:.2f} | R:R {:.2f} | Expires in {:.0f} min",
            direction, ticker, zone_low, zone_high, rr, pzo.time_remaining_min
        )

        # Broadcast zone order created
        self.trigger_event("ZONE_ORDER_CREATED", {
            "ticker": ticker,
            "symbol": pzo.symbol,
            "direction": direction,
            "entry_zone_low": zone_low,
            "entry_zone_high": zone_high,
            "alert_price": alert_entry,
            "stop_loss": alert_sl,
            "target1": alert_t1,
            "confidence": round(pzo.confidence * 100, 1),
            "risk_reward": round(rr, 2),
            "strategy": pzo.strategy_name,
            "expires_in_min": pzo.time_remaining_min,
        })

    def _execute_pending_zone_order(self, pzo: PendingZoneOrder, execution_price: float) -> None:
        """
        Called when live price enters the zone. Places the actual simulated order.
        Re-scales SL and targets relative to the actual execution price.
        """
        # Re-scale SL and targets proportionally from alert price → execution price
        if pzo.alert_entry_price and pzo.alert_entry_price > 0:
            scale = execution_price / pzo.alert_entry_price
            if pzo.direction == "BUY":
                sl = execution_price - abs(execution_price - pzo.stop_loss * scale)
                t1 = execution_price + abs(pzo.target1 * scale - execution_price)
                t2 = execution_price + abs(pzo.target2 * scale - execution_price)
                t3 = execution_price + abs(pzo.target3 * scale - execution_price)
            else:
                sl = execution_price + abs(pzo.stop_loss * scale - execution_price)
                t1 = execution_price - abs(execution_price - pzo.target1 * scale)
                t2 = execution_price - abs(execution_price - pzo.target2 * scale)
                t3 = execution_price - abs(execution_price - pzo.target3 * scale)
        else:
            sl, t1, t2, t3 = pzo.stop_loss, pzo.target1, pzo.target2, pzo.target3

        # Final R:R check after rescaling
        risk = abs(execution_price - sl)
        reward = abs(t1 - execution_price)
        final_rr = reward / risk if risk > 0 else 0.0
        if final_rr < settings.MIN_RISK_REWARD:
            logger.warning("⚠️ Zone order for {} has R:R {:.2f} after rescaling — skipping", pzo.ticker, final_rr)
            return

        # Calculate quantity from risk
        bal = virtual_portfolio.balance
        risk_amt = bal * (settings.DEFAULT_RISK_PER_TRADE_PCT / 100)
        qty = max(1.0, round(risk_amt / risk)) if risk > 0 else 10.0

        order = SimulatedOrder(
            order_id=str(uuid.uuid4()),
            ticker=pzo.ticker,
            symbol=pzo.symbol,
            direction=pzo.direction,
            quantity=qty,
            entry_price=execution_price,
            stop_loss=round(sl, 2),
            target1=round(t1, 2),
            target2=round(t2, 2),
            target3=round(t3, 2),
            strategy_name=pzo.strategy_name,
            confidence=pzo.confidence,
            timeframe=pzo.timeframe,
            exchange=pzo.exchange,
            ai_reason=(
                f"{pzo.ai_reason} | Zone filled at live ₹{execution_price:.2f} "
                f"(alert was ₹{pzo.alert_entry_price:.2f})"
            ),
            indicator_snapshot=pzo.indicator_snapshot,
        )

        res = virtual_portfolio.place_order(order)
        if res.get("status") == "FILLED":
            logger.success(
                "✅ Zone Order FILLED: {} {} @ ₹{:.2f} | Zone was ₹{:.2f}–₹{:.2f}",
                pzo.direction, pzo.ticker, execution_price, pzo.entry_zone_low, pzo.entry_zone_high
            )
            self.trigger_event("TRADE_OPENED", {
                "ticker": order.ticker,
                "symbol": order.symbol,
                "direction": order.direction,
                "entry_price": res.get("fill_price"),
                "stop_loss": order.stop_loss,
                "target1": order.target1,
                "target2": order.target2,
                "target3": order.target3,
                "confidence": round(order.confidence * 100, 1),
                "strategy": order.strategy_name,
                "reason": f"Zone entry triggered at ₹{execution_price:.2f}",
                "qty": res.get("quantity"),
                "balance": round(virtual_portfolio.balance, 2),
                "zone_entry": True,
                "alert_price": pzo.alert_entry_price,
            })
        else:
            logger.warning("Zone order rejected for {}: {}", pzo.ticker, res.get("reason"))

    # ─── Background Monitor & Exits Loop ──────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Loop that runs regularly to manage active positions and pending zone orders."""
        while self._is_running:
            try:
                self._last_run_time = datetime.utcnow()

                # 1. Check pending zone orders — execute if price entered zone
                self._check_pending_zone_orders()

                # 2. Manage active open positions (SL/TP, trailing, smart exits)
                open_positions = list(virtual_portfolio.open_positions.values())
                if open_positions:
                    prices = {}
                    for pos in open_positions:
                        df = historical_data.get_candles(pos.ticker, pos.timeframe, periods=50)
                        if not df.empty:
                            last_close = df["close"].iloc[-1]
                            prices[pos.ticker] = last_close
                            self._apply_smart_trade_rules(pos, df, last_close)

                    closed_list = virtual_portfolio.update_prices(prices)
                    for closed in closed_list:
                        if closed.get("status") == "CLOSED":
                            self._handle_trade_closed(closed)
                        elif closed.get("status") == "PARTIAL":
                            self._handle_trade_partial(closed)

                time.sleep(15)  # Run checks every 15 seconds
            except Exception as e:
                logger.error("AutoTrader loop error: {}", e)
                time.sleep(5)  # Backoff

    def _check_pending_zone_orders(self) -> None:
        """
        Checks all pending zone orders against the current live price.
        Executes orders where price is in zone; cancels expired ones.
        """
        expired = []
        to_execute = []

        for ticker, pzo in self._pending_zone_orders.items():
            # Skip if already in a position
            existing = [p for p in virtual_portfolio.open_positions.values() if p.ticker == ticker]
            if existing:
                expired.append(ticker)
                continue

            if pzo.is_expired:
                expired.append(ticker)
                logger.info("⏰ Zone order expired for {} — no fill in zone ₹{:.2f}–₹{:.2f}",
                            ticker, pzo.entry_zone_low, pzo.entry_zone_high)
                self.trigger_event("ZONE_ORDER_EXPIRED", {
                    "ticker": ticker,
                    "symbol": pzo.symbol,
                    "direction": pzo.direction,
                    "entry_zone_low": pzo.entry_zone_low,
                    "entry_zone_high": pzo.entry_zone_high,
                    "alert_price": pzo.alert_entry_price,
                    "reason": "Expired — price never reached entry zone",
                })
                continue

            # Fetch live price
            live_price = historical_data.get_live_price(ticker)
            if live_price and live_price > 0:
                if pzo.price_in_zone(live_price):
                    to_execute.append((pzo, live_price))
                    expired.append(ticker)  # Remove from pending after execution
                    logger.info(
                        "🎯 Zone triggered! {} {} live ₹{:.2f} is in zone ₹{:.2f}–₹{:.2f}",
                        pzo.direction, ticker, live_price, pzo.entry_zone_low, pzo.entry_zone_high
                    )

        # Remove expired/filled
        for ticker in expired:
            self._pending_zone_orders.pop(ticker, None)

        # Execute zone fills
        for pzo, price in to_execute:
            self._execute_pending_zone_order(pzo, price)

    def _apply_smart_trade_rules(self, pos, df: pd.DataFrame, current_price: float) -> None:
        """
        Evaluate position against active trailing stops, trend changes,
        volatility bands, and strategy drops.
        """
        try:
            indicators = compute_all_indicators(df)

            # 1. Trailing stops to break-even once Target 1 is hit
            if pos.target1_hit and pos.stop_loss != pos.entry_price:
                old_sl = pos.stop_loss
                pos.stop_loss = pos.entry_price
                logger.info("🛡️ Smart Trail: Trailed SL for {} from ₹{:.2f} to Entry break-even ₹{:.2f}",
                            pos.symbol, old_sl, pos.entry_price)
                self.trigger_event("SMART_MANAGEMENT", {
                    "symbol": pos.symbol,
                    "action": "SL_TRAILED_BREAK_EVEN",
                    "reason": "Target 1 hit. Stop loss moved to entry break-even to protect capital."
                })

            # 2. Trailing to Target 1 once Target 2 is hit
            if pos.target2_hit and pos.stop_loss != pos.target1:
                old_sl = pos.stop_loss
                pos.stop_loss = pos.target1
                logger.info("🛡️ Smart Trail: Trailed SL for {} from ₹{:.2f} to Target 1 ₹{:.2f}",
                            pos.symbol, old_sl, pos.target1)
                self.trigger_event("SMART_MANAGEMENT", {
                    "symbol": pos.symbol,
                    "action": "SL_TRAILED_T1",
                    "reason": "Target 2 hit. Stop loss moved to Target 1 to lock in profits."
                })

            # 3. Calculate current P&L percentage to prevent exiting on minor drawdowns
            pnl_pct = 0.0
            if pos.direction == "BUY":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            elif pos.direction == "SELL":
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

            # 4. Trend Reversal Check (EMA 20 crossing EMA 50 opposite)
            # Only trigger trend reversal or volatility exits if the position is in profit (to lock in gains).
            # If the position is at a loss, let it run to the stop-loss to give it breathing room.
            ema_20 = indicators.get("ema_20", 0)
            ema_50 = indicators.get("ema_50", 0)

            if pnl_pct >= 0:
                if pos.direction == "BUY" and ema_20 < ema_50 and ema_50 > 0:
                    logger.info("🛡️ Smart Exit: Trend reversal detected for {} in profit ({:.2f}%). Closing position early.", pos.symbol, pnl_pct)
                    virtual_portfolio.close_position(pos.order_id, current_price, "AI_EXIT_TREND_REVERSAL")
                    return

                if pos.direction == "SELL" and ema_20 > ema_50 and ema_50 > 0:
                    logger.info("🛡️ Smart Exit: Trend reversal detected for {} in profit ({:.2f}%). Closing position early.", pos.symbol, pnl_pct)
                    virtual_portfolio.close_position(pos.order_id, current_price, "AI_EXIT_TREND_REVERSAL")
                    return

                # 5. Volatility Exit: Bollinger Bands breakdown
                bb_lower = indicators.get("bb_lower", 0.0)
                if pos.direction == "BUY" and current_price < bb_lower and bb_lower > 0:
                    logger.info("🛡️ Smart Exit: Volatility breakdown for {} in profit ({:.2f}%). Closing position.", pos.symbol, pnl_pct)
                    virtual_portfolio.close_position(pos.order_id, current_price, "AI_EXIT_VOLATILITY")
                    return

        except Exception as e:
            logger.debug("Error applying smart trade rules for {}: {}", pos.symbol, e)

    def _handle_trade_closed(self, closed_result: dict) -> None:
        """
        Processes a completed trade: retrains models, registers patterns success,
        and broadcasts alerts.
        """
        ticker = closed_result.get("ticker")
        net_pnl = closed_result.get("net_pnl", 0.0)
        win = closed_result.get("win", False)

        logger.success("🤖 AutoTrader: Trade closed for {}. P&L: ₹{:.2f}", ticker, net_pnl)

        try:
            with db_session() as db:
                trade = db.query(Trade).filter_by(
                    ticker=ticker, status="CLOSED"
                ).order_by(Trade.exit_time.desc()).first()

                if trade:
                    trade_id = trade.id
                    strategy_name = trade.strategy_used

                    # Update pattern performance database
                    pattern = trade.indicator_snapshot.get("candle_pattern") if trade.indicator_snapshot else None
                    if pattern:
                        self._update_pattern_stats(pattern, "CANDLESTICK", win, net_pnl, trade.holding_minutes)

                    # Trigger learning engine weights updates
                    self_learning.process_trade_result(trade_id)

                    notify_data = {
                        "ticker": trade.ticker,
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "net_pnl": net_pnl,
                        "pnl_pct": trade.pnl_pct,
                        "duration": trade.holding_minutes,
                        "exit_reason": trade.exit_reason,
                        "balance": round(virtual_portfolio.balance, 2),
                        "win_rate": virtual_portfolio.win_rate,
                        "strategy": strategy_name
                    }
                    self.trigger_event("TRADE_CLOSED", notify_data)
        except Exception as e:
            logger.error("Error in trade post-closing logic: {}", e)

    def _handle_trade_partial(self, partial_result: dict) -> None:
        """Processes a partial target exit and broadcasts alerts."""
        try:
            ticker = partial_result.get("ticker")
            net_pnl = partial_result.get("net_pnl", 0.0)
            reason = partial_result.get("reason", "PARTIAL")

            logger.success("🤖 AutoTrader: Partial profit booked for {}. P&L: ₹{:.2f}", ticker, net_pnl)

            with db_session() as db:
                trade = db.query(Trade).filter_by(ticker=ticker, status="OPEN").first()
                symbol = trade.symbol if trade else ticker.split(".")[0]
                direction = trade.direction if trade else "BUY"
                strategy_name = trade.strategy_used if trade else "Strategy"

            notify_data = {
                "ticker": ticker,
                "symbol": symbol,
                "direction": direction,
                "net_pnl": net_pnl,
                "reason": reason,
                "balance": round(virtual_portfolio.balance, 2),
                "win_rate": virtual_portfolio.win_rate,
                "strategy": strategy_name
            }
            self.trigger_event("TRADE_PARTIAL", notify_data)
        except Exception as e:
            logger.error("Error in trade post-partial logic: {}", e)

    def _update_pattern_stats(self, pattern_name: str, pattern_type: str,
                             win: bool, pnl: float, duration: int) -> None:
        """Updates rolling performance counters for a specific pattern."""
        try:
            with db_session() as db:
                perf = db.query(PatternPerformance).filter_by(
                    pattern_name=pattern_name,
                    pattern_type=pattern_type
                ).first()

                if not perf:
                    perf = PatternPerformance(
                        pattern_name=pattern_name,
                        pattern_type=pattern_type
                    )
                    db.add(perf)

                perf.occurrences += 1
                if win:
                    perf.wins += 1
                else:
                    perf.losses += 1

                perf.win_rate = perf.wins / perf.occurrences
                perf.avg_return = (
                    (perf.avg_return * (perf.occurrences - 1) + pnl) / perf.occurrences
                )
                if duration:
                    perf.avg_holding_minutes = (
                        (perf.avg_holding_minutes * (perf.occurrences - 1) + duration) / perf.occurrences
                    )
                perf.success_probability = perf.win_rate

                db.commit()
        except Exception as e:
            logger.debug("Failed to update pattern stats: {}", e)


auto_trader = AutoTrader()

"""
Autonomous Paper Trading & Smart Trade Manager
=================================================
Monitors market scan signals, places simulated paper orders,
and manages active positions with trailing stops and early exits.
"""

import threading
import time
import uuid
from datetime import datetime
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


class AutoTrader:
    """
    Autonomous trading agent that executes simulated orders in the background.
    """

    def __init__(self):
        self._is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str, dict], None]] = []
        self._status = "IDLE"
        self._last_run_time: Optional[datetime] = None

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
        Receives actionable setups from scanner and places orders.
        """
        if not self._is_running:
            return

        ticker = scan_result.get("ticker")
        direction = scan_result.get("direction")
        
        # Check if already in a position
        existing = [p for p in virtual_portfolio.open_positions.values() if p.ticker == ticker]
        if existing:
            return

        logger.info("🤖 AutoTrader: Setup found for {}. Placing order...", ticker)

        order_id = str(uuid.uuid4())
        qty = float(scan_result.get("volume_ratio", 1.0)) * 10
        if qty < 1:
            qty = 10.0

        # Calculate quantities based on target risk (1% of balance by default)
        bal = virtual_portfolio.balance
        risk_amt = bal * (settings.VIRTUAL_RISK_PER_TRADE / 100)
        entry = float(scan_result.get("entry", 0.0))
        sl = float(scan_result.get("stop_loss", 0.0))
        
        risk_diff = abs(entry - sl)
        if risk_diff > 0:
            qty = max(1.0, round(risk_amt / risk_diff))
        
        order = SimulatedOrder(
            order_id=order_id,
            ticker=ticker,
            symbol=scan_result.get("symbol"),
            direction=direction,
            quantity=qty,
            entry_price=entry,
            stop_loss=sl,
            target1=float(scan_result.get("target1", 0.0)),
            target2=float(scan_result.get("target2", 0.0)),
            target3=float(scan_result.get("target3", 0.0)),
            strategy_name=scan_result.get("signals")[0] if scan_result.get("signals") else "Composite Consensus",
            confidence=float(scan_result.get("confidence", 0.0)) / 100.0,
            timeframe=scan_result.get("timeframe", "1d"),
            exchange=scan_result.get("exchange", "NSE"),
            ai_reason=f"Auto setup: {', '.join(scan_result.get('signals', []))}. Confidence: {scan_result.get('confidence')}%"
        )

        res = virtual_portfolio.place_order(order)
        if res.get("status") == "FILLED":
            # Fire notifications event
            notify_data = {
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
                "reason": order.ai_reason,
                "qty": res.get("quantity"),
                "balance": round(virtual_portfolio.balance, 2)
            }
            self.trigger_event("TRADE_OPENED", notify_data)

    # ─── Background Monitor & Exits Loop ──────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Loop that runs regularly to manage active positions."""
        while self._is_running:
            try:
                self._last_run_time = datetime.utcnow()
                open_positions = list(virtual_portfolio.open_positions.values())

                if open_positions:
                    # Gather prices
                    prices = {}
                    for pos in open_positions:
                        df = historical_data.get_candles(pos.ticker, pos.timeframe, periods=50)
                        if not df.empty:
                            last_close = df["close"].iloc[-1]
                            prices[pos.ticker] = last_close
                            
                            # Manage smart rules
                            self._apply_smart_trade_rules(pos, df, last_close)

                    # Update prices in portfolio (triggers standard SL/TP hits)
                    closed_list = virtual_portfolio.update_prices(prices)

                    for closed in closed_list:
                        if closed.get("status") == "CLOSED":
                            self._handle_trade_closed(closed)

                time.sleep(15)  # Run checks every 15 seconds
            except Exception as e:
                logger.error("AutoTrader loop error: {}", e)
                time.sleep(5)  # Backoff

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

            # 3. Extend Target 3 under strong trend (ADX > 35)
            adx = indicators.get("adx", 20)
            atr = indicators.get("atr", 1.0)
            if adx > 35 and current_price > pos.entry_price:
                # Extend target 3 slightly if not already done
                pass

            # 4. Trend Reversal Check (EMA 20 crossing EMA 50 opposite)
            ema_20 = indicators.get("ema_20", 0)
            ema_50 = indicators.get("ema_50", 0)
            
            if pos.direction == "BUY" and ema_20 < ema_50 and ema_50 > 0:
                logger.info("🛡️ Smart Exit: Trend reversal detected for {}. Closing position early.", pos.symbol)
                virtual_portfolio.close_position(pos.order_id, current_price, "AI_EXIT_TREND_REVERSAL")
                return

            if pos.direction == "SELL" and ema_20 > ema_50 and ema_50 > 0:
                logger.info("🛡️ Smart Exit: Trend reversal detected for {}. Closing position early.", pos.symbol)
                virtual_portfolio.close_position(pos.order_id, current_price, "AI_EXIT_TREND_REVERSAL")
                return

            # 5. Volatility Exit: Bollinger Bands breakdown
            # Buy exit if price drops below lower BB
            bb_lower = indicators.get("bb_lower", 0.0)
            if pos.direction == "BUY" and current_price < bb_lower and bb_lower > 0:
                logger.info("🛡️ Smart Exit: Volatility breakdown for {}. Closing position.", pos.symbol)
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

        # 1. Update Self-Learning strategy weights
        try:
            with db_session() as db:
                # Find matching trade record to get ID
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

                    # Send websocket confirmation details
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

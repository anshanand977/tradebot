"""
FastAPI Application — AI Trading Analyst
==========================================
Local REST API running on http://127.0.0.1:8765
Serves the frontend and exposes all backend functionality via API endpoints.
WebSocket support for real-time scanner updates.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from app.config import settings, FRONTEND_DIR, LOGS_DIR
from app.db.database import init_db
from app.db.models import Signal, Trade, AppSetting
from app.db.database import db_session
from app.data.nse_client import nse_client
from app.data.historical_data import historical_data
from app.core.strategies.strategy_manager import signal_generator, strategy_manager
from app.core.simulation.virtual_portfolio import virtual_portfolio, SimulatedOrder
from app.core.scanner.market_scanner import market_scanner
from app.ai.ollama_client import ollama_client
from app.ai.self_learning import self_learning
from app.core.risk.risk_manager import risk_manager
from app.ai.news_sentiment import news_sentiment_analyst


from apscheduler.schedulers.background import BackgroundScheduler


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Configure file logging
    log_file = LOGS_DIR / "tradebot.log"
    logger.add(
        log_file,
        rotation=settings.LOG_ROTATION,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    logger.info("📁 File logging configured at: {}", log_file)
    logger.info("🚀 Starting AI Trading Analyst v{}", settings.APP_VERSION)
    init_db()

    def load_db_settings():
        try:
            with db_session() as db:
                db_settings = db.query(AppSetting).all()
                for s in db_settings:
                    val = s.value
                    if s.value_type == "int":
                        val = int(val)
                    elif s.value_type == "float":
                        val = float(val)
                    elif s.value_type == "bool":
                        val = val.lower() == "true"
                    
                    if s.key == "initial_balance":
                        settings.VIRTUAL_INITIAL_BALANCE = val
                    elif s.key == "risk_per_trade_pct":
                        settings.DEFAULT_RISK_PER_TRADE_PCT = val
                    elif s.key == "min_strategy_agreement":
                        settings.MIN_STRATEGY_AGREEMENT = val
                logger.info("Loaded custom settings from database")
        except Exception as e:
            logger.debug("Failed to load settings from DB: {}", e)

    load_db_settings()
    historical_data.sync_symbols_to_db()
    logger.info("✅ Database initialized")

    # Load learned weights into strategy manager
    weights = self_learning.get_strategy_weights()
    if weights:
        strategy_manager.apply_weights_from_db(weights)
        logger.info("📚 Loaded {} learned strategy weights", len(weights))

    # Import and configure auto trader & research lab
    from app.core.simulation.auto_trader import auto_trader
    from app.core.research.research_lab import research_engine

    # Callback to push auto trader alerts through websocket broadcast
    loop = asyncio.get_running_loop()
    def autotrader_ws_cb(event_type: str, data: dict):
        try:
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": event_type,
                    "data": data
                }),
                loop
            )
        except Exception as e:
            logger.debug("Failed to broadcast WS event: {}", e)

    auto_trader.register_callback(autotrader_ws_cb)
    market_scanner.add_signal_callback(auto_trader.handle_scanner_signal)

    # Start autonomous engines
    auto_trader.start()
    research_engine.start()

    # Start background scheduler for automatic scanning
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=market_scanner.run_full_scan,
        trigger="interval",
        seconds=settings.SCANNER_INTERVAL_SECONDS,
        id="market_scanner_job"
    )
    scheduler.start()
    logger.info("⏰ Background scheduler started (scan interval: {}s)", settings.SCANNER_INTERVAL_SECONDS)

    yield

    # Stop engines on shutdown
    auto_trader.stop()
    research_engine.stop()
    scheduler.shutdown()
    logger.info("⏰ Background scheduler stopped")
    logger.info("👋 Shutting down AI Trading Analyst")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered offline trading analyst for Indian markets",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (only if directory exists)
_static_dir = FRONTEND_DIR / "static"
if _static_dir.exists() and any(_static_dir.iterdir()):
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ─── WebSocket Manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: Dict):
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# ─── Request/Response Models ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str
    timeframe: str = "1d"

class OrderRequest(BaseModel):
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    strategy_name: str
    confidence: float = 0.0
    timeframe: str = "1d"
    exchange: str = "NSE"
    ai_reason: Optional[str] = ""

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[Dict]] = None
    ticker: Optional[str] = None
    model: Optional[str] = None

class ScanRequest(BaseModel):
    timeframe: str = "1d"
    universe: Optional[List[str]] = None
    max_stocks: int = 50

class ClosePositionRequest(BaseModel):
    order_id: str
    exit_price: Optional[float] = None


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Frontend not found. Run setup to generate frontend.</h1>")


# ─── System ───────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "ollama": ollama_client.get_status(),
        "risk": risk_manager.get_status(),
        "scanner": {
            "is_running": market_scanner.is_running,
            "last_scan": market_scanner.last_scan_time.isoformat() if market_scanner.last_scan_time else None,
            "scan_count": market_scanner.scan_count,
        },
        "portfolio": virtual_portfolio.get_stats(),
    }


# ─── Market Data ──────────────────────────────────────────────────────────────

@app.get("/api/candles/{ticker}")
async def get_candles(
    ticker: str,
    timeframe: str = Query("1d"),
    periods: int = Query(200),
):
    """Returns OHLCV candles for a ticker."""
    resolved_ticker = nse_client.get_ticker(ticker)
    df = historical_data.get_candles(resolved_ticker, timeframe, periods=periods, force_refresh=False)
    if df.empty:
        raise HTTPException(404, f"No data for {resolved_ticker}/{timeframe}")

    records = []
    for ts, row in df.iterrows():
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            t_val = int(ts.tz_convert("UTC").timestamp())
        elif hasattr(ts, "tz_localize"):
            t_val = int(ts.tz_localize("UTC").timestamp())
        elif hasattr(ts, "timestamp"):
            import pytz
            t_val = int(pytz.utc.localize(ts).timestamp())
        else:
            t_val = 0

        records.append({
            "t": t_val,
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "c": round(float(row["close"]), 2),
            "v": int(row["volume"]),
        })

    return {"ticker": resolved_ticker, "timeframe": timeframe, "candles": records}


@app.get("/api/symbols")
async def get_symbols(exchange: str = "NSE", limit: int = 100):
    """Returns the stock universe."""
    symbols = nse_client.get_symbol_universe()[:limit]
    return {"symbols": symbols, "total": len(symbols)}


@app.get("/api/indices")
async def get_indices():
    """Returns major index data."""
    indices = {}
    for name, ticker in [("NIFTY50", "^NSEI"), ("SENSEX", "^BSESN"), ("BANKNIFTY", "^NSEBANK")]:
        df = historical_data.get_candles(ticker, "1d", periods=2)
        if not df.empty:
            close = df["close"].iloc[-1]
            prev = df["close"].iloc[-2] if len(df) > 1 else close
            indices[name] = {
                "value": round(close, 2),
                "change": round(close - prev, 2),
                "change_pct": round((close - prev) / prev * 100, 2) if prev else 0,
            }
    return {"indices": indices}


def make_json_serializable(val):
    import numpy as np
    if isinstance(val, dict):
        return {k: make_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [make_json_serializable(v) for v in val]
    elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, np.ndarray):
        return make_json_serializable(val.tolist())
    else:
        return val


@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    """Full multi-strategy analysis for a single stock."""
    ticker = req.ticker if "." in req.ticker else f"{req.ticker}.NS"
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    df = historical_data.get_candles(ticker, req.timeframe, periods=300)
    if df.empty:
        # Try downloading first
        historical_data._download_and_store(ticker, req.timeframe)
        df = historical_data.get_candles(ticker, req.timeframe, periods=300)

    if df.empty:
        raise HTTPException(404, f"No data for {ticker}. Data may need downloading.")

    rec = signal_generator.generate(
        df=df, ticker=ticker, symbol=symbol, exchange="NSE", timeframe=req.timeframe
    )

    resp_dict = {
        "ticker": rec.ticker,
        "symbol": rec.symbol,
        "direction": rec.direction,
        "confidence": round(float(rec.confidence) * 100, 1) if rec.confidence is not None else 0.0,
        "probability": round(float(rec.probability) * 100, 1) if rec.probability is not None else 0.0,
        "entry_price": float(rec.entry_price) if rec.entry_price is not None else None,
        "stop_loss": float(rec.stop_loss) if rec.stop_loss is not None else None,
        "target1": float(rec.target1) if rec.target1 is not None else None,
        "target2": float(rec.target2) if rec.target2 is not None else None,
        "target3": float(rec.target3) if rec.target3 is not None else None,
        "risk_reward": float(rec.risk_reward) if rec.risk_reward is not None else None,
        "risk_level": rec.risk_level,
        "strategies_agreed": int(rec.strategies_agreed) if rec.strategies_agreed is not None else 0,
        "total_strategies": int(rec.total_strategies_voted) if rec.total_strategies_voted is not None else 0,
        "strategy_votes": rec.strategy_votes,
        "contributing_signals": rec.contributing_signals,
        "pattern": rec.pattern_context,
        "market_regime": rec.market_regime,
        "is_actionable": bool(rec.is_actionable),
        "summary": rec.summary,
        "indicator_snapshot": rec.indicator_snapshot,
        "timestamp": rec.timestamp.isoformat() if rec.timestamp else datetime.utcnow().isoformat(),
    }
    return make_json_serializable(resp_dict)


# ─── Scanner ──────────────────────────────────────────────────────────────────

@app.get("/api/scanner/results")
async def get_scanner_results(
    limit: int = Query(50),
    only_signals: bool = Query(False),
):
    """Get current scanner results."""
    results = market_scanner.get_results(limit=limit, only_signals=only_signals)
    return {
        "results": results,
        "count": len(results),
        "last_scan": market_scanner.last_scan_time.isoformat() if market_scanner.last_scan_time else None,
        "scan_count": market_scanner.scan_count,
    }


@app.post("/api/scanner/run")
async def run_scanner(req: ScanRequest, background_tasks: BackgroundTasks):
    """Trigger a full market scan in the background."""
    def _scan():
        count = market_scanner.run_full_scan()
        logger.info("Scan complete: {} signals", count)

    background_tasks.add_task(_scan)
    return {"status": "started", "message": "Scanner running in background"}


@app.get("/api/scanner/top-signals")
async def get_top_signals(n: int = Query(10)):
    signals = market_scanner.get_top_signals(n=n)
    return {"signals": signals, "count": len(signals)}


# ─── Portfolio ────────────────────────────────────────────────────────────────

@app.get("/api/portfolio/stats")
async def get_portfolio_stats():
    return virtual_portfolio.get_stats()


@app.get("/api/portfolio/positions")
async def get_positions():
    return {"positions": virtual_portfolio.get_open_positions()}


@app.get("/api/portfolio/pending-zone-orders")
async def get_pending_zone_orders():
    from app.core.simulation.auto_trader import auto_trader
    return {"pending_zone_orders": auto_trader.pending_zone_orders}


@app.post("/api/portfolio/order")
async def place_order(req: OrderRequest):
    """Place a simulated order at the CURRENT LIVE market price, not the stale alert price."""
    full_ticker = req.ticker if "." in req.ticker else f"{req.ticker}.NS"
    symbol = full_ticker.replace(".NS", "").replace(".BO", "")

    # ── Step 1: Fetch live market price ───────────────────────────────────────
    live_price = historical_data.get_live_price(full_ticker)
    alert_price = req.entry_price  # Price when the signal was generated

    if live_price and live_price > 0:
        execution_price = live_price
        price_note = f"Filled at live price ₹{live_price:.2f} (alert was ₹{alert_price:.2f})"
        logger.info("🎯 Order execution: {} live=₹{:.2f} alert=₹{:.2f}", full_ticker, live_price, alert_price)
    else:
        execution_price = alert_price
        price_note = f"Filled at alert price ₹{alert_price:.2f} (live price unavailable)"
        logger.warning("⚠️ Falling back to alert price for {}: ₹{:.2f}", full_ticker, alert_price)

    # ── Step 2: Re-scale stop loss and targets proportionally ─────────────────
    # Preserve the original R:R ratio when the fill price differs from the alert price
    if alert_price and alert_price > 0 and execution_price != alert_price:
        scale = execution_price / alert_price
        if req.direction == "BUY":
            sl   = execution_price - abs(execution_price - req.stop_loss  * scale)
            t1   = execution_price + abs(req.target1 * scale - execution_price)
            t2   = execution_price + abs(req.target2 * scale - execution_price)
            t3   = execution_price + abs(req.target3 * scale - execution_price)
        else:  # SELL
            sl   = execution_price + abs(req.stop_loss  * scale - execution_price)
            t1   = execution_price - abs(execution_price - req.target1 * scale)
            t2   = execution_price - abs(execution_price - req.target2 * scale)
            t3   = execution_price - abs(execution_price - req.target3 * scale)
    else:
        sl, t1, t2, t3 = req.stop_loss, req.target1, req.target2, req.target3

    # ── Step 3: Build and place order ─────────────────────────────────────────
    order = SimulatedOrder(
        order_id=str(uuid.uuid4()),
        ticker=full_ticker,
        symbol=symbol,
        direction=req.direction,
        quantity=0,  # Will be set by risk manager below
        entry_price=execution_price,
        stop_loss=round(sl, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        target3=round(t3, 2),
        strategy_name=req.strategy_name,
        confidence=req.confidence,
        timeframe=req.timeframe,
        exchange=req.exchange,
        ai_reason=(req.ai_reason or "") + f" | {price_note}",
    )

    # Calculate quantity from risk manager
    risk = risk_manager.check_trade(
        ticker=order.ticker,
        entry_price=order.entry_price,
        stop_loss=order.stop_loss,
        portfolio_balance=virtual_portfolio.balance,
        open_positions=len(virtual_portfolio.open_positions),
    )
    order.quantity = risk.position_size
    result = virtual_portfolio.place_order(order)
    return result


@app.post("/api/portfolio/close")
async def close_position(req: ClosePositionRequest):
    """Close an open position."""
    exit_price = req.exit_price or 0
    if not exit_price:
        pos = virtual_portfolio.open_positions.get(req.order_id)
        exit_price = pos.current_price if pos else 0

    result = virtual_portfolio.close_position(req.order_id, exit_price, "MANUAL")
    return result


@app.get("/api/portfolio/equity-curve")
async def get_equity_curve():
    """Returns historical equity curve data."""
    with db_session() as db:
        from app.db.models import EquityCurvePoint
        points = (
            db.query(EquityCurvePoint)
            .filter_by(portfolio_id=1)
            .order_by(EquityCurvePoint.timestamp)
            .limit(1000)
            .all()
        )
        return {
            "data": [
                {"time": p.timestamp.isoformat(), "value": p.balance, "drawdown": p.drawdown_pct}
                for p in points
            ]
        }


@app.get("/api/portfolio/trades")
async def get_trades(limit: int = Query(50), status: str = Query("ALL")):
    """Returns trade history."""
    with db_session() as db:
        query = db.query(Trade).filter_by(portfolio_id=1)
        if status != "ALL":
            query = query.filter_by(status=status)
        trades = query.order_by(Trade.created_at.desc()).limit(limit).all()

        return {
            "trades": [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "status": t.status,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "net_pnl": t.net_pnl,
                    "pnl_pct": t.pnl_pct,
                    "strategy": t.strategy_used,
                    "exit_reason": t.exit_reason,
                    "holding_minutes": t.holding_minutes,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                }
                for t in trades
            ]
        }


@app.get("/api/portfolio/history")
async def get_portfolio_history(
    status: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    sort_by: str = Query("entry_time"),
    sort_order: str = Query("desc"),
):
    """Returns trade history with advanced filtering and sorting."""
    with db_session() as db:
        query = db.query(Trade).filter_by(portfolio_id=1)
        if status and status != "ALL":
            query = query.filter(Trade.status == status)
        if ticker:
            query = query.filter(Trade.ticker.like(f"%{ticker}%"))
        if timeframe and timeframe != "ALL":
            query = query.filter(Trade.timeframe == timeframe)
        
        if sort_by == "exit_time":
            col = Trade.exit_time
        elif sort_by == "net_pnl":
            col = Trade.net_pnl
        elif sort_by == "pnl_pct":
            col = Trade.pnl_pct
        elif sort_by == "ticker":
            col = Trade.ticker
        else:
            col = Trade.entry_time
            
        if sort_order == "desc":
            query = query.order_by(col.desc())
        else:
            query = query.order_by(col.asc())
            
        trades = query.all()
        
        return {
            "trades": [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "symbol": t.symbol,
                    "exchange": t.exchange or "NSE",
                    "direction": t.direction,
                    "status": t.status,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "stop_loss": t.stop_loss,
                    "target1": t.target1,
                    "target2": t.target2,
                    "target3": t.target3,
                    "strategy_used": t.strategy_used,
                    "confidence_at_entry": t.confidence_at_entry,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "gross_pnl": t.gross_pnl,
                    "brokerage": t.brokerage,
                    "net_pnl": t.net_pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_minutes": t.holding_minutes,
                    "exit_reason": t.exit_reason,
                    "timeframe": t.timeframe or "1d",
                    "coins_used": t.coins_used,
                    "coins_remaining": t.coins_remaining,
                    "ai_reason": t.ai_reason or "",
                    "notes": t.notes or "",
                }
                for t in trades
            ]
        }


@app.get("/api/portfolio/stats-extended")
async def get_portfolio_stats_extended():
    """Returns advanced metrics: ROI, Profit Factor, Sharpe/Sortino ratios, streaks, monthly returns."""
    import numpy as np
    from app.db.models import EquityCurvePoint, Portfolio
    
    with db_session() as db:
        all_trades = db.query(Trade).filter_by(portfolio_id=1).all()
        closed_trades = [t for t in all_trades if t.status == "CLOSED"]
        
        portfolio = db.query(Portfolio).filter_by(id=1).first()
        current_bal = portfolio.current_balance if portfolio else settings.VIRTUAL_INITIAL_BALANCE
        peak_bal = portfolio.peak_balance if portfolio else settings.VIRTUAL_INITIAL_BALANCE
        
        initial_balance = settings.VIRTUAL_INITIAL_BALANCE
        roi = round(((current_bal - initial_balance) / initial_balance) * 100, 2)
        
        points = db.query(EquityCurvePoint).filter_by(portfolio_id=1).all()
        max_drawdown = 0.0
        if points:
            max_drawdown = max([p.drawdown_pct or 0.0 for p in points])
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        gross_profits = 0.0
        gross_losses = 0.0
        
        sorted_trades = sorted(closed_trades, key=lambda x: x.entry_time or datetime.min)
        for t in sorted_trades:
            pnl = t.net_pnl or 0.0
            if pnl > 0:
                gross_profits += pnl
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif pnl < 0:
                gross_losses += abs(pnl)
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
                
        profit_factor = round(gross_profits / gross_losses, 2) if gross_losses > 0 else (round(gross_profits, 2) if gross_profits > 0 else 1.0)
        
        returns = [t.pnl_pct / 100.0 for t in closed_trades if t.pnl_pct is not None]
        sharpe = 0.0
        sortino = 0.0
        if len(returns) > 1:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = round((avg_ret / std_ret) * np.sqrt(252), 2) if std_ret > 0 else 0.0
            
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                sortino = round((avg_ret / downside_std) * np.sqrt(252), 2) if downside_std > 0 else 0.0
            else:
                sortino = round(avg_ret / 1e-6, 2) if avg_ret > 0 else 0.0
                
        monthly_returns = {}
        for t in closed_trades:
            if t.exit_time:
                key = f"{t.exit_time.year}-{t.exit_time.month:02d}"
                monthly_returns[key] = round(monthly_returns.get(key, 0.0) + (t.net_pnl or 0.0), 2)
                
        return {
            "roi": roi,
            "profit_factor": profit_factor,
            "max_drawdown": round(max_drawdown, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "monthly_returns": monthly_returns,
            "total_trades": len(all_trades),
            "closed_trades": len(closed_trades),
            "win_rate": portfolio.win_rate if portfolio else 0.0,
        }


@app.get("/api/portfolio/strategy-analytics")
async def get_portfolio_strategy_analytics():
    """Returns performance breakdown by strategy."""
    with db_session() as db:
        trades = db.query(Trade).filter_by(portfolio_id=1).all()
        
        breakdown = {}
        for t in trades:
            strat = t.strategy_used or "Unknown"
            if strat not in breakdown:
                breakdown[strat] = {
                    "strategy_name": strat,
                    "total_trades": 0,
                    "closed_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "net_pnl": 0.0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "total_holding_minutes": 0,
                }
                
            info = breakdown[strat]
            info["total_trades"] += 1
            if t.status == "CLOSED":
                info["closed_trades"] += 1
                pnl = t.net_pnl or 0.0
                info["net_pnl"] += pnl
                if pnl > 0:
                    info["winning_trades"] += 1
                    info["gross_profit"] += pnl
                else:
                    info["losing_trades"] += 1
                    info["gross_loss"] += abs(pnl)
                info["total_holding_minutes"] += t.holding_minutes or 0
                
        results = []
        for strat, info in breakdown.items():
            win_rate = 0.0
            if info["closed_trades"] > 0:
                win_rate = round((info["winning_trades"] / info["closed_trades"]) * 100, 2)
                
            profit_factor = 1.0
            if info["gross_loss"] > 0:
                profit_factor = round(info["gross_profit"] / info["gross_loss"], 2)
            elif info["gross_profit"] > 0:
                profit_factor = round(info["gross_profit"], 2)
                
            avg_holding = 0.0
            if info["closed_trades"] > 0:
                avg_holding = round(info["total_holding_minutes"] / info["closed_trades"], 1)
                
            results.append({
                "strategy_name": strat,
                "total_trades": info["total_trades"],
                "closed_trades": info["closed_trades"],
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "net_pnl": round(info["net_pnl"], 2),
                "avg_holding_minutes": avg_holding,
            })
            
        return {"strategies": results}


class TradeNoteRequest(BaseModel):
    trade_id: int
    notes: str


@app.post("/api/portfolio/trade/note")
async def update_trade_note(req: TradeNoteRequest):
    """Update notes for a specific trade."""
    with db_session() as db:
        trade = db.query(Trade).filter_by(id=req.trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        trade.notes = req.notes
        db.commit()
        return {"status": "success", "message": "Note updated successfully"}


# ─── Strategies ───────────────────────────────────────────────────────────────

@app.get("/api/strategies")
async def get_strategies():
    return {"strategies": strategy_manager.get_strategy_list()}


@app.post("/api/strategies/{name}/toggle")
async def toggle_strategy(name: str, enabled: bool = Query(True)):
    if enabled:
        strategy_manager.enable(name)
    else:
        strategy_manager.disable(name)
    return {"name": name, "enabled": enabled}


# ─── AI Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_with_ai(msg: ChatMessage):
    """Chat with the local LLM about market topics."""
    context = ""
    if msg.ticker:
        ticker = msg.ticker if "." in msg.ticker else f"{msg.ticker}.NS"
        df = historical_data.get_candles(ticker, "1d", periods=100)
        if not df.empty:
            from app.core.analysis.indicators import compute_all_indicators
            inds = compute_all_indicators(df)
            context = f"Currently analyzing: {msg.ticker}. Price: ₹{inds.get('close', 0):.2f}"

    response = ollama_client.chat(
        message=msg.message,
        history=msg.history,
        model=msg.model,
        context=context,
    )
    return {"response": response, "model": ollama_client.get_best_model()}


@app.get("/api/ai/status")
async def ai_status():
    return ollama_client.get_status()


# ─── Analytics / Self-Learning ────────────────────────────────────────────────

@app.get("/api/analytics/learning")
async def get_learning_report():
    return {"report": self_learning.get_learning_report()}


@app.get("/api/analytics/best-strategies")
async def get_best_strategies():
    return {"best": self_learning.get_best_strategies()}


# ─── News Sentiment ───────────────────────────────────────────────────────────

@app.get("/api/news/trends")
async def get_news_trends(limit: int = Query(8)):
    """Fetch and analyze general market news trends."""
    articles = news_sentiment_analyst.fetch_market_trends_news(limit=limit)
    analysis = news_sentiment_analyst.analyze_sentiment(articles)
    return {
        "articles": articles,
        "sentiment": analysis
    }


@app.get("/api/news/{ticker}")
async def get_ticker_news(ticker: str, limit: int = Query(20)):
    """Fetch and analyze news sentiment for a specific stock ticker."""
    resolved_ticker = nse_client.get_ticker(ticker)
    articles = news_sentiment_analyst.fetch_ticker_news(resolved_ticker, limit=limit)
    analysis = news_sentiment_analyst.analyze_sentiment(articles, ticker=resolved_ticker)
    return {
        "ticker": resolved_ticker,
        "articles": articles,
        "sentiment": analysis
    }


# ─── Data Management ──────────────────────────────────────────────────────────

@app.post("/api/data/download")
async def download_data(
    background_tasks: BackgroundTasks,
    symbols: Optional[List[str]] = None,
    max_stocks: int = Query(50),
):
    """Download historical data in the background."""
    def _download():
        tickers = [f"{s}.NS" for s in (symbols or nse_client.get_nifty50())][:max_stocks]
        historical_data.download_universe(tickers=tickers)

    background_tasks.add_task(_download)
    return {"status": "started", "message": f"Downloading data for up to {max_stocks} stocks"}


@app.get("/api/data/status/{ticker}")
async def get_data_status(ticker: str, timeframe: str = Query("1d")):
    full_ticker = nse_client.get_ticker(ticker)
    df = historical_data.get_candles(full_ticker, timeframe, periods=5)
    return {
        "ticker": full_ticker,
        "timeframe": timeframe,
        "has_data": not df.empty,
        "candles": len(df),
        "latest": df.index[-1].isoformat() if not df.empty else None,
    }


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    return {
        "initial_balance": settings.VIRTUAL_INITIAL_BALANCE,
        "scanner_interval": settings.SCANNER_INTERVAL_SECONDS,
        "min_confidence": settings.MIN_CONFIDENCE_THRESHOLD * 100,
        "min_strategy_agreement": settings.MIN_STRATEGY_AGREEMENT,
        "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
        "max_position_size_pct": settings.MAX_POSITION_SIZE_PCT,
        "risk_per_trade_pct": settings.DEFAULT_RISK_PER_TRADE_PCT,
        "default_timeframes": settings.DEFAULT_TIMEFRAMES,
    }


class SaveSettingsRequest(BaseModel):
    initial_balance: float
    risk_per_trade_pct: float
    min_strategy_agreement: int

@app.post("/api/settings")
async def save_settings(req: SaveSettingsRequest):
    try:
        with db_session() as db:
            for key, val, val_type in [
                ("initial_balance", str(req.initial_balance), "float"),
                ("risk_per_trade_pct", str(req.risk_per_trade_pct), "float"),
                ("min_strategy_agreement", str(req.min_strategy_agreement), "int"),
            ]:
                item = db.query(AppSetting).filter_by(key=key).first()
                if not item:
                    item = AppSetting(key=key, value=val, value_type=val_type)
                    db.add(item)
                else:
                    item.value = val
            db.commit()
        
        # Apply in memory to settings
        settings.VIRTUAL_INITIAL_BALANCE = req.initial_balance
        settings.DEFAULT_RISK_PER_TRADE_PCT = req.risk_per_trade_pct
        settings.MIN_STRATEGY_AGREEMENT = req.min_strategy_agreement
        
        # Sync virtual portfolio initial balance
        virtual_portfolio.initial_balance = req.initial_balance
        
        logger.info("Saved settings: balance={}, risk={}, agreement={}", 
                    req.initial_balance, req.risk_per_trade_pct, req.min_strategy_agreement)
        
        return {"status": "SUCCESS", "message": "Settings saved successfully"}
    except Exception as e:
        logger.error("Save settings error: {}", e)
        raise HTTPException(500, f"Failed to save settings: {e}")


@app.post("/api/risk/reset-day")
async def reset_trading_day():
    try:
        from app.core.risk.risk_manager import risk_manager
        risk_manager.reset_daily_counters()
        return {"status": "SUCCESS", "message": "Trading day refreshed successfully"}
    except Exception as e:
        logger.error("Reset trading day error: {}", e)
        raise HTTPException(500, f"Failed to reset trading day: {e}")


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/scanner")
async def scanner_websocket(websocket: WebSocket):
    """Real-time scanner updates via WebSocket."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send scanner results every 30s
            results = market_scanner.get_results(limit=20)
            await websocket.send_json({
                "type": "scanner_update",
                "results": results[:10],
                "timestamp": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """Real-time portfolio updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            from app.core.simulation.auto_trader import auto_trader
            await websocket.send_json({
                "type": "portfolio_update",
                "stats": virtual_portfolio.get_stats(),
                "positions": virtual_portfolio.get_open_positions(),
                "pending_zone_orders": auto_trader.pending_zone_orders,
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ─── Logs API ─────────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs(
    level: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    limit: int = Query(200),
):
    """Fetch recent system logs, supporting search and level filtering."""
    log_file = LOGS_DIR / "tradebot.log"
    if not log_file.exists():
        return {"logs": []}
        
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        parsed_logs = []
        for line in reversed(lines):
            if query and query.lower() not in line.lower():
                continue
                
            parts = line.split(" | ")
            if len(parts) >= 3:
                timestamp = parts[0].strip()
                log_level = parts[1].strip()
                
                if level and level != "ALL" and log_level.upper() != level.upper():
                    continue
                    
                message = " | ".join(parts[2:]).strip()
                parsed_logs.append({
                    "timestamp": timestamp,
                    "level": log_level,
                    "message": message
                })
            else:
                if level and level != "ALL":
                    continue
                parsed_logs.append({
                    "timestamp": "",
                    "level": "INFO",
                    "message": line.strip()
                })
                
            if len(parsed_logs) >= limit:
                break
                
        return {"logs": parsed_logs}
    except Exception as e:
        logger.error("Failed to read log file: {}", e)
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {str(e)}")


@app.get("/api/logs/download")
async def download_logs():
    """Download the full log file."""
    log_file = LOGS_DIR / "tradebot.log"
    if not log_file.exists():
        log_file.write_text("")
    return FileResponse(
        path=log_file,
        filename="tradebot.log",
        media_type="text/plain"
    )


@app.post("/api/logs/clear")
async def clear_logs():
    """Clear all entries in the log file."""
    log_file = LOGS_DIR / "tradebot.log"
    try:
        if log_file.exists():
            with open(log_file, "w", encoding="utf-8") as f:
                f.truncate(0)
        logger.info("🧹 System logs cleared via API")
        return {"status": "success", "message": "Logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear log file: {str(e)}")


# ─── Research & Automation APIs ───────────────────────────────────────────────

@app.get("/api/research/candidates")
async def get_research_candidates():
    """Fetch all candidate strategies discovered by the research engine."""
    try:
        with db_session() as db:
            from app.db.models import CandidateStrategy
            candidates = db.query(CandidateStrategy).order_by(CandidateStrategy.created_at.desc()).all()
            return {
                "status": "success",
                "candidates": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "definition": c.combination_definition,
                        "status": c.status,
                        "sharpe": round(c.backtest_sharpe, 2),
                        "profit_factor": round(c.backtest_profit_factor, 2),
                        "win_rate": round(c.backtest_win_rate * 100, 1),
                        "drawdown": round(c.backtest_drawdown, 1),
                        "created_at": c.created_at.isoformat()
                    }
                    for c in candidates
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/patterns")
async def get_pattern_performance():
    """Fetch candlestick/chart pattern performance metrics."""
    try:
        with db_session() as db:
            from app.db.models import PatternPerformance
            patterns = db.query(PatternPerformance).order_by(PatternPerformance.win_rate.desc()).all()
            return {
                "status": "success",
                "patterns": [
                    {
                        "id": p.id,
                        "name": p.pattern_name,
                        "type": p.pattern_type,
                        "occurrences": p.occurrences,
                        "wins": p.wins,
                        "losses": p.losses,
                        "win_rate": round(p.win_rate * 100, 1),
                        "avg_return": round(p.avg_return, 2),
                        "avg_holding_minutes": round(p.avg_holding_minutes, 1),
                        "success_probability": round(p.success_probability * 100, 1),
                        "market_regime": p.market_regime
                    }
                    for p in patterns
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/system-status")
async def get_system_status():
    """Fetch real-time CPU, RAM, active threads, and engine statuses."""
    import psutil
    import os
    import threading
    from app.core.simulation.auto_trader import auto_trader
    from app.core.research.research_lab import research_engine

    try:
        # DB Size
        db_size_mb = 0.0
        if os.path.exists(settings.DB_PATH):
            db_size_mb = round(os.path.getsize(settings.DB_PATH) / (1024 * 1024), 2)
            
        # CPU & Memory
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        ram_usage = memory.percent
        
        # Thread count
        thread_count = threading.active_count()
        
        return {
            "status": "success",
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "thread_count": thread_count,
            "db_size_mb": db_size_mb,
            "autotrader_status": auto_trader.status,
            "research_status": research_engine.status,
            "research_generation": research_engine.generation,
            "active_combinations": research_engine.active_combinations,
            "last_scan_time": market_scanner.last_scan_time.isoformat() if market_scanner.last_scan_time else None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run():
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    run()

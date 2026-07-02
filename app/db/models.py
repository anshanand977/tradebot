"""
Database Layer - SQLAlchemy Models
===================================
All ORM models for the SQLite database.
12 tables covering the full application data lifecycle.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime,
    Text, JSON, ForeignKey, Index, UniqueConstraint,
    create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYMBOLS — Stock universe
# ─────────────────────────────────────────────────────────────────────────────
class Symbol(Base):
    __tablename__ = "symbols"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(20), nullable=False)          # e.g. RELIANCE
    ticker      = Column(String(25), nullable=False, unique=True)  # e.g. RELIANCE.NS
    name        = Column(String(100))
    exchange    = Column(String(10), default="NSE")            # NSE / BSE
    sector      = Column(String(50))
    industry    = Column(String(80))
    market_cap  = Column(Float)
    is_active   = Column(Boolean, default=True)
    in_nifty50  = Column(Boolean, default=False)
    in_banknifty = Column(Boolean, default=False)
    in_fno      = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candles     = relationship("Candle", back_populates="symbol_ref", lazy="dynamic")

    __table_args__ = (
        Index("ix_symbols_symbol", "symbol"),
        Index("ix_symbols_exchange", "exchange"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CANDLES — OHLCV data
# ─────────────────────────────────────────────────────────────────────────────
class Candle(Base):
    __tablename__ = "candles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(String(25), ForeignKey("symbols.ticker"), nullable=False)
    timeframe  = Column(String(5), nullable=False)    # 1m, 5m, 15m, 1h, 1d, 1wk
    timestamp  = Column(DateTime, nullable=False)
    open       = Column(Float, nullable=False)
    high       = Column(Float, nullable=False)
    low        = Column(Float, nullable=False)
    close      = Column(Float, nullable=False)
    volume     = Column(Float, nullable=False)
    vwap       = Column(Float)
    delivery_pct = Column(Float)

    symbol_ref = relationship("Symbol", back_populates="candles")

    __table_args__ = (
        UniqueConstraint("ticker", "timeframe", "timestamp", name="uq_candle"),
        Index("ix_candles_ticker_tf_ts", "ticker", "timeframe", "timestamp"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNALS — Generated trade signals
# ─────────────────────────────────────────────────────────────────────────────
class Signal(Base):
    __tablename__ = "signals"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    ticker              = Column(String(25), nullable=False)
    symbol              = Column(String(20))
    exchange            = Column(String(10))
    direction           = Column(String(5))         # BUY / SELL / HOLD / NONE
    timeframe           = Column(String(5))
    confidence          = Column(Float)             # 0.0 – 1.0
    probability         = Column(Float)             # Real calculated probability
    risk_score          = Column(Float)             # 0.0 – 1.0 (higher = riskier)
    entry_price         = Column(Float)
    stop_loss           = Column(Float)
    target1             = Column(Float)
    target2             = Column(Float)
    target3             = Column(Float)
    risk_reward         = Column(Float)
    strategy_votes      = Column(JSON)              # {strategy: {dir, conf}}
    strategies_agreed   = Column(Integer)
    total_strategies    = Column(Integer)
    contributing_signals = Column(JSON)             # List of signal reasons
    market_regime       = Column(String(20))        # TRENDING / RANGING / VOLATILE
    indicator_snapshot  = Column(JSON)              # All indicator values at signal time
    pattern_detected    = Column(JSON)              # List of patterns
    smc_context         = Column(JSON)              # SMC analysis context
    was_traded          = Column(Boolean, default=False)
    trade_id            = Column(Integer, ForeignKey("trades.id"), nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_signals_ticker_ts", "ticker", "created_at"),
        Index("ix_signals_direction", "direction"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRADES — Simulated trade records
# ─────────────────────────────────────────────────────────────────────────────
class Trade(Base):
    __tablename__ = "trades"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id      = Column(Integer, ForeignKey("portfolios.id"))
    signal_id         = Column(Integer, nullable=True)
    ticker            = Column(String(25), nullable=False)
    symbol            = Column(String(20))
    direction         = Column(String(5))              # BUY / SELL
    status            = Column(String(10), default="OPEN")  # OPEN/CLOSED/CANCELLED
    entry_price       = Column(Float)
    exit_price        = Column(Float)
    quantity          = Column(Float)
    stop_loss         = Column(Float)
    target1           = Column(Float)
    target2           = Column(Float)
    target3           = Column(Float)
    exit_reason       = Column(String(30))             # TARGET1/SL/MANUAL/EOD
    gross_pnl         = Column(Float)
    brokerage         = Column(Float)
    net_pnl           = Column(Float)
    pnl_pct           = Column(Float)
    holding_minutes   = Column(Integer)
    strategy_used     = Column(String(50))
    strategies_voted  = Column(JSON)
    confidence_at_entry = Column(Float)
    probability_at_entry = Column(Float)
    market_regime     = Column(String(20))
    indicator_snapshot = Column(JSON)
    entry_time        = Column(DateTime)
    exit_time         = Column(DateTime)
    exchange          = Column(String(10), default="NSE")
    timeframe         = Column(String(5), default="1d")
    coins_used        = Column(Float)
    coins_remaining   = Column(Float)
    ai_reason         = Column(Text)
    notes             = Column(Text)
    created_at        = Column(DateTime, default=datetime.utcnow)

    portfolio        = relationship("Portfolio", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_ticker_status", "ticker", "status"),
        Index("ix_trades_entry_time", "entry_time"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. PORTFOLIOS — Virtual portfolios
# ─────────────────────────────────────────────────────────────────────────────
class Portfolio(Base):
    __tablename__ = "portfolios"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    name             = Column(String(50), default="Virtual Portfolio")
    is_live          = Column(Boolean, default=False)   # NEVER True by default
    initial_balance  = Column(Float, default=100_000.0)
    current_balance  = Column(Float, default=100_000.0)
    total_pnl        = Column(Float, default=0.0)
    total_trades     = Column(Integer, default=0)
    winning_trades   = Column(Integer, default=0)
    losing_trades    = Column(Integer, default=0)
    win_rate         = Column(Float, default=0.0)
    max_drawdown     = Column(Float, default=0.0)
    peak_balance     = Column(Float, default=100_000.0)
    is_active        = Column(Boolean, default=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trades           = relationship("Trade", back_populates="portfolio")
    equity_curve     = relationship("EquityCurvePoint", back_populates="portfolio")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EQUITY CURVE — Portfolio value over time
# ─────────────────────────────────────────────────────────────────────────────
class EquityCurvePoint(Base):
    __tablename__ = "equity_curve"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    timestamp    = Column(DateTime, default=datetime.utcnow)
    balance      = Column(Float)
    drawdown_pct = Column(Float)
    daily_pnl    = Column(Float)

    portfolio    = relationship("Portfolio", back_populates="equity_curve")

    __table_args__ = (Index("ix_equity_portfolio_ts", "portfolio_id", "timestamp"),)


# ─────────────────────────────────────────────────────────────────────────────
# 7. STRATEGY PERFORMANCE — Rolling performance per strategy
# ─────────────────────────────────────────────────────────────────────────────
class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name  = Column(String(50), nullable=False)
    market_regime  = Column(String(20))           # TRENDING / RANGING / VOLATILE
    timeframe      = Column(String(5))
    total_signals  = Column(Integer, default=0)
    total_trades   = Column(Integer, default=0)
    wins           = Column(Integer, default=0)
    losses         = Column(Integer, default=0)
    win_rate       = Column(Float, default=0.0)
    avg_profit_pct = Column(Float, default=0.0)
    avg_loss_pct   = Column(Float, default=0.0)
    profit_factor  = Column(Float, default=0.0)
    current_weight = Column(Float, default=1.0)   # Voting weight (adjusted by self-learning)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("strategy_name", "market_regime", "timeframe",
                         name="uq_strategy_perf"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_name        = Column(String(100))
    strategy_name   = Column(String(50))
    ticker          = Column(String(25))
    timeframe       = Column(String(5))
    start_date      = Column(DateTime)
    end_date        = Column(DateTime)
    initial_capital = Column(Float)
    final_capital   = Column(Float)
    total_return_pct = Column(Float)
    cagr_pct        = Column(Float)
    total_trades    = Column(Integer)
    win_rate        = Column(Float)
    profit_factor   = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio    = Column(Float)
    sortino_ratio   = Column(Float)
    avg_holding_days = Column(Float)
    best_trade_pct  = Column(Float)
    worst_trade_pct = Column(Float)
    monthly_returns = Column(JSON)   # {YYYY-MM: pct}
    equity_curve    = Column(JSON)   # [{ts, value}]
    parameters      = Column(JSON)   # Strategy parameters used
    created_at      = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# 9. WATCHLISTS
# ─────────────────────────────────────────────────────────────────────────────
class Watchlist(Base):
    __tablename__ = "watchlists"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(50), nullable=False, unique=True)
    tickers    = Column(JSON, default=list)          # List of tickers
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# 10. LEARNING LOG — Self-learning events
# ─────────────────────────────────────────────────────────────────────────────
class LearningLog(Base):
    __tablename__ = "learning_log"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    trade_id         = Column(Integer, ForeignKey("trades.id"))
    strategy_name    = Column(String(50))
    old_weight       = Column(Float)
    new_weight       = Column(Float)
    reason           = Column(String(200))
    market_regime    = Column(String(20))
    trade_outcome    = Column(String(10))      # WIN / LOSS
    pnl_pct          = Column(Float)
    created_at       = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# 11. SETTINGS — Key-value app settings
# ─────────────────────────────────────────────────────────────────────────────
class AppSetting(Base):
    __tablename__ = "app_settings"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    key        = Column(String(100), unique=True, nullable=False)
    value      = Column(Text)
    value_type = Column(String(20), default="str")  # str, int, float, bool, json
    category   = Column(String(50), default="general")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# 12. LOGS — Application event log
# ─────────────────────────────────────────────────────────────────────────────
class AppLog(Base):
    __tablename__ = "app_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    level      = Column(String(10))    # INFO / WARNING / ERROR / DEBUG
    module     = Column(String(50))
    message    = Column(Text)
    extra      = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_logs_level_ts", "level", "created_at"),)


# ─────────────────────────────────────────────────────────────────────────────
# 13. ORDERS — Virtual order book (referenced by virtual_portfolio.py)
# ─────────────────────────────────────────────────────────────────────────────
class Order(Base):
    __tablename__ = "orders"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    order_id     = Column(String(36), unique=True)     # UUID
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker       = Column(String(25))
    symbol       = Column(String(20))
    direction    = Column(String(5))   # BUY / SELL
    status       = Column(String(10))  # PENDING / FILLED / REJECTED / CANCELLED
    order_type   = Column(String(10), default="MARKET")
    quantity     = Column(Float)
    price        = Column(Float)       # Requested price
    fill_price   = Column(Float)       # Actual fill price
    stop_loss    = Column(Float)
    target1      = Column(Float)
    target2      = Column(Float)
    target3      = Column(Float)
    strategy     = Column(String(50))
    confidence   = Column(Float)
    reject_reason = Column(String(200))
    created_at   = Column(DateTime, default=datetime.utcnow)
    filled_at    = Column(DateTime)

    __table_args__ = (Index("ix_orders_portfolio", "portfolio_id", "status"),)


# ─────────────────────────────────────────────────────────────────────────────
# 14. CANDIDATE STRATEGIES — Discovered by AI Research Lab
# ─────────────────────────────────────────────────────────────────────────────
class CandidateStrategy(Base):
    __tablename__ = "candidate_strategies"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    name                  = Column(String(100), unique=True, nullable=False)
    combination_definition = Column(JSON)             # Detailed indicator and rules mapping
    status                = Column(String(20), default="TESTING")  # TESTING / PROMOTED / REJECTED
    backtest_sharpe       = Column(Float, default=0.0)
    backtest_profit_factor = Column(Float, default=0.0)
    backtest_win_rate     = Column(Float, default=0.0)
    backtest_drawdown     = Column(Float, default=0.0)
    paper_trades_count    = Column(Integer, default=0)
    paper_win_rate        = Column(Float, default=0.0)
    paper_profit_factor   = Column(Float, default=0.0)
    created_at            = Column(DateTime, default=datetime.utcnow)
    updated_at            = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# 15. PATTERN PERFORMANCE — Catalog of detected chart and SMC patterns
# ─────────────────────────────────────────────────────────────────────────────
class PatternPerformance(Base):
    __tablename__ = "pattern_performance"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    pattern_name       = Column(String(50), nullable=False)
    pattern_type       = Column(String(20))           # CANDLESTICK / SMC / CHART / BREAKOUT
    occurrences        = Column(Integer, default=0)
    wins               = Column(Integer, default=0)
    losses             = Column(Integer, default=0)
    win_rate           = Column(Float, default=0.0)
    avg_return         = Column(Float, default=0.0)
    avg_holding_minutes = Column(Float, default=0.0)
    success_probability = Column(Float, default=0.0)  # Calibrated probability
    market_regime      = Column(String(20), default="ALL")

    __table_args__ = (
        UniqueConstraint("pattern_name", "market_regime", "pattern_type", name="uq_pattern_perf"),
    )

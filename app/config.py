"""
Global Configuration for AI Trading Analyst
============================================
All tunable constants live here. Never hardcode values in business logic.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running inside PyInstaller single-file EXE
    BUNDLE_DIR = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).parent
    BASE_DIR = EXE_DIR
    FRONTEND_DIR = BUNDLE_DIR / "frontend"
else:
    # Running in normal Python dev mode
    BASE_DIR = Path(__file__).resolve().parent.parent
    FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for _dir in [DATA_DIR, DB_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Application Settings ─────────────────────────────────────────────────────
class Settings(BaseSettings):
    """
    Application settings.  Values can be overridden via environment variables
    or a .env file in the project root.
    """

    # App
    APP_NAME: str = "AI Trading Analyst"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8765

    # Database
    DB_PATH: str = str(DB_DIR / "tradebot.db")
    DB_URL: str = f"sqlite:///{DB_DIR / 'tradebot.db'}"

    # Ollama (local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "mistral"
    OLLAMA_TIMEOUT: int = 120  # seconds

    # Virtual Portfolio
    VIRTUAL_INITIAL_BALANCE: float = 100_000.0  # Coins (₹ equivalent)
    VIRTUAL_BROKERAGE_PER_TRADE: float = 20.0   # Flat ₹20 like Zerodha
    VIRTUAL_SLIPPAGE_PCT: float = 0.05           # 0.05% slippage

    # Risk Management
    MAX_DAILY_LOSS_PCT: float = 2.0       # % of capital
    MAX_CONCURRENT_POSITIONS: int = 5
    DEFAULT_RISK_PER_TRADE_PCT: float = 1.0  # % of capital per trade
    MAX_POSITION_SIZE_PCT: float = 20.0      # Max 20% in one stock
    MAX_SECTOR_EXPOSURE_PCT: float = 30.0
    MAX_DRAWDOWN_PCT: float = 10.0

    # Scanner
    SCANNER_INTERVAL_SECONDS: int = 60
    SCANNER_UNIVERSE_SIZE: int = 500       # NSE 500
    MIN_STRATEGY_AGREEMENT: int = 4        # Min strategies must agree for signal
    MIN_CONFIDENCE_THRESHOLD: float = 0.72 # Minimum 72% confidence
    MIN_RISK_REWARD: float = 2.0           # Industry standard: minimum 1:2 R:R before trade is actionable
    ENTRY_ZONE_BUFFER_PCT: float = 0.8     # ±0.8% zone around optimal entry price for pending zone orders
    ZONE_ORDER_EXPIRY_HOURS: float = 6.0   # Pending zone orders expire after this many hours

    # Data
    DATA_PROVIDER: str = "yfinance"        # Primary data source
    HISTORICAL_YEARS: int = 5
    SUPPORTED_TIMEFRAMES: list = ["1m", "3m", "5m", "15m", "30m", "1h", "1d", "1wk"]
    DEFAULT_TIMEFRAMES: list = ["5m", "15m", "1h", "1d"]  # Default analysis set

    # Exchanges
    SUPPORTED_EXCHANGES: list = ["NSE", "BSE"]
    DEFAULT_EXCHANGE: str = "NSE"
    NSE_SUFFIX: str = ".NS"
    BSE_SUFFIX: str = ".BO"

    # Market Hours (IST)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MIN: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MIN: int = 30
    PRE_MARKET_HOUR: int = 9
    PRE_MARKET_MIN: int = 0

    # Strategies
    STRATEGY_WEIGHT_DECAY: float = 0.95   # Weight decay for old performance
    STRATEGY_MIN_TRADES_FOR_LEARNING: int = 20

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "10 MB"

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()


# ─── Market Constants ─────────────────────────────────────────────────────────
NIFTY50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFY", "SBIN", "HINDUNILVR", "ITC", "LT",
    "KOTAKBANK", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "POWERGRID", "NTPC", "TATAMOTORS", "ONGC",
    "COALINDIA", "ADANIENT", "ADANIPORTS", "JSWSTEEL", "TATASTEEL",
    "TECHM", "INDUSINDBK", "BAJAJ-AUTO", "BAJAJFINSV", "GRASIM",
    "DIVISLAB", "CIPLA", "DRREDDY", "EICHERMOT", "BPCL",
    "APOLLOHOSP", "BRITANNIA", "HEROMOTOCO", "HINDALCO", "TATACONSUM",
    "M&M", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN", "LTIM",
]

BANKNIFTY_STOCKS = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "BANKBARODA", "AUBANK",
]

MAJOR_INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTYMIDCAP": "^NSEMDCP50",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
}

# ─── Timeframe Display Names ───────────────────────────────────────────────────
TIMEFRAME_LABELS = {
    "1m":  "1 Minute",
    "3m":  "3 Minute",
    "5m":  "5 Minute",
    "15m": "15 Minute",
    "30m": "30 Minute",
    "1h":  "1 Hour",
    "1d":  "Daily",
    "1wk": "Weekly",
}

# ─── Signal Colors (for UI) ───────────────────────────────────────────────────
SIGNAL_COLORS = {
    "BUY":   "#00d4aa",
    "SELL":  "#ff4976",
    "HOLD":  "#f5a623",
    "NONE":  "#6c7a8d",
}

# ─── Risk Levels ──────────────────────────────────────────────────────────────
RISK_LEVELS = {
    "LOW":    (0.0, 2.0),   # Below industry standard
    "MEDIUM": (2.0, 3.0),   # Industry standard (1:2 to 1:3)
    "HIGH":   (3.0, 99.0),  # Above standard (reward dominant)
}

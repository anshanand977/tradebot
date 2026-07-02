"""
NSE / BSE Market Data Client
==============================
Manages the Indian stock universe (NSE 500 + BANKNIFTY + major indices).
Fetches symbol lists from public sources and caches them locally.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
import pandas as pd
from loguru import logger

from app.config import settings, NIFTY50_STOCKS, BANKNIFTY_STOCKS, MAJOR_INDICES, DATA_DIR


CACHE_FILE = DATA_DIR / "cache" / "nse_symbols.json"
CACHE_TTL_HOURS = 24


# ─── Comprehensive NSE 500 Symbol List ────────────────────────────────────────
# Curated list of major NSE stocks across all sectors
NSE_500_SYMBOLS = [
    # NIFTY 50
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
    # NIFTY NEXT 50
    "HAVELLS", "PIDILITIND", "MUTHOOTFIN", "LUPIN", "SIEMENS",
    "GODREJCP", "AMBUJACEM", "ICICIPRULI", "TORNTPHARM", "DLF",
    "CHOLAFIN", "BERGEPAINT", "SRF", "ADANIGREEN", "ADANIPOWER",
    "ZOMATO", "PAYTM", "NYKAA", "POLICYBZR", "DELHIVERY",
    # Large Cap
    "RECLTD", "PFC", "HUDCO", "IRFC", "IREDA",
    "HAL", "BEL", "BHEL", "SAIL", "NMDC",
    "VEDL", "JINDALSTEL", "HINDZINC", "NALCO", "MOIL",
    "AARTIIND", "ALKEM", "AUROPHARMA", "BIOCON", "CADILAHC",
    "GLENMARK", "IPCALAB", "LALPATHLAB", "METROPOLIS", "NATCOPHARM",
    # Banking & Finance
    "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "FEDERALBNK", "BANDHANBNK", "AUBANK", "RBLBANK", "IDBI",
    "BAJAJHFL", "MFSL", "ICICIGI", "NIACL", "GICRE",
    # IT
    "LTTS", "MINDTREE", "MPHASIS", "PERSISTENT", "COFORGE",
    "KPITTECH", "TATAELXSI", "BSOFT", "MASTEK", "NIITLTD",
    # Auto
    "ASHOKLEY", "TVSMOTOR", "ESCORTS", "FORCE", "BALKRISIND",
    "EXIDEIND", "AMARAJABAT", "MOTHERSON", "BOSCHLTD", "ENDURANCE",
    # Pharma
    "ASTRAZEN", "PFIZER", "SANOFI", "ABBOTINDIA", "GLAXO",
    # FMCG
    "DABUR", "MARICO", "EMAMILTD", "COLPAL", "PGHH",
    "GILLETTE", "WHIRLPOOL", "VOLTAS", "BLUESTARCO", "CROMPTON",
    # Real Estate
    "GODREJPROP", "PRESTIGE", "OBEROIRLTY", "PHOENIXLTD", "SOBHA",
    # Energy
    "IOC", "HPCL", "MRPL", "GAIL", "PETRONET",
    "TATAPOWER", "ADANIGREEN", "GREENPANEL", "TORNTPOWER", "CESC",
    # Metals & Mining
    "HINDALCO", "NATIONALUM", "APLAPOLLO", "RATNAMANI", "WELCORP",
    # Chemicals
    "PIDILITIND", "AARTIIND", "VINDHYATEL", "DEEPAKNITRITE", "NAVINFLUOR",
    # Infrastructure
    "LTTS", "KNRCON", "SADBHAV", "IRB", "PNCINFRA",
    # Consumer
    "INDIGO", "SPICEJET", "INTERGLOBE", "MAHINDCIE", "MHRIL",
    # Media
    "SUNTV", "ZEEL", "PVR", "INOXLEISUR", "TIPS",
    # Textiles
    "PAGEIND", "RAYMOND", "ARVIND", "VARDHMAN", "WELSPUNIND",
    # Small & Mid Cap
    "DIXON", "AMBER", "BLKASHYAP", "KANSAINER", "SUPRAJIT",
    "BAJAJELEC", "FINEORG", "GALAXYSURF", "GHCL", "HIKAL",
    "IDFCFIRSTB", "IIFL", "INDIANB", "INDOCO", "JKPAPER",
    "KAJARIACER", "KPRMILL", "MAHLOG", "MASFIN", "NAGAFERT",
    "NESCO", "ORIENTELEC", "POLYCAB", "PRINCEPIPE", "RAJESHEXPO",
    "RPGLIFE", "SAREGAMA", "SHOPERSTOP", "SKIPPER", "SPARC",
    "SUNDARMFIN", "SUPRAJIT", "SWSOLAR", "TANLA", "TRENT",
    "UCOBANK", "UJJIVANSFB", "VSTIND", "ZENSARTECH",
]

# Remove duplicates while preserving order
seen = set()
NSE_500_SYMBOLS = [x for x in NSE_500_SYMBOLS if not (x in seen or seen.add(x))]

SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "HPCL": "Energy", "GAIL": "Energy",
    "PETRONET": "Energy", "TATAPOWER": "Energy", "ADANIGREEN": "Energy",

    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT",
    "COFORGE": "IT", "TATAELXSI": "IT",

    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "IDFCFIRSTB": "Banking",
    "FEDERALBNK": "Banking", "BANDHANBNK": "Banking",

    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "MUTHOOTFIN": "Finance",
    "CHOLAFIN": "Finance", "SBILIFE": "Finance", "HDFCLIFE": "Finance",

    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma", "LUPIN": "Pharma", "AUROPHARMA": "Pharma",
    "BIOCON": "Pharma",

    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",

    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "COALINDIA": "Metals",

    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "EICHERMOT": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto", "ASHOKLEY": "Auto",

    "ASIANPAINT": "Consumer", "TITAN": "Consumer", "HAVELLS": "Consumer",
    "VOLTAS": "Consumer", "CROMPTON": "Consumer",

    "LT": "Infrastructure", "ADANIPORTS": "Infrastructure",
    "DLF": "Realty", "GODREJPROP": "Realty",
    "BHARTIARTL": "Telecom",
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "GRASIM": "Cement",
    "POWERGRID": "Power", "NTPC": "Power",
    "SIEMENS": "Capital Goods", "BEL": "Defence", "HAL": "Defence",
}


class NSEClient:
    """Manages the Indian stock universe and sector information."""

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_loaded_at: Optional[datetime] = None

    def get_symbol_universe(self) -> List[Dict]:
        """
        Returns the full symbol universe with metadata.
        Loads from cache file if fresh, otherwise rebuilds from embedded list.
        """
        if self._is_cache_valid():
            return self._cache["symbols"]

        symbols = self._build_symbol_list()
        self._save_cache(symbols)
        return symbols

    def get_nifty50(self) -> List[str]:
        return NIFTY50_STOCKS

    def get_banknifty(self) -> List[str]:
        return BANKNIFTY_STOCKS

    def get_all_tickers(self, exchange: str = "NSE") -> List[str]:
        """Returns yfinance-compatible tickers (e.g. RELIANCE.NS)."""
        suffix = settings.NSE_SUFFIX if exchange == "NSE" else settings.BSE_SUFFIX
        return [f"{s}{suffix}" for s in NSE_500_SYMBOLS]

    def get_ticker(self, symbol: str, exchange: str = "NSE") -> str:
        symbol = symbol.replace("&", "%26").strip()
        # If the symbol already has a suffix/dot, return as is
        if "." in symbol:
            return symbol
        
        # If it's a known Indian stock, append local suffix
        if symbol.upper() in NSE_500_SYMBOLS:
            suffix = settings.NSE_SUFFIX if exchange == "NSE" else settings.BSE_SUFFIX
            return f"{symbol.upper()}{suffix}"
            
        # Otherwise, treat as global symbol (e.g. US stock like AAPL, MSFT, TSLA)
        return symbol.upper()

    def get_sector(self, symbol: str) -> str:
        return SECTOR_MAP.get(symbol, "Other")

    def get_index_tickers(self) -> Dict[str, str]:
        return MAJOR_INDICES

    def _build_symbol_list(self) -> List[Dict]:
        result = []
        for symbol in NSE_500_SYMBOLS:
            result.append({
                "symbol": symbol,
                "ticker": f"{symbol}.NS",
                "exchange": "NSE",
                "sector": self.get_sector(symbol),
                "in_nifty50": symbol in NIFTY50_STOCKS,
                "in_banknifty": symbol in BANKNIFTY_STOCKS,
            })
        logger.info("Built symbol universe: {} stocks", len(result))
        return result

    def _is_cache_valid(self) -> bool:
        if not CACHE_FILE.exists():
            return False
        try:
            with open(CACHE_FILE) as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.utcnow() - cached_at < timedelta(hours=CACHE_TTL_HOURS):
                self._cache = data
                return True
        except Exception:
            pass
        return False

    def _save_cache(self, symbols: List[Dict]) -> None:
        data = {
            "cached_at": datetime.utcnow().isoformat(),
            "symbols": symbols,
        }
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        self._cache = data
        self._cache_loaded_at = datetime.utcnow()
        logger.debug("Symbol cache saved: {} symbols", len(symbols))


nse_client = NSEClient()

"""
Historical Market Data Manager
================================
Downloads, stores, and serves OHLCV data from yfinance.
All data is stored locally in SQLite for offline use.
Supports incremental updates — never re-downloads what's already stored.
"""

from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Tuple
import time

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings, DATA_DIR
from app.db.database import db_session
from app.db.models import Candle, Symbol
from app.data.nse_client import nse_client


# ─── yfinance timeframe mapping ───────────────────────────────────────────────
YF_INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "1d":  "1d",
    "1wk": "1wk",
}

# yfinance limits: intraday data only available for last N days
YF_INTRADAY_LIMITS = {
    "1m":  5,
    "3m":  30,
    "5m":  30,
    "15m": 30,
    "30m": 30,
    "1h":  365,
    "1d":  365 * 15,
    "1wk": 365 * 20,
}


class HistoricalDataManager:
    """
    Manages OHLCV data download and storage.

    Usage:
        manager = HistoricalDataManager()
        df = manager.get_candles("RELIANCE.NS", "15m", periods=200)
    """

    def __init__(self):
        self._download_count = 0
        self._last_download = datetime.utcnow()
        self._candle_cache = {}

    # ─── Public API ───────────────────────────────────────────────────────────
    def get_candles(
        self,
        ticker: str,
        timeframe: str,
        periods: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with OHLCV data.
        Tries local DB first, downloads only what's missing.

        Returns columns: [open, high, low, close, volume, timestamp]
        Indexed by timestamp (UTC).
        """
        if timeframe not in YF_INTERVAL_MAP:
            logger.warning("Unsupported timeframe: {}", timeframe)
            return pd.DataFrame()

        cache_key = (ticker, timeframe, periods)
        import time
        now = time.time()

        if not force_refresh and cache_key in self._candle_cache:
            cache_time, cached_df = self._candle_cache[cache_key]
            if now - cache_time < 5:
                return cached_df.copy()

        # Try loading from local DB
        df = self._load_from_db(ticker, timeframe, periods)

        # Check if we need fresh data
        needs_refresh = force_refresh or self._needs_update(df, ticker, timeframe)

        if needs_refresh:
            fresh_df = self._download_and_store(ticker, timeframe)
            if not fresh_df.empty:
                # Reload after storing
                df = self._load_from_db(ticker, timeframe, periods)

        result_df = df if not df.empty else pd.DataFrame()

        if not result_df.empty:
            self._candle_cache[cache_key] = (now, result_df)

        return result_df.copy() if not result_df.empty else result_df

    def download_universe(
        self,
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_tickers: int = 50,
    ) -> Dict[str, int]:
        """
        Downloads historical data for a list of tickers.
        Used during initial setup and scheduled refresh.
        Returns {ticker: candles_stored}.
        """
        if tickers is None:
            tickers = nse_client.get_all_tickers()[:max_tickers]
        if timeframes is None:
            timeframes = settings.DEFAULT_TIMEFRAMES

        results = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            logger.info("Downloading {}/{}: {}", i, total, ticker)
            count = 0
            for tf in timeframes:
                df = self._download_and_store(ticker, tf)
                count += len(df)
                self._rate_limit()
            results[ticker] = count

        logger.success("Universe download complete. {} tickers processed.", len(results))
        return results

    def sync_symbols_to_db(self) -> int:
        """Syncs the NSE symbol universe into the symbols table."""
        symbols_data = nse_client.get_symbol_universe()
        count = 0
        with db_session() as db:
            # Sync stock symbols
            for s in symbols_data:
                existing = db.query(Symbol).filter_by(ticker=s["ticker"]).first()
                if not existing:
                    sym = Symbol(
                        symbol=s["symbol"],
                        ticker=s["ticker"],
                        exchange=s["exchange"],
                        sector=s["sector"],
                        in_nifty50=s["in_nifty50"],
                        in_banknifty=s["in_banknifty"],
                    )
                    db.add(sym)
                    count += 1

            # Sync index tickers
            from app.config import MAJOR_INDICES
            for name, ticker in MAJOR_INDICES.items():
                existing = db.query(Symbol).filter_by(ticker=ticker).first()
                if not existing:
                    sym = Symbol(
                        symbol=name,
                        ticker=ticker,
                        exchange="NSE",
                        sector="Index",
                        in_nifty50=False,
                        in_banknifty=False,
                    )
                    db.add(sym)
                    count += 1

            db.commit()
        logger.info("Synced {} new symbols/indices to DB", count)
        return count

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Returns the latest close price from the local DB."""
        with db_session() as db:
            row = (
                db.query(Candle.close)
                .filter_by(ticker=ticker, timeframe="1d")
                .order_by(Candle.timestamp.desc())
                .first()
            )
            return row[0] if row else None

    # ─── Private Methods ──────────────────────────────────────────────────────
    def _load_from_db(self, ticker: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Load candle data from SQLite (optimized projection query)."""
        with db_session() as db:
            rows = (
                db.query(
                    Candle.timestamp,
                    Candle.open,
                    Candle.high,
                    Candle.low,
                    Candle.close,
                    Candle.volume
                )
                .filter_by(ticker=ticker, timeframe=timeframe)
                .order_by(Candle.timestamp.desc())
                .limit(periods)
                .all()
            )

            if not rows:
                return pd.DataFrame()

            data = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in reversed(rows)
            ]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _download_and_store(self, ticker: str, timeframe: str) -> pd.DataFrame:
        """Download from yfinance and persist to DB."""
        limit_days = YF_INTRADAY_LIMITS.get(timeframe, 365)
        if timeframe == "3m":
            limit_days = 5
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=min(limit_days, 365 * settings.HISTORICAL_YEARS))

        # Find the latest stored timestamp to only download new data
        latest_ts = self._get_latest_stored_ts(ticker, timeframe)
        if latest_ts:
            start_date = latest_ts + timedelta(minutes=1)
            if (end_date - start_date).total_seconds() < 60:
                logger.debug("{}/{} is up to date", ticker, timeframe)
                return pd.DataFrame()

        try:
            logger.debug("Downloading {}/{} from {} to {}", ticker, timeframe,
                         start_date.date(), end_date.date())
            yf_ticker = yf.Ticker(ticker)
            
            if timeframe == "3m":
                # Resample from 1m candles
                df = yf_ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1m",
                    auto_adjust=True,
                    prepost=False,
                )
                if not df.empty:
                    resample_rules = {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum"
                    }
                    df.columns = [c.capitalize() for c in df.columns]
                    df = df.resample("3min").agg(resample_rules).dropna()
            else:
                df = yf_ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval=YF_INTERVAL_MAP[timeframe],
                    auto_adjust=True,
                    prepost=False,
                )

            if df.empty:
                logger.debug("No data returned for {}/{}", ticker, timeframe)
                return pd.DataFrame()

            # Normalize
            df.index = pd.to_datetime(df.index, utc=True)
            df.index = df.index.tz_convert("UTC").tz_localize(None)
            df.columns = [c.lower() for c in df.columns]

            # Store to DB
            stored = self._store_candles(ticker, timeframe, df)
            logger.debug("Stored {} candles for {}/{}", stored, ticker, timeframe)
            self._download_count += 1
            return df

        except Exception as e:
            logger.error("Failed to download {}/{}: {}", ticker, timeframe, e)
            return pd.DataFrame()

    def _store_candles(self, ticker: str, timeframe: str, df: pd.DataFrame) -> int:
        """Insert candles into DB, ignoring duplicates."""
        if df.empty:
            return 0

        # Get existing timestamps in one query
        with db_session() as db:
            existing = (
                db.query(Candle.timestamp)
                .filter_by(ticker=ticker, timeframe=timeframe)
                .all()
            )
        existing_set = {ts[0] for ts in existing}

        stored = 0
        new_candles = []
        for ts, row in df.iterrows():
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if ts_dt not in existing_set:
                candle = Candle(
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=ts_dt,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                )
                new_candles.append(candle)
                stored += 1

        if new_candles:
            with db_session() as db:
                db.add_all(new_candles)

        return stored

    def _get_latest_stored_ts(self, ticker: str, timeframe: str) -> Optional[datetime]:
        """Returns the most recent timestamp stored for a ticker/timeframe."""
        with db_session() as db:
            candle = (
                db.query(Candle)
                .filter_by(ticker=ticker, timeframe=timeframe)
                .order_by(Candle.timestamp.desc())
                .first()
            )
            return candle.timestamp if candle else None

    def _needs_update(self, df: pd.DataFrame, ticker: str, timeframe: str) -> bool:
        """Determine if we should fetch new data."""
        if df.empty:
            return True
        last_ts = df.index[-1]
        if timeframe in ("1d", "1wk"):
            return (datetime.utcnow() - last_ts).days >= 1
        else:
            return (datetime.utcnow() - last_ts).total_seconds() > 300  # 5 minutes

    def _rate_limit(self) -> None:
        """Simple rate limiter to avoid hammering yfinance."""
        elapsed = (datetime.utcnow() - self._last_download).total_seconds()
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_download = datetime.utcnow()


# ─── Singleton ────────────────────────────────────────────────────────────────
historical_data = HistoricalDataManager()

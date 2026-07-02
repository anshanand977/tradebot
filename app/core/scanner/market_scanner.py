"""
Market Scanner — Background Worker
=====================================
Continuously scans the stock universe every N seconds.
Ranks opportunities by composite score and pushes results to the API.
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger

from app.config import settings
from app.data.nse_client import nse_client, NSE_500_SYMBOLS, NIFTY50_STOCKS
from app.data.historical_data import historical_data
from app.core.strategies.strategy_manager import signal_generator, TradeRecommendation
from app.core.analysis.indicators import compute_all_indicators


class ScannerResult:
    """Result for one scanned stock."""
    def __init__(
        self,
        ticker: str,
        symbol: str,
        close: float,
        change_pct: float,
        volume_ratio: float,
        rsi: float,
        adx: float,
        recommendation: Optional[TradeRecommendation],
        scan_time: datetime,
    ):
        self.ticker = ticker
        self.symbol = symbol
        self.close = close
        self.change_pct = change_pct
        self.volume_ratio = volume_ratio
        self.rsi = rsi
        self.adx = adx
        self.recommendation = recommendation
        self.scan_time = scan_time

    @property
    def composite_score(self) -> float:
        """
        0–100 score for ranking. Higher = better opportunity.
        Factors: confidence, R:R, volume, ADX strength.
        """
        if not self.recommendation or not self.recommendation.is_actionable:
            return 0.0
        rec = self.recommendation
        conf_score   = rec.confidence * 30
        rr_score     = min(rec.risk_reward / 5 * 20, 20)
        vol_score    = min(self.volume_ratio / 3 * 20, 20)
        adx_score    = min(self.adx / 50 * 15, 15)
        prob_score   = rec.probability * 15
        return round(conf_score + rr_score + vol_score + adx_score + prob_score, 2)

    def to_dict(self) -> Dict:
        rec = self.recommendation
        return {
            "ticker":          self.ticker,
            "symbol":          self.symbol,
            "close":           round(float(self.close), 2),
            "change_pct":      round(float(self.change_pct), 2),
            "volume_ratio":    round(float(self.volume_ratio), 2),
            "rsi":             round(float(self.rsi), 1),
            "adx":             round(float(self.adx), 1),
            "composite_score": float(self.composite_score),
            "scan_time":       self.scan_time.isoformat(),
            "has_signal":      bool(rec is not None and rec.is_actionable),
            "direction":       rec.direction if rec else "NO_TRADE",
            "confidence":      round(float(rec.confidence) * 100, 1) if rec else 0.0,
            "probability":     round(float(rec.probability) * 100, 1) if rec else 0.0,
            "entry":           round(float(rec.entry_price), 2) if (rec and rec.entry_price is not None) else 0.0,
            "stop_loss":       round(float(rec.stop_loss), 2) if (rec and rec.stop_loss is not None) else 0.0,
            "target1":         round(float(rec.target1), 2) if (rec and rec.target1 is not None) else 0.0,
            "target2":         round(float(rec.target2), 2) if (rec and rec.target2 is not None) else 0.0,
            "target3":         round(float(rec.target3), 2) if (rec and rec.target3 is not None) else 0.0,
            "risk_reward":     round(float(rec.risk_reward), 2) if (rec and rec.risk_reward is not None) else 0.0,
            "risk_level":      rec.risk_level if rec else "",
            "strategies_agreed": int(rec.strategies_agreed) if rec else 0,
            "total_strategies":  int(rec.total_strategies_voted) if rec else 0,
            "signals":         rec.contributing_signals[:5] if rec else [],
            "pattern":         rec.pattern_context if rec else "",
            "market_regime":   rec.market_regime if rec else "",
        }


class MarketScanner:
    """
    Background scanner that analyses the stock universe continuously.
    Uses a thread pool for parallel analysis.
    """

    def __init__(self, timeframe: str = "1d", max_workers: int = 8):
        self.timeframe = timeframe
        self.max_workers = max_workers
        self._results: List[ScannerResult] = []
        self._is_running = False
        self._scan_count = 0
        self._last_scan_time: Optional[datetime] = None
        self._on_signal_callbacks: List[Callable] = []

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_results(self, limit: int = 50, only_signals: bool = False) -> List[Dict]:
        """Returns ranked scan results."""
        results = sorted(self._results, key=lambda x: x.composite_score, reverse=True)
        if only_signals:
            results = [r for r in results if r.has_signal]
        return [r.to_dict() for r in results[:limit]]

    def get_top_signals(self, n: int = 10) -> List[Dict]:
        return self.get_results(limit=n, only_signals=True)

    def scan_watchlist(self, tickers: List[str]) -> List[Dict]:
        """Immediate scan of a specific ticker list."""
        return self._scan_batch(tickers)

    def scan_single(self, ticker: str) -> Optional[Dict]:
        """Scan a single ticker synchronously."""
        result = self._scan_ticker(ticker)
        if result:
            return result.to_dict()
        return None

    def add_signal_callback(self, callback: Callable) -> None:
        """Register a callback to be called when a new signal is generated."""
        self._on_signal_callbacks.append(callback)

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def last_scan_time(self) -> Optional[datetime]:
        return self._last_scan_time

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ─── Scanner Loop ─────────────────────────────────────────────────────────

    def run_full_scan(self, universe: Optional[List[str]] = None) -> int:
        """
        Scan the full universe synchronously.
        Returns number of signals found.
        """
        if universe is None:
            # Start with NIFTY50 for speed, expand to full universe
            universe_symbols = NIFTY50_STOCKS + [
                s for s in NSE_500_SYMBOLS if s not in NIFTY50_STOCKS
            ]
            universe_symbols = universe_symbols[:settings.SCANNER_UNIVERSE_SIZE]
        else:
            universe_symbols = [s.replace(".NS", "") for s in universe]

        tickers = [f"{s}.NS" for s in universe_symbols]
        logger.info("🔍 Starting full scan: {} stocks on {}", len(tickers), self.timeframe)

        scan_time = datetime.utcnow()
        results = self._scan_batch(tickers, scan_time)
        self._scan_count += 1
        self._last_scan_time = scan_time

        signals = [r for r in results if r.get("has_signal")]
        logger.success("✅ Scan complete: {} signals from {} stocks", len(signals), len(tickers))

        # Trigger callbacks for new signals
        for sig in signals[:5]:  # Top 5 signals
            for cb in self._on_signal_callbacks:
                try:
                    cb(sig)
                except Exception:
                    pass

        return len(signals)

    def _scan_batch(self, tickers: List[str], scan_time: datetime = None) -> List[Dict]:
        """Scan a batch of tickers in parallel."""
        if scan_time is None:
            scan_time = datetime.utcnow()
        self._results = []
        results_dicts = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._scan_ticker, t, scan_time): t for t in tickers}
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        self._results.append(result)
                        results_dicts.append(result.to_dict())
                except Exception as e:
                    ticker = futures[future]
                    logger.debug("Scan failed for {}: {}", ticker, e)

        return results_dicts

    def _scan_ticker(self, ticker: str, scan_time: datetime = None) -> Optional[ScannerResult]:
        """Analyze a single ticker and return a ScannerResult."""
        try:
            # Load data
            df = historical_data.get_candles(ticker, self.timeframe, periods=300)
            if df.empty or len(df) < 30:
                return None

            close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2] if len(df) > 1 else close
            change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0

            # Quick indicator check for filtering
            indicators = compute_all_indicators(df)
            rsi = indicators.get("rsi_14", 50)
            adx = indicators.get("adx", 20)
            vol_ratio = indicators.get("vol_ratio", 1.0)

            # Full signal generation
            symbol = ticker.replace(".NS", "").replace(".BO", "")
            rec = signal_generator.generate(
                df=df,
                ticker=ticker,
                symbol=symbol,
                exchange="NSE",
                timeframe=self.timeframe,
            )

            return ScannerResult(
                ticker=ticker,
                symbol=symbol,
                close=close,
                change_pct=change_pct,
                volume_ratio=vol_ratio,
                rsi=rsi,
                adx=adx,
                recommendation=rec if rec.is_actionable else None,
                scan_time=scan_time or datetime.utcnow(),
            )
        except Exception as e:
            logger.debug("Error scanning {}: {}", ticker, e)
            return None


# Filter helpers
def filter_top_gainers(results: List[Dict], n: int = 10) -> List[Dict]:
    return sorted(results, key=lambda x: x.get("change_pct", 0), reverse=True)[:n]

def filter_top_losers(results: List[Dict], n: int = 10) -> List[Dict]:
    return sorted(results, key=lambda x: x.get("change_pct", 0))[:n]

def filter_high_volume(results: List[Dict], n: int = 10) -> List[Dict]:
    return sorted(results, key=lambda x: x.get("volume_ratio", 0), reverse=True)[:n]

def filter_breakout(results: List[Dict]) -> List[Dict]:
    return [r for r in results if r.get("direction") in ("BUY", "SELL") and r.get("volume_ratio", 0) > 2]


market_scanner = MarketScanner()

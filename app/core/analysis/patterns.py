"""
Chart Pattern & Candlestick Pattern Recognition
=================================================
Detects both candlestick patterns (single/multi-bar) and
chart patterns (geometric, using swing high/low analysis).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class CandlestickPattern:
    name: str
    type: str          # "BULLISH" / "BEARISH" / "NEUTRAL"
    strength: float    # 0.0 - 1.0
    description: str


@dataclass
class ChartPattern:
    name: str
    type: str          # "BULLISH" / "BEARISH" / "NEUTRAL"
    breakout_price: Optional[float]
    target_price: Optional[float]
    confidence: float
    description: str


# ═══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(df: pd.DataFrame) -> List[CandlestickPattern]:
    """Detect all candlestick patterns on the last few candles."""
    patterns = []
    if len(df) < 3:
        return patterns

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    i = len(df) - 1  # Last candle index

    body = abs(c[i] - o[i])
    total_range = h[i] - l[i] + 1e-10
    upper_wick = h[i] - max(o[i], c[i])
    lower_wick = min(o[i], c[i]) - l[i]
    body_pct = body / total_range
    is_bullish = c[i] > o[i]

    # ─── Single Candle Patterns ───────────────────────────────────────────────

    # Doji
    if body_pct < 0.1:
        patterns.append(CandlestickPattern(
            name="Doji",
            type="NEUTRAL",
            strength=0.6,
            description="Indecision: opening and closing are nearly equal"
        ))

    # Hammer (bullish)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and is_bullish:
        patterns.append(CandlestickPattern(
            name="Hammer",
            type="BULLISH",
            strength=0.75,
            description="Strong buying pressure; potential reversal from downtrend"
        ))

    # Shooting Star (bearish)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and not is_bullish:
        patterns.append(CandlestickPattern(
            name="Shooting Star",
            type="BEARISH",
            strength=0.75,
            description="Rejection at highs; potential reversal from uptrend"
        ))

    # Inverted Hammer (bullish, needs confirmation)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and is_bullish:
        patterns.append(CandlestickPattern(
            name="Inverted Hammer",
            type="BULLISH",
            strength=0.6,
            description="Potential bullish reversal; needs confirmation"
        ))

    # Marubozu Bullish
    if is_bullish and upper_wick < body * 0.05 and lower_wick < body * 0.05:
        patterns.append(CandlestickPattern(
            name="Bullish Marubozu",
            type="BULLISH",
            strength=0.85,
            description="Strong bullish momentum; no wicks = full conviction"
        ))

    # Marubozu Bearish
    if not is_bullish and upper_wick < body * 0.05 and lower_wick < body * 0.05:
        patterns.append(CandlestickPattern(
            name="Bearish Marubozu",
            type="BEARISH",
            strength=0.85,
            description="Strong bearish momentum; no wicks = full conviction"
        ))

    # ─── Two-Candle Patterns ─────────────────────────────────────────────────

    if i >= 1:
        po, ph, pl, pc = o[i-1], h[i-1], l[i-1], c[i-1]
        prev_bullish = pc > po
        prev_body = abs(pc - po)

        # Bullish Engulfing
        if (not prev_bullish and is_bullish and
                o[i] <= pc and c[i] >= po and body > prev_body * 1.1):
            patterns.append(CandlestickPattern(
                name="Bullish Engulfing",
                type="BULLISH",
                strength=0.85,
                description="Bullish candle fully engulfs the prior bearish candle"
            ))

        # Bearish Engulfing
        if (prev_bullish and not is_bullish and
                o[i] >= pc and c[i] <= po and body > prev_body * 1.1):
            patterns.append(CandlestickPattern(
                name="Bearish Engulfing",
                type="BEARISH",
                strength=0.85,
                description="Bearish candle fully engulfs the prior bullish candle"
            ))

        # Bullish Harami
        if (not prev_bullish and is_bullish and
                o[i] > pc and c[i] < po and body < prev_body * 0.6):
            patterns.append(CandlestickPattern(
                name="Bullish Harami",
                type="BULLISH",
                strength=0.65,
                description="Small bullish candle within a large bearish candle"
            ))

        # Bearish Harami
        if (prev_bullish and not is_bullish and
                o[i] < pc and c[i] > po and body < prev_body * 0.6):
            patterns.append(CandlestickPattern(
                name="Bearish Harami",
                type="BEARISH",
                strength=0.65,
                description="Small bearish candle within a large bullish candle"
            ))

        # Inside Bar
        if h[i] < ph and l[i] > pl:
            patterns.append(CandlestickPattern(
                name="Inside Bar",
                type="NEUTRAL",
                strength=0.55,
                description="Consolidation / coiling within prior candle's range"
            ))

    # ─── Three-Candle Patterns ────────────────────────────────────────────────

    if i >= 2:
        po2, ph2, pl2, pc2 = o[i-2], h[i-2], l[i-2], c[i-2]
        po1, ph1, pl1, pc1 = o[i-1], h[i-1], l[i-1], c[i-1]

        # Morning Star (bullish reversal)
        if (pc2 < po2 and                           # Day 1: bearish
                abs(pc1 - po1) < abs(pc2 - po2) * 0.5 and  # Day 2: small body
                is_bullish and c[i] > (po2 + pc2) / 2):    # Day 3: bullish > midpoint
            patterns.append(CandlestickPattern(
                name="Morning Star",
                type="BULLISH",
                strength=0.88,
                description="3-candle bullish reversal pattern"
            ))

        # Evening Star (bearish reversal)
        if (pc2 > po2 and                           # Day 1: bullish
                abs(pc1 - po1) < abs(pc2 - po2) * 0.5 and  # Day 2: small body
                not is_bullish and c[i] < (po2 + pc2) / 2):  # Day 3: bearish below midpoint
            patterns.append(CandlestickPattern(
                name="Evening Star",
                type="BEARISH",
                strength=0.88,
                description="3-candle bearish reversal pattern"
            ))

        # Three White Soldiers
        if (c[i-2] > o[i-2] and c[i-1] > o[i-1] and is_bullish and
                c[i] > c[i-1] > c[i-2] and
                o[i] > o[i-1] > o[i-2]):
            patterns.append(CandlestickPattern(
                name="Three White Soldiers",
                type="BULLISH",
                strength=0.90,
                description="3 consecutive bullish candles; strong momentum"
            ))

        # Three Black Crows
        if (c[i-2] < o[i-2] and c[i-1] < o[i-1] and not is_bullish and
                c[i] < c[i-1] < c[i-2] and
                o[i] < o[i-1] < o[i-2]):
            patterns.append(CandlestickPattern(
                name="Three Black Crows",
                type="BEARISH",
                strength=0.90,
                description="3 consecutive bearish candles; strong momentum"
            ))

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# CHART PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_swing_highs_lows(df: pd.DataFrame, n: int = 3) -> tuple:
    """Simple swing high/low extraction for chart patterns."""
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if df["high"].iloc[i] == df["high"].iloc[i-n:i+n+1].max():
            highs.append((df.index[i], df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i-n:i+n+1].min():
            lows.append((df.index[i], df["low"].iloc[i]))
    return highs, lows


def detect_chart_patterns(df: pd.DataFrame) -> List[ChartPattern]:
    """Detect major chart patterns using swing high/low geometry."""
    patterns = []
    if len(df) < 30:
        return patterns

    close = df["close"].iloc[-1]
    highs, lows = _find_swing_highs_lows(df, n=3)

    if len(highs) < 3 or len(lows) < 3:
        return patterns

    # Extract just values
    h_vals = [h[1] for h in highs[-6:]]
    l_vals = [l[1] for l in lows[-6:]]

    # ─── Double Top ───────────────────────────────────────────────────────────
    if len(h_vals) >= 2:
        h1, h2 = h_vals[-2], h_vals[-1]
        if abs(h1 - h2) / h1 < 0.015:   # Tops within 1.5%
            neckline = min(l_vals[-2:]) if len(l_vals) >= 2 else close
            if close < neckline:
                target = neckline - (h1 - neckline)
                patterns.append(ChartPattern(
                    name="Double Top",
                    type="BEARISH",
                    breakout_price=neckline,
                    target_price=target,
                    confidence=0.75,
                    description=f"Double top at ~₹{h1:.0f}; neckline break confirms reversal"
                ))

    # ─── Double Bottom ────────────────────────────────────────────────────────
    if len(l_vals) >= 2:
        l1, l2 = l_vals[-2], l_vals[-1]
        if abs(l1 - l2) / l1 < 0.015:
            neckline = max(h_vals[-2:]) if len(h_vals) >= 2 else close
            if close > neckline:
                target = neckline + (neckline - l1)
                patterns.append(ChartPattern(
                    name="Double Bottom",
                    type="BULLISH",
                    breakout_price=neckline,
                    target_price=target,
                    confidence=0.75,
                    description=f"Double bottom at ~₹{l1:.0f}; neckline break confirms reversal"
                ))

    # ─── Higher Highs / Lower Lows (Trend) ───────────────────────────────────
    if len(h_vals) >= 3 and h_vals[-1] > h_vals[-2] > h_vals[-3]:
        if len(l_vals) >= 3 and l_vals[-1] > l_vals[-2] > l_vals[-3]:
            patterns.append(ChartPattern(
                name="Ascending Channel",
                type="BULLISH",
                breakout_price=None,
                target_price=None,
                confidence=0.70,
                description="Higher highs and higher lows; established uptrend"
            ))

    if len(h_vals) >= 3 and h_vals[-1] < h_vals[-2] < h_vals[-3]:
        if len(l_vals) >= 3 and l_vals[-1] < l_vals[-2] < l_vals[-3]:
            patterns.append(ChartPattern(
                name="Descending Channel",
                type="BEARISH",
                breakout_price=None,
                target_price=None,
                confidence=0.70,
                description="Lower highs and lower lows; established downtrend"
            ))

    # ─── Symmetrical Triangle ─────────────────────────────────────────────────
    if len(h_vals) >= 3 and len(l_vals) >= 3:
        highs_declining = h_vals[-1] < h_vals[-2] < h_vals[-3]
        lows_rising     = l_vals[-1] > l_vals[-2] > l_vals[-3]
        if highs_declining and lows_rising:
            apex = (h_vals[-1] + l_vals[-1]) / 2
            patterns.append(ChartPattern(
                name="Symmetrical Triangle",
                type="NEUTRAL",
                breakout_price=h_vals[-1],
                target_price=None,
                confidence=0.65,
                description="Converging price action; breakout imminent"
            ))

    # ─── Ascending Triangle ───────────────────────────────────────────────────
    if len(h_vals) >= 3 and len(l_vals) >= 3:
        flat_highs = abs(h_vals[-1] - h_vals[-2]) / h_vals[-1] < 0.01
        lows_rising = l_vals[-1] > l_vals[-2]
        if flat_highs and lows_rising:
            patterns.append(ChartPattern(
                name="Ascending Triangle",
                type="BULLISH",
                breakout_price=h_vals[-1],
                target_price=h_vals[-1] + (h_vals[-1] - l_vals[0]),
                confidence=0.72,
                description="Flat resistance with rising support; bullish breakout expected"
            ))

    # ─── Descending Triangle ──────────────────────────────────────────────────
    if len(h_vals) >= 3 and len(l_vals) >= 3:
        flat_lows = abs(l_vals[-1] - l_vals[-2]) / l_vals[-1] < 0.01
        highs_falling = h_vals[-1] < h_vals[-2]
        if flat_lows and highs_falling:
            patterns.append(ChartPattern(
                name="Descending Triangle",
                type="BEARISH",
                breakout_price=l_vals[-1],
                target_price=l_vals[-1] - (h_vals[0] - l_vals[-1]),
                confidence=0.72,
                description="Flat support with falling resistance; bearish breakdown expected"
            ))

    # ─── Head & Shoulders ─────────────────────────────────────────────────────
    if len(h_vals) >= 5:
        left, head, right = h_vals[-5], h_vals[-3], h_vals[-1]
        if head > left and head > right and abs(left - right) / head < 0.03:
            if len(l_vals) >= 2:
                neckline = np.mean([l_vals[-2], l_vals[-1]])
                patterns.append(ChartPattern(
                    name="Head & Shoulders",
                    type="BEARISH",
                    breakout_price=neckline,
                    target_price=neckline - (head - neckline),
                    confidence=0.80,
                    description="Classic bearish reversal; head higher than both shoulders"
                ))

    # ─── Inverse Head & Shoulders ─────────────────────────────────────────────
    if len(l_vals) >= 5:
        left, head, right = l_vals[-5], l_vals[-3], l_vals[-1]
        if head < left and head < right and abs(left - right) / abs(head) < 0.03:
            if len(h_vals) >= 2:
                neckline = np.mean([h_vals[-2], h_vals[-1]])
                patterns.append(ChartPattern(
                    name="Inverse Head & Shoulders",
                    type="BULLISH",
                    breakout_price=neckline,
                    target_price=neckline + (neckline - head),
                    confidence=0.80,
                    description="Classic bullish reversal; head lower than both shoulders"
                ))

    return patterns

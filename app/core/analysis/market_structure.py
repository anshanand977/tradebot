"""
Market Structure Analysis
===========================
Detects HH, HL, LH, LL, trend direction, swing points,
and classifies the current market regime.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class MarketStructure:
    trend: str              # "BULLISH" / "BEARISH" / "RANGING"
    regime: str             # "TRENDING" / "RANGING" / "VOLATILE"
    swing_high: float
    swing_low: float
    last_structure: str     # "HH" / "HL" / "LH" / "LL"
    structure_sequence: List[str]
    trend_strength: float   # 0.0 - 1.0 (based on ADX equivalent)
    range_bound: bool
    range_top: Optional[float] = None
    range_bottom: Optional[float] = None


def detect_market_structure(df: pd.DataFrame, swing_n: int = 3) -> MarketStructure:
    """
    Analyzes price action to determine current market structure.
    """
    if len(df) < 20:
        return MarketStructure(
            trend="UNKNOWN", regime="RANGING",
            swing_high=df["high"].max(), swing_low=df["low"].min(),
            last_structure="UNKNOWN", structure_sequence=[],
            trend_strength=0.0, range_bound=True,
        )

    # Find pivot highs and lows
    pivot_highs: List[float] = []
    pivot_lows: List[float] = []

    for i in range(swing_n, len(df) - swing_n):
        if df["high"].iloc[i] == df["high"].iloc[i-swing_n:i+swing_n+1].max():
            pivot_highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == df["low"].iloc[i-swing_n:i+swing_n+1].min():
            pivot_lows.append(df["low"].iloc[i])

    structure_labels = []
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        # Analyze last 4 swings
        h1, h2 = pivot_highs[-2], pivot_highs[-1]
        l1, l2 = pivot_lows[-2], pivot_lows[-1]

        if h2 > h1:
            structure_labels.append("HH")
        else:
            structure_labels.append("LH")

        if l2 > l1:
            structure_labels.append("HL")
        else:
            structure_labels.append("LL")

    # Determine trend from structure
    hh_count = structure_labels.count("HH") + structure_labels.count("HL")
    ll_count = structure_labels.count("LH") + structure_labels.count("LL")

    if hh_count > ll_count:
        trend = "BULLISH"
    elif ll_count > hh_count:
        trend = "BEARISH"
    else:
        trend = "RANGING"

    # ADX-equivalent: linear regression slope as strength
    if len(df) >= 14:
        y = df["close"].tail(14).values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = abs(slope) / (df["close"].mean() + 1e-10) * 100
        trend_strength = min(1.0, normalized_slope * 5)
    else:
        trend_strength = 0.5

    # Volatility / Regime
    returns = df["close"].pct_change().tail(20)
    volatility = returns.std() * np.sqrt(252)  # Annualized
    atr_14 = (df["high"] - df["low"]).tail(14).mean() / df["close"].mean() * 100

    if atr_14 > 3.0:
        regime = "VOLATILE"
    elif trend_strength > 0.5:
        regime = "TRENDING"
    else:
        regime = "RANGING"

    # Range detection
    last_50 = df.tail(50)
    price_range_pct = (last_50["high"].max() - last_50["low"].min()) / last_50["close"].mean() * 100
    range_bound = price_range_pct < 8.0  # Less than 8% range = range-bound

    return MarketStructure(
        trend=trend,
        regime=regime,
        swing_high=pivot_highs[-1] if pivot_highs else df["high"].max(),
        swing_low=pivot_lows[-1] if pivot_lows else df["low"].min(),
        last_structure=structure_labels[-1] if structure_labels else "UNKNOWN",
        structure_sequence=structure_labels,
        trend_strength=round(trend_strength, 3),
        range_bound=range_bound,
        range_top=last_50["high"].max() if range_bound else None,
        range_bottom=last_50["low"].min() if range_bound else None,
    )

"""
Smart Money Concepts (SMC) Analysis
======================================
Detects institutional-level market structures:
  - Order Blocks (OB)
  - Fair Value Gaps (FVG)
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Liquidity Sweeps
  - Premium / Discount Zones
  - Mitigation Blocks
  - Breaker Blocks
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from loguru import logger


@dataclass
class OrderBlock:
    type: str           # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    timestamp: object
    strength: float     # 0.0 - 1.0 based on size and move after
    mitigated: bool = False
    is_breaker: bool = False   # True if price came back and failed

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class FairValueGap:
    type: str           # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    timestamp: object
    filled: bool = False

    @property
    def size_pct(self) -> float:
        return abs(self.top - self.bottom) / self.bottom * 100


@dataclass
class LiquiditySweep:
    type: str           # "BSL" (buyside) or "SSL" (sellside)
    level: float
    timestamp: object
    reversal_followed: bool = False


@dataclass
class SMCContext:
    """Full Smart Money Context for a ticker at a given moment."""
    ticker: str
    timeframe: str
    timestamp: object

    # Structure
    market_structure: str = "UNKNOWN"   # BULLISH / BEARISH / RANGING
    last_bos: Optional[str] = None       # "BULLISH" / "BEARISH"
    last_choch: Optional[str] = None

    # Zones
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)
    liquidity_sweeps: List[LiquiditySweep] = field(default_factory=list)

    # Bias
    bias: str = "NEUTRAL"               # BULLISH / BEARISH / NEUTRAL
    premium_discount: str = "DISCOUNT"  # PREMIUM / EQUILIBRIUM / DISCOUNT
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

    # Nearby zones
    nearest_ob_bullish: Optional[OrderBlock] = None
    nearest_ob_bearish: Optional[OrderBlock] = None
    nearest_fvg_bullish: Optional[FairValueGap] = None
    nearest_fvg_bearish: Optional[FairValueGap] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SWING HIGH / LOW DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, strength: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Find swing highs and swing lows.
    strength: number of candles on each side that must be lower/higher.
    Returns (swing_highs, swing_lows) as boolean Series.
    """
    n = len(df)
    swing_highs = pd.Series(False, index=df.index)
    swing_lows  = pd.Series(False, index=df.index)

    for i in range(strength, n - strength):
        window_high = df["high"].iloc[i - strength:i + strength + 1]
        window_low  = df["low"].iloc[i - strength:i + strength + 1]

        if df["high"].iloc[i] == window_high.max():
            swing_highs.iloc[i] = True
        if df["low"].iloc[i] == window_low.min():
            swing_lows.iloc[i] = True

    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════════════════════
# BREAK OF STRUCTURE & CHANGE OF CHARACTER
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bos_choch(df: pd.DataFrame, swing_strength: int = 3) -> List[Dict]:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).

    BOS = continuation: price breaks in the direction of the current trend.
    CHoCH = reversal: price breaks against the current trend.

    Returns list of events with: type, direction, price, timestamp
    """
    swing_highs, swing_lows = find_swing_points(df, swing_strength)
    events = []

    sh_levels = df["high"][swing_highs].tolist()
    sl_levels = df["low"][swing_lows].tolist()
    sh_times  = df.index[swing_highs].tolist()
    sl_times  = df.index[swing_lows].tolist()

    if not sh_levels or not sl_levels:
        return events

    # Track trend direction
    last_sh = sh_levels[-1] if sh_levels else None
    last_sl = sl_levels[-1] if sl_levels else None
    prev_sh = sh_levels[-2] if len(sh_levels) > 1 else None
    prev_sl = sl_levels[-2] if len(sl_levels) > 1 else None

    current_close = df["close"].iloc[-1]

    # BOS Bullish: close breaks above last swing high
    if last_sh and current_close > last_sh:
        trend_before = "BULLISH" if (prev_sh and last_sh > prev_sh) else "BEARISH"
        event_type = "BOS" if trend_before == "BULLISH" else "CHoCH"
        events.append({
            "type": event_type,
            "direction": "BULLISH",
            "price": last_sh,
            "timestamp": sh_times[-1] if sh_times else None,
            "confirmed": True,
        })

    # BOS Bearish: close breaks below last swing low
    if last_sl and current_close < last_sl:
        trend_before = "BEARISH" if (prev_sl and last_sl < prev_sl) else "BULLISH"
        event_type = "BOS" if trend_before == "BEARISH" else "CHoCH"
        events.append({
            "type": event_type,
            "direction": "BEARISH",
            "price": last_sl,
            "timestamp": sl_times[-1] if sl_times else None,
            "confirmed": True,
        })

    return events


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> List[OrderBlock]:
    """
    Detect Order Blocks: the last opposing candle before a strong move (BOS).

    Bullish OB: The last bearish candle before a bullish impulse that broke structure.
    Bearish OB: The last bullish candle before a bearish impulse that broke structure.
    """
    obs = []
    if len(df) < 10:
        return obs

    close_price = df["close"].iloc[-1]
    lookback = min(lookback, len(df) - 2)

    for i in range(lookback, 2, -1):
        # Detect strong bullish impulse (3 consecutive bullish candles with high volume)
        future_slice = df.iloc[i:i + 5]
        if len(future_slice) < 3:
            continue

        future_returns = future_slice["close"].pct_change().sum()
        avg_future_vol = future_slice["volume"].mean()
        avg_vol = df["volume"].rolling(20).mean().iloc[i]

        # Bullish OB: bearish candle before a bullish impulse
        if (future_returns > 0.015 and  # >1.5% move
                avg_future_vol > avg_vol * 1.5 and  # Volume spike
                df["close"].iloc[i] < df["open"].iloc[i]):  # Bearish candle

            ob = OrderBlock(
                type="BULLISH",
                top=df["open"].iloc[i],
                bottom=df["low"].iloc[i],
                timestamp=df.index[i],
                strength=min(1.0, future_returns * 20),
                mitigated=close_price < df["low"].iloc[i],
            )
            # Check if it's a breaker (mitigated and then failed)
            ob.is_breaker = ob.mitigated and close_price < ob.bottom
            obs.append(ob)

        # Bearish OB: bullish candle before a bearish impulse
        elif (future_returns < -0.015 and
              avg_future_vol > avg_vol * 1.5 and
              df["close"].iloc[i] > df["open"].iloc[i]):

            ob = OrderBlock(
                type="BEARISH",
                top=df["high"].iloc[i],
                bottom=df["close"].iloc[i],
                timestamp=df.index[i],
                strength=min(1.0, abs(future_returns) * 20),
                mitigated=close_price > df["high"].iloc[i],
            )
            ob.is_breaker = ob.mitigated and close_price > ob.top
            obs.append(ob)

    return obs


# ═══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 30,
                            min_gap_pct: float = 0.3) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (FVG) — 3-candle imbalances.

    Bullish FVG: low[i+1] > high[i-1]  (gap between candle 1 top and candle 3 bottom)
    Bearish FVG: high[i+1] < low[i-1]
    """
    fvgs = []
    lookback = min(lookback, len(df) - 3)
    close_price = df["close"].iloc[-1]

    for i in range(1, lookback + 1):
        idx = len(df) - 1 - i

        if idx < 1 or idx + 1 >= len(df):
            continue

        c1_high = df["high"].iloc[idx - 1]
        c1_low  = df["low"].iloc[idx - 1]
        c3_high = df["high"].iloc[idx + 1]
        c3_low  = df["low"].iloc[idx + 1]

        # Bullish FVG
        if c3_low > c1_high:
            gap_pct = (c3_low - c1_high) / c1_high * 100
            if gap_pct >= min_gap_pct:
                fvg = FairValueGap(
                    type="BULLISH",
                    top=c3_low,
                    bottom=c1_high,
                    timestamp=df.index[idx],
                    filled=close_price < c1_high,
                )
                fvgs.append(fvg)

        # Bearish FVG
        elif c3_high < c1_low:
            gap_pct = (c1_low - c3_high) / c1_low * 100
            if gap_pct >= min_gap_pct:
                fvg = FairValueGap(
                    type="BEARISH",
                    top=c1_low,
                    bottom=c3_high,
                    timestamp=df.index[idx],
                    filled=close_price > c1_low,
                )
                fvgs.append(fvg)

    return fvgs


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 30) -> List[LiquiditySweep]:
    """
    Detect liquidity sweeps: equal highs/lows that get taken out then reversed.
    BSL (Buy-Side Liquidity): Equal highs swept, then sharp reversal down
    SSL (Sell-Side Liquidity): Equal lows swept, then sharp reversal up
    """
    sweeps = []
    if len(df) < 10:
        return sweeps

    tol = df["close"].iloc[-1] * 0.002  # 0.2% tolerance for "equal" levels

    for i in range(5, min(lookback, len(df) - 3)):
        idx = len(df) - i

        # Look for equal highs (buyside liquidity)
        prev_high = df["high"].iloc[idx - 3:idx].max()
        curr_high = df["high"].iloc[idx]
        if abs(curr_high - prev_high) < tol and curr_high > prev_high:
            # Check for reversal after sweep
            next_close = df["close"].iloc[idx + 1] if idx + 1 < len(df) else curr_high
            reversed_down = next_close < prev_high * 0.998
            sweeps.append(LiquiditySweep(
                type="BSL",
                level=curr_high,
                timestamp=df.index[idx],
                reversal_followed=reversed_down,
            ))

        # Look for equal lows (sellside liquidity)
        prev_low = df["low"].iloc[idx - 3:idx].min()
        curr_low = df["low"].iloc[idx]
        if abs(curr_low - prev_low) < tol and curr_low < prev_low:
            next_close = df["close"].iloc[idx + 1] if idx + 1 < len(df) else curr_low
            reversed_up = next_close > prev_low * 1.002
            sweeps.append(LiquiditySweep(
                type="SSL",
                level=curr_low,
                timestamp=df.index[idx],
                reversal_followed=reversed_up,
            ))

    return sweeps


# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM / DISCOUNT
# ═══════════════════════════════════════════════════════════════════════════════

def get_premium_discount(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    Determine if price is in Premium (above 50% of swing range) or Discount.
    Equilibrium = exactly at 50%.
    """
    window = df.tail(lookback)
    swing_high = window["high"].max()
    swing_low  = window["low"].min()
    close      = df["close"].iloc[-1]
    mid        = (swing_high + swing_low) / 2

    if close > mid * 1.005:
        return "PREMIUM"
    elif close < mid * 0.995:
        return "DISCOUNT"
    else:
        return "EQUILIBRIUM"


# ═══════════════════════════════════════════════════════════════════════════════
# FULL SMC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_smc(df: pd.DataFrame, ticker: str, timeframe: str) -> SMCContext:
    """
    Runs the full SMC analysis pipeline and returns a complete SMCContext.
    """
    ctx = SMCContext(
        ticker=ticker,
        timeframe=timeframe,
        timestamp=df.index[-1] if not df.empty else None,
    )

    if len(df) < 20:
        return ctx

    close = df["close"].iloc[-1]

    try:
        # Order Blocks
        ctx.order_blocks = detect_order_blocks(df)
        bullish_obs = [ob for ob in ctx.order_blocks if ob.type == "BULLISH" and not ob.mitigated]
        bearish_obs = [ob for ob in ctx.order_blocks if ob.type == "BEARISH" and not ob.mitigated]

        # Nearest OBs to current price
        if bullish_obs:
            ctx.nearest_ob_bullish = max(
                [ob for ob in bullish_obs if ob.top < close],
                key=lambda x: x.top, default=None
            )
        if bearish_obs:
            ctx.nearest_ob_bearish = min(
                [ob for ob in bearish_obs if ob.bottom > close],
                key=lambda x: x.bottom, default=None
            )

        # FVGs
        ctx.fair_value_gaps = detect_fair_value_gaps(df)
        unfilled_bullish_fvgs = [f for f in ctx.fair_value_gaps if f.type == "BULLISH" and not f.filled]
        unfilled_bearish_fvgs = [f for f in ctx.fair_value_gaps if f.type == "BEARISH" and not f.filled]

        if unfilled_bullish_fvgs:
            ctx.nearest_fvg_bullish = max(
                [f for f in unfilled_bullish_fvgs if f.top < close],
                key=lambda x: x.top, default=None
            )
        if unfilled_bearish_fvgs:
            ctx.nearest_fvg_bearish = min(
                [f for f in unfilled_bearish_fvgs if f.bottom > close],
                key=lambda x: x.bottom, default=None
            )

        # Liquidity Sweeps
        ctx.liquidity_sweeps = detect_liquidity_sweeps(df)

        # BOS / CHoCH
        events = detect_bos_choch(df)
        if events:
            latest = events[-1]
            if latest["type"] == "BOS":
                ctx.last_bos = latest["direction"]
                ctx.market_structure = latest["direction"]
            elif latest["type"] == "CHoCH":
                ctx.last_choch = latest["direction"]

        # Premium / Discount
        ctx.premium_discount = get_premium_discount(df)

        # Swing highs/lows
        sh, sl = find_swing_points(df)
        sh_vals = df["high"][sh]
        sl_vals = df["low"][sl]
        ctx.swing_high = float(sh_vals.iloc[-1]) if not sh_vals.empty else None
        ctx.swing_low  = float(sl_vals.iloc[-1]) if not sl_vals.empty else None

        # Overall Bias
        bullish_score = 0
        bearish_score = 0

        if ctx.market_structure == "BULLISH":
            bullish_score += 2
        elif ctx.market_structure == "BEARISH":
            bearish_score += 2

        if ctx.premium_discount == "DISCOUNT":
            bullish_score += 1
        elif ctx.premium_discount == "PREMIUM":
            bearish_score += 1

        if ctx.nearest_ob_bullish:
            bullish_score += 1
        if ctx.nearest_ob_bearish:
            bearish_score += 1

        if bullish_score > bearish_score + 1:
            ctx.bias = "BULLISH"
        elif bearish_score > bullish_score + 1:
            ctx.bias = "BEARISH"
        else:
            ctx.bias = "NEUTRAL"

    except Exception as e:
        logger.warning("SMC analysis error for {}: {}", ticker, e)

    return ctx

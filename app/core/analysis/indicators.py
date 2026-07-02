"""
Technical Indicators Engine
=============================
Pure pandas/numpy implementation of all required indicators.
Every function returns a named Series or DataFrame column.
No TA-Lib dependency — uses pandas-ta for reliability.

All functions:
  - Accept a DataFrame with [open, high, low, close, volume] columns
  - Return Series/DataFrame that can be appended to the input DataFrame
  - Handle NaN values gracefully
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from loguru import logger


@dataclass
class IndicatorResult:
    """Typed result from indicator computation."""
    name: str
    value: Any
    signal: str        # "BULLISH" / "BEARISH" / "NEUTRAL"
    strength: float    # 0.0 - 1.0
    description: str = ""
    raw: Optional[pd.Series] = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════════

def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Exponential Moving Average."""
    cache_col = f"ema_{period}_{col}"
    if cache_col in df.columns:
        return df[cache_col]
    res = df[col].ewm(span=period, adjust=False).mean()
    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Simple Moving Average."""
    cache_col = f"sma_{period}_{col}"
    if cache_col in df.columns:
        return df[cache_col]
    res = df[col].rolling(window=period).mean()
    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.
    Resets daily (groups by date).
    """
    cache_col = "vwap"
    if cache_col in df.columns:
        return df[cache_col]

    if "volume" not in df.columns or df["volume"].sum() == 0:
        return df["close"].copy()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    # Group by date for intraday reset
    if hasattr(df.index, "date"):
        dates = pd.Series(df.index).apply(lambda x: x.date() if hasattr(x, "date") else x)
        dates.index = df.index
        groups = df.groupby(dates.values)
        vwap_series = pd.Series(index=df.index, dtype=float)
        for _, group in groups:
            tp = (group["high"] + group["low"] + group["close"]) / 3
            cum_tp_vol = (tp * group["volume"]).cumsum()
            cum_vol = group["volume"].cumsum()
            vwap_series.loc[group.index] = cum_tp_vol / cum_vol
        res = vwap_series
    else:
        res = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def hull_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Hull Moving Average — faster and smoother than EMA."""
    cache_col = f"hma_{period}"
    if cache_col in df.columns:
        return df[cache_col]

    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma_half = df["close"].ewm(span=half, adjust=False).mean()
    wma_full = df["close"].ewm(span=period, adjust=False).mean()
    diff = 2 * wma_half - wma_full
    res = diff.ewm(span=sqrt, adjust=False).mean()

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERTREND
# ═══════════════════════════════════════════════════════════════════════════════

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Returns DataFrame with columns: [supertrend, direction]
    direction: 1 = bullish (price above ST), -1 = bearish
    """
    cache_cols = [f"supertrend_val_{period}_{multiplier}", f"supertrend_dir_{period}_{multiplier}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "supertrend": df[cache_cols[0]],
            "direction": df[cache_cols[1]]
        }, index=df.index)

    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    final_upper = upper.copy()
    final_lower = lower.copy()
    supertrend_arr = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Upper band
        if upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band
        if lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Direction
        if supertrend_arr.iloc[i - 1] == final_upper.iloc[i - 1]:
            if df["close"].iloc[i] <= final_upper.iloc[i]:
                supertrend_arr.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend_arr.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if df["close"].iloc[i] >= final_lower.iloc[i]:
                supertrend_arr.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend_arr.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    res = pd.DataFrame({"supertrend": supertrend_arr, "direction": direction}, index=df.index)
    try:
        df[cache_cols[0]] = supertrend_arr
        df[cache_cols[1]] = direction
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# OSCILLATORS
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    cache_col = f"rsi_{period}"
    if cache_col in df.columns:
        return df[cache_col]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    res = 100 - (100 / (1 + rs))

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD - Moving Average Convergence/Divergence.
    Returns: [macd_line, signal_line, histogram]
    """
    cache_cols = [f"macd_line_{fast}_{slow}", f"macd_signal_{fast}_{slow}_{signal}", f"macd_hist_{fast}_{slow}_{signal}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "macd": df[cache_cols[0]],
            "macd_signal": df[cache_cols[1]],
            "macd_hist": df[cache_cols[2]],
        }, index=df.index)

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    res = pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=df.index)

    try:
        df[cache_cols[0]] = macd_line
        df[cache_cols[1]] = signal_line
        df[cache_cols[2]] = histogram
    except Exception:
        pass
    return res


def stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14,
              k: int = 3, d: int = 3) -> pd.DataFrame:
    """Stochastic RSI — oscillator of the RSI."""
    cache_cols = [f"stoch_k_{rsi_period}_{stoch_period}_{k}", f"stoch_d_{rsi_period}_{stoch_period}_{k}_{d}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "stoch_k": df[cache_cols[0]],
            "stoch_d": df[cache_cols[1]]
        }, index=df.index)

    rsi_vals = rsi(df, rsi_period)
    rsi_min = rsi_vals.rolling(stoch_period).min()
    rsi_max = rsi_vals.rolling(stoch_period).max()
    stoch_k = 100 * (rsi_vals - rsi_min) / (rsi_max - rsi_min + 1e-10)
    stoch_k = stoch_k.rolling(k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    res = pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=df.index)

    try:
        df[cache_cols[0]] = stoch_k
        df[cache_cols[1]] = stoch_d
    except Exception:
        pass
    return res


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    cache_col = f"cci_{period}"
    if cache_col in df.columns:
        return df[cache_col]

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_tp = typical_price.rolling(period).mean()
    mean_dev = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    res = (typical_price - mean_tp) / (0.015 * mean_dev)

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns: [adx, plus_di, minus_di]
    """
    cache_cols = [f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "adx": df[cache_cols[0]],
            "plus_di": df[cache_cols[1]],
            "minus_di": df[cache_cols[2]],
        }, index=df.index)

    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    atr14 = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr14

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx_val = dx.ewm(com=period - 1, adjust=False).mean()
    res = pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=df.index)

    try:
        df[cache_cols[0]] = adx_val
        df[cache_cols[1]] = plus_di
        df[cache_cols[2]] = minus_di
    except Exception:
        pass
    return res


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    cache_col = f"atr_{period}"
    if cache_col in df.columns:
        return df[cache_col]

    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    res = tr.ewm(com=period - 1, adjust=False).mean()

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    cache_col = "obv"
    if cache_col in df.columns:
        return df[cache_col]

    direction = np.sign(df["close"].diff()).fillna(0)
    res = (direction * df["volume"]).cumsum()

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    cache_col = f"cmf_{period}"
    if cache_col in df.columns:
        return df[cache_col]

    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10)
    res = (clv * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()

    try:
        df[cache_col] = res
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# BANDS & CHANNELS
# ═══════════════════════════════════════════════════════════════════════════════

def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    cache_cols = [f"bb_upper_{period}_{std_dev}", f"bb_mid_{period}", f"bb_lower_{period}_{std_dev}", f"bb_width_{period}_{std_dev}", f"bb_pct_{period}_{std_dev}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "bb_upper": df[cache_cols[0]],
            "bb_mid": df[cache_cols[1]],
            "bb_lower": df[cache_cols[2]],
            "bb_width": df[cache_cols[3]],
            "bb_pct": df[cache_cols[4]],
        }, index=df.index)

    mid = sma(df, period)
    std = df["close"].rolling(period).std()
    bb_upper = mid + std_dev * std
    bb_lower = mid - std_dev * std
    bb_width = (std * 2 * std_dev) / mid
    bb_pct = (df["close"] - bb_lower) / (2 * std_dev * std + 1e-10)

    res = pd.DataFrame({
        "bb_upper": bb_upper,
        "bb_mid": mid,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "bb_pct": bb_pct,
    }, index=df.index)

    try:
        df[cache_cols[0]] = bb_upper
        df[cache_cols[1]] = mid
        df[cache_cols[2]] = bb_lower
        df[cache_cols[3]] = bb_width
        df[cache_cols[4]] = bb_pct
    except Exception:
        pass
    return res


def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                      multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channels."""
    cache_cols = [f"kc_upper_{ema_period}_{atr_period}_{multiplier}", f"kc_mid_{ema_period}", f"kc_lower_{ema_period}_{atr_period}_{multiplier}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "kc_upper": df[cache_cols[0]],
            "kc_mid": df[cache_cols[1]],
            "kc_lower": df[cache_cols[2]],
        }, index=df.index)

    mid = ema(df, ema_period)
    atr_val = atr(df, atr_period)
    kc_upper = mid + multiplier * atr_val
    kc_lower = mid - multiplier * atr_val

    res = pd.DataFrame({
        "kc_upper": kc_upper,
        "kc_mid":   mid,
        "kc_lower": kc_lower,
    }, index=df.index)

    try:
        df[cache_cols[0]] = kc_upper
        df[cache_cols[1]] = mid
        df[cache_cols[2]] = kc_lower
    except Exception:
        pass
    return res


def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channels."""
    cache_cols = [f"dc_upper_{period}", f"dc_lower_{period}", f"dc_mid_{period}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "dc_upper": df[cache_cols[0]],
            "dc_lower": df[cache_cols[1]],
            "dc_mid": df[cache_cols[2]],
        }, index=df.index)

    dc_upper = df["high"].rolling(period).max()
    dc_lower = df["low"].rolling(period).min()
    dc_mid = (dc_upper + dc_lower) / 2

    res = pd.DataFrame({
        "dc_upper": dc_upper,
        "dc_lower": dc_lower,
        "dc_mid":   dc_mid,
    }, index=df.index)

    try:
        df[cache_cols[0]] = dc_upper
        df[cache_cols[1]] = dc_lower
        df[cache_cols[2]] = dc_mid
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# ICHIMOKU
# ═══════════════════════════════════════════════════════════════════════════════

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> pd.DataFrame:
    """Ichimoku Cloud."""
    cache_cols = [f"ichi_tenkan_{tenkan}", f"ichi_kijun_{kijun}", f"ichi_senkou_a_{tenkan}_{kijun}", f"ichi_senkou_b_{senkou_b}", f"ichi_chikou_{kijun}"]
    if all(col in df.columns for col in cache_cols):
        return pd.DataFrame({
            "ichi_tenkan": df[cache_cols[0]],
            "ichi_kijun":  df[cache_cols[1]],
            "ichi_senkou_a": df[cache_cols[2]],
            "ichi_senkou_b": df[cache_cols[3]],
            "ichi_chikou": df[cache_cols[4]],
        }, index=df.index)

    tenkan_sen = (df["high"].rolling(tenkan).max() + df["low"].rolling(tenkan).min()) / 2
    kijun_sen = (df["high"].rolling(kijun).max() + df["low"].rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((df["high"].rolling(senkou_b).max() + df["low"].rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = df["close"].shift(-kijun)

    res = pd.DataFrame({
        "ichi_tenkan": tenkan_sen,
        "ichi_kijun":  kijun_sen,
        "ichi_senkou_a": senkou_a,
        "ichi_senkou_b": senkou_b_val,
        "ichi_chikou": chikou,
    }, index=df.index)

    try:
        df[cache_cols[0]] = tenkan_sen
        df[cache_cols[1]] = kijun_sen
        df[cache_cols[2]] = senkou_a
        df[cache_cols[3]] = senkou_b_val
        df[cache_cols[4]] = chikou
    except Exception:
        pass
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# PIVOT POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """
    Classic Pivot Points based on the previous day's OHLC.
    Returns: {PP, R1, R2, R3, S1, S2, S3}
    """
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    h, l, c = prev["high"], prev["low"], prev["close"]
    pp = (h + l + c) / 3
    return {
        "PP": pp,
        "R1": 2 * pp - l,
        "R2": pp + (h - l),
        "R3": h + 2 * (pp - l),
        "S1": 2 * pp - h,
        "S2": pp - (h - l),
        "S3": l - 2 * (h - pp),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def support_resistance_levels(df: pd.DataFrame, lookback: int = 50,
                               min_touches: int = 2) -> Dict[str, list]:
    """
    Detect dynamic support and resistance levels from price action.
    Uses local extrema with touch counting.
    """
    highs = df["high"].values
    lows  = df["low"].values
    close = df["close"].values[-1]
    tolerance = close * 0.003   # 0.3% tolerance band

    # Find pivot highs and lows
    pivot_highs, pivot_lows = [], []
    for i in range(2, min(lookback, len(df) - 2)):
        idx = len(df) - 1 - i
        if df["high"].iloc[idx] >= df["high"].iloc[idx-2:idx+2].max() * 0.999:
            pivot_highs.append(df["high"].iloc[idx])
        if df["low"].iloc[idx] <= df["low"].iloc[idx-2:idx+2].min() * 1.001:
            pivot_lows.append(df["low"].iloc[idx])

    def cluster(prices, tol):
        if not prices:
            return []
        prices = sorted(prices, reverse=True)
        levels = []
        cluster_group = [prices[0]]
        for p in prices[1:]:
            if abs(p - cluster_group[-1]) < tol:
                cluster_group.append(p)
            else:
                levels.append(np.mean(cluster_group))
                cluster_group = [p]
        levels.append(np.mean(cluster_group))
        return levels

    resistance = [r for r in cluster(pivot_highs, tolerance) if r > close]
    support    = [s for s in cluster(pivot_lows, tolerance) if s < close]

    return {
        "resistance": sorted(resistance[:5]),
        "support":    sorted(support[-5:], reverse=True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE — Run all indicators and return a full snapshot
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes all indicators and returns a flat dictionary of current values.
    Used by strategies and the probability engine.
    """
    if len(df) < 50:
        return {}

    close = df["close"].iloc[-1]
    result = {}

    try:
        # Moving Averages
        for p in [9, 20, 50, 200]:
            result[f"ema_{p}"] = ema(df, p).iloc[-1]
            result[f"sma_{p}"] = sma(df, p).iloc[-1]

        result["vwap"] = vwap(df).iloc[-1]

        # Supertrend
        st = supertrend(df)
        result["supertrend"] = st["supertrend"].iloc[-1]
        result["supertrend_dir"] = int(st["direction"].iloc[-1])

        # Oscillators
        result["rsi_14"] = rsi(df, 14).iloc[-1]
        result["rsi_7"]  = rsi(df, 7).iloc[-1]

        macd_df = macd(df)
        result["macd"]        = macd_df["macd"].iloc[-1]
        result["macd_signal"] = macd_df["macd_signal"].iloc[-1]
        result["macd_hist"]   = macd_df["macd_hist"].iloc[-1]
        result["macd_prev_hist"] = macd_df["macd_hist"].iloc[-2] if len(df) > 1 else 0

        stoch = stoch_rsi(df)
        result["stoch_k"] = stoch["stoch_k"].iloc[-1]
        result["stoch_d"] = stoch["stoch_d"].iloc[-1]

        result["cci_20"] = cci(df, 20).iloc[-1]
        result["atr_14"] = atr(df, 14).iloc[-1]
        result["atr_pct"] = result["atr_14"] / close * 100  # ATR as % of price

        adx_df = adx(df)
        result["adx"]       = adx_df["adx"].iloc[-1]
        result["plus_di"]   = adx_df["plus_di"].iloc[-1]
        result["minus_di"]  = adx_df["minus_di"].iloc[-1]

        # Volume
        result["obv"]        = obv(df).iloc[-1]
        result["obv_prev"]   = obv(df).iloc[-2] if len(df) > 1 else 0
        result["cmf_20"]     = cmf(df, 20).iloc[-1]
        result["vol_20_avg"] = df["volume"].rolling(20).mean().iloc[-1]
        result["vol_ratio"]  = df["volume"].iloc[-1] / (result["vol_20_avg"] + 1e-10)

        # Bands
        bb = bollinger_bands(df)
        result["bb_upper"] = bb["bb_upper"].iloc[-1]
        result["bb_mid"]   = bb["bb_mid"].iloc[-1]
        result["bb_lower"] = bb["bb_lower"].iloc[-1]
        result["bb_pct"]   = bb["bb_pct"].iloc[-1]
        result["bb_width"] = bb["bb_width"].iloc[-1]

        kc = keltner_channels(df)
        result["kc_upper"] = kc["kc_upper"].iloc[-1]
        result["kc_lower"] = kc["kc_lower"].iloc[-1]

        dc = donchian_channels(df)
        result["dc_upper"] = dc["dc_upper"].iloc[-1]
        result["dc_lower"] = dc["dc_lower"].iloc[-1]

        # Ichimoku
        ichi = ichimoku(df)
        result["ichi_tenkan"]   = ichi["ichi_tenkan"].iloc[-1]
        result["ichi_kijun"]    = ichi["ichi_kijun"].iloc[-1]
        result["ichi_senkou_a"] = ichi["ichi_senkou_a"].iloc[-1]
        result["ichi_senkou_b"] = ichi["ichi_senkou_b"].iloc[-1]

        # Pivots
        result["pivots"] = pivot_points(df)

        # S/R
        sr = support_resistance_levels(df)
        result["resistance"] = sr["resistance"]
        result["support"]    = sr["support"]

        # Close relative to indicators
        result["close"]              = close
        result["above_ema_20"]       = close > result["ema_20"]
        result["above_ema_50"]       = close > result["ema_50"]
        result["above_ema_200"]      = close > result["ema_200"]
        result["above_vwap"]         = close > result["vwap"]
        result["above_ichi_cloud"]   = close > max(result["ichi_senkou_a"] or 0, result["ichi_senkou_b"] or 0)

    except Exception as e:
        logger.warning("Indicator computation partial failure: {}", e)

    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in result.items()}

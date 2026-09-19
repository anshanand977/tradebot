"""
Strategy Plugins — All 14 Trading Strategies
==============================================
Each strategy is self-contained, inherits from BaseStrategy,
and implements its own analysis logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger

from app.core.strategies.base_strategy import BaseStrategy, StrategySignal


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TREND FOLLOWING (EMA Multi-Timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

class TrendFollowingStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Trend Following"
    @property
    def description(self) -> str:
        return "EMA alignment (9/20/50/200) confirms trend direction; enter on pullbacks"
    @property
    def category(self) -> str: return "TREND"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close = indicators.get("close", df["close"].iloc[-1])
            e9   = indicators.get("ema_9",   close)
            e20  = indicators.get("ema_20",  close)
            e50  = indicators.get("ema_50",  close)
            e200 = indicators.get("ema_200", close)
            adx  = indicators.get("adx", 0)

            signals = []
            score = 0

            # Bullish alignment
            if e9 > e20 > e50 > e200:
                score += 3
                signals.append("EMA 9>20>50>200 bullish alignment")

            if close > e20:
                score += 1
                signals.append("Price above EMA 20")

            if adx > 25:
                score += 1
                signals.append(f"Strong trend ADX={adx:.1f}")

            # Bearish alignment
            if e9 < e20 < e50 < e200:
                score -= 3
                signals.append("EMA 9<20<50<200 bearish alignment")

            if close < e20:
                score -= 1

            if score >= 4:
                stop = self._atr_stop(indicators, close, "BUY", 2.0)
                targets = self._atr_targets(indicators, close, stop, "BUY")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"R:R {rr:.2f} below 1:2 minimum")
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=min(0.92, 0.60 + score * 0.06),
                    entry_price=close, stop_loss=stop, targets=targets, risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
            elif score <= -4:
                stop = self._atr_stop(indicators, close, "SELL", 2.0)
                targets = self._atr_targets(indicators, close, stop, "SELL")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"R:R {rr:.2f} below 1:2 minimum")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=min(0.92, 0.60 + abs(score) * 0.06),
                    entry_price=close, stop_loss=stop, targets=targets, risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BREAKOUT STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class BreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Breakout"
    @property
    def description(self) -> str:
        return "Detects price breaking above resistance / below support with volume"
    @property
    def category(self) -> str: return "BREAKOUT"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close = indicators.get("close", df["close"].iloc[-1])
            dc_upper = indicators.get("dc_upper", close)
            dc_lower = indicators.get("dc_lower", close)
            vol_ratio = indicators.get("vol_ratio", 1.0)
            atr = indicators.get("atr_14", close * 0.01)
            resistance = indicators.get("resistance", [])
            support = indicators.get("support", [])

            signals = []

            # Bullish breakout: price breaks Donchian high with volume
            near_resistance = any(abs(close - r) / r < 0.005 for r in resistance[:3])
            if close >= dc_upper * 0.998 and vol_ratio >= 2.0:  # Require 2x volume for breakout
                signals.append(f"Price at Donchian upper ({dc_upper:.0f})")
                signals.append(f"Volume spike {vol_ratio:.1f}x average (2x required)")
                if close > dc_upper:
                    signals.append("Confirmed breakout above 20-period high")
                stop = close - 2.0 * atr
                targets = self._atr_targets(indicators, close, stop, "BUY", [2.0, 3.0, 5.0])
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Breakout R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "BREAKOUT")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=min(0.88, 0.65 + vol_ratio * 0.04),
                    entry_price=close, stop_loss=stop, targets=targets, risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Bearish breakdown
            if close <= dc_lower * 1.002 and vol_ratio >= 2.0:  # Require 2x volume
                signals.append(f"Price at Donchian lower ({dc_lower:.0f})")
                signals.append(f"Volume spike {vol_ratio:.1f}x average")
                stop = close + 2.0 * atr
                targets = self._atr_targets(indicators, close, stop, "SELL", [2.0, 3.0, 5.0])
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Breakout R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "BREAKOUT")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=min(0.88, 0.65 + vol_ratio * 0.04),
                    entry_price=close, stop_loss=stop, targets=targets, risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PULLBACK STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class PullbackStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Pullback"
    @property
    def description(self) -> str:
        return "Enters on EMA/VWAP pullback in the direction of the dominant trend"
    @property
    def category(self) -> str: return "TREND"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close = indicators.get("close", df["close"].iloc[-1])
            e20   = indicators.get("ema_20", close)
            e50   = indicators.get("ema_50", close)
            vwap  = indicators.get("vwap", close)
            rsi   = indicators.get("rsi_14", 50)
            adx   = indicators.get("adx", 20)

            signals = []

            # Bullish trend + pullback to EMA — FIX: use ATR-based stop, not % of VWAP
            bullish_trend = e20 > e50
            at_ema20 = abs(close - e20) / e20 < 0.008
            at_vwap  = abs(close - vwap) / vwap < 0.005

            if bullish_trend and (at_ema20 or at_vwap) and rsi < 55 and rsi > 35:
                signals.append("Pullback to EMA 20 in uptrend" if at_ema20 else "Pullback to VWAP in uptrend")
                signals.append(f"RSI at {rsi:.1f} — not overbought")
                if adx > 20: signals.append(f"Trend confirmed ADX={adx:.1f}")
                # ATR-based stop — NOT percentage of EMA (previous bug: too tight for high-beta stocks)
                atr = indicators.get("atr_14", close * 0.015)
                atr_stop = close - 2.0 * atr
                ema_stop = min(e20, vwap) * 0.997
                stop = max(atr_stop, ema_stop)   # Whichever is tighter is safer
                targets = self._atr_targets(indicators, close, stop, "BUY")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Pullback R:R {rr:.2f} — wait for better entry")
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.74 if at_vwap else 0.70,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Bearish trend + pullback to EMA
            bearish_trend = e20 < e50
            if bearish_trend and (at_ema20 or at_vwap) and rsi > 45 and rsi < 65:
                signals.append("Pullback to EMA 20 in downtrend")
                atr = indicators.get("atr_14", close * 0.015)
                atr_stop = close + 2.0 * atr
                ema_stop = max(e20, vwap) * 1.003
                stop = min(atr_stop, ema_stop)
                targets = self._atr_targets(indicators, close, stop, "SELL")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Pullback R:R {rr:.2f} — wait for better entry")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.72,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════════════════

class MeanReversionStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Mean Reversion"
    @property
    def description(self) -> str:
        return "Fades extreme moves using Bollinger Bands + RSI oversold/overbought"
    @property
    def category(self) -> str: return "MEAN_REVERSION"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close    = indicators.get("close", df["close"].iloc[-1])
            bb_lower = indicators.get("bb_lower", close * 0.97)
            bb_upper = indicators.get("bb_upper", close * 1.03)
            bb_mid   = indicators.get("bb_mid", close)
            rsi      = indicators.get("rsi_14", 50)
            stoch_k  = indicators.get("stoch_k", 50)

            signals = []

            # Bullish MR: price below BB lower + RSI deeply oversold (< 25 for clean signal)
            if close <= bb_lower and rsi < 25 and stoch_k < 20:
                signals.append(f"Price below Bollinger lower band (₹{bb_lower:.0f})")
                signals.append(f"RSI deeply oversold at {rsi:.1f}")
                signals.append(f"StochRSI oversold at {stoch_k:.1f}")
                risk = (bb_mid - close)  # Distance to mean
                stop = close - risk * 0.5  # SL below current by half the distance to mean
                target = bb_mid
                t2 = bb_upper
                rr = self._calculate_rr(close, stop, target)
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Mean Rev R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "REVERSAL")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.76,
                    entry_price=close, stop_loss=stop,
                    targets=[target, t2, t2 + (t2 - target)],
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Bearish MR: price above BB upper + RSI deeply overbought (> 75 for clean signal)
            if close >= bb_upper and rsi > 75 and stoch_k > 80:
                signals.append(f"Price above Bollinger upper band (₹{bb_upper:.0f})")
                signals.append(f"RSI deeply overbought at {rsi:.1f}")
                risk = (close - bb_mid)
                stop = close + risk * 0.5
                target = bb_mid
                rr = self._calculate_rr(close, stop, target)
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Mean Rev R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "REVERSAL")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.74,
                    entry_price=close, stop_loss=stop,
                    targets=[target, bb_lower, bb_lower - (bb_mid - bb_lower)],
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OPENING RANGE BREAKOUT (ORB)
# ═══════════════════════════════════════════════════════════════════════════════

class ORBStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Opening Range Breakout"
    @property
    def description(self) -> str:
        return "Detects breakout above/below the first 15-minute candle range"
    @property
    def category(self) -> str: return "BREAKOUT"

    def analyze(self, df, indicators, ticker="", timeframe="5m") -> StrategySignal:
        try:
            if len(df) < 5:
                return self._no_trade(ticker, timeframe)

            close = df["close"].iloc[-1]
            atr = indicators.get("atr_14", close * 0.01)

            # Get first 3 candles (15 min) of today
            today = df.index[-1].date() if hasattr(df.index[-1], 'date') else None
            if today:
                today_df = df[pd.to_datetime(df.index).date == today] if today else df.tail(10)
            else:
                today_df = df.tail(10)

            if len(today_df) >= 3:
                orb_df = today_df.iloc[:3]
                orb_high = orb_df["high"].max()
                orb_low  = orb_df["low"].min()
            else:
                return self._no_trade(ticker, timeframe, "Insufficient intraday data for ORB")

            vol_ratio = indicators.get("vol_ratio", 1.0)
            signals = [f"ORB range: ₹{orb_low:.0f} - ₹{orb_high:.0f}"]

            if close > orb_high and vol_ratio >= 1.3:
                signals.append(f"Breakout above ORB high (₹{orb_high:.0f})")
                stop = orb_high - atr * 0.5
                targets = self._atr_targets(indicators, close, stop, "BUY", [1.5, 2.5, 3.5])
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.74,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )

            if close < orb_low and vol_ratio >= 1.3:
                signals.append(f"Breakdown below ORB low (₹{orb_low:.0f})")
                stop = orb_low + atr * 0.5
                targets = self._atr_targets(indicators, close, stop, "SELL", [1.5, 2.5, 3.5])
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.74,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VWAP REVERSAL
# ═══════════════════════════════════════════════════════════════════════════════

class VWAPReversalStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "VWAP Reversal"
    @property
    def description(self) -> str:
        return "Fades extreme deviation from VWAP with RSI confirmation"
    @property
    def category(self) -> str: return "MEAN_REVERSION"

    def analyze(self, df, indicators, ticker="", timeframe="5m") -> StrategySignal:
        try:
            close = indicators.get("close", df["close"].iloc[-1])
            vwap  = indicators.get("vwap", close)
            rsi   = indicators.get("rsi_14", 50)
            atr   = indicators.get("atr_14", close * 0.01)

            dev_pct = (close - vwap) / vwap * 100
            signals = [f"VWAP: ₹{vwap:.0f} | Deviation: {dev_pct:.2f}%"]

            # Bullish: VWAP far below + RSI deeply oversold (tightened from 1.5% to 2.5%)
            if dev_pct < -2.5 and rsi < 30:
                signals.append(f"Price {abs(dev_pct):.1f}% below VWAP — extreme deviation")
                signals.append(f"RSI deeply oversold: {rsi:.1f}")
                stop = close - 2.0 * atr
                target = vwap
                rr = self._calculate_rr(close, stop, target)
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"VWAP Rev R:R {rr:.2f} — price too close to VWAP")
                t2 = vwap + (vwap - close) * 0.5
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "REVERSAL")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.72,
                    entry_price=close, stop_loss=stop, targets=[target, t2, t2],
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Bearish: price far above VWAP + RSI deeply overbought (tightened from 1.5% to 2.5%)
            if dev_pct > 2.5 and rsi > 70:
                signals.append(f"Price {dev_pct:.1f}% above VWAP — extreme deviation")
                signals.append(f"RSI deeply overbought: {rsi:.1f}")
                stop = close + 2.0 * atr
                target = vwap
                rr = self._calculate_rr(close, stop, target)
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"VWAP Rev R:R {rr:.2f} — price too close to VWAP")
                t2 = vwap - (close - vwap) * 0.5
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "REVERSAL")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.72,
                    entry_price=close, stop_loss=stop, targets=[target, t2, t2],
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SUPERTREND + EMA
# ═══════════════════════════════════════════════════════════════════════════════

class SupertrendEMAStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "SuperTrend + EMA"
    @property
    def description(self) -> str:
        return "SuperTrend direction confirmed by EMA alignment and trend strength"
    @property
    def category(self) -> str: return "TREND"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close = indicators.get("close", df["close"].iloc[-1])
            st_dir = indicators.get("supertrend_dir", 0)
            e20    = indicators.get("ema_20", close)
            e50    = indicators.get("ema_50", close)
            adx    = indicators.get("adx", 20)

            signals = []

            # Bullish: SuperTrend = 1 (bullish) + EMA alignment + ADX > 20
            if st_dir == 1 and close > e20 > e50 and adx > 20:
                signals.append("SuperTrend bullish (price above SuperTrend line)")
                signals.append("EMA 20 > EMA 50 confirmation")
                if adx > 25: signals.append(f"ADX={adx:.1f} confirms strong trend")
                stop = self._atr_stop(indicators, close, "BUY", 2.0)
                targets = self._atr_targets(indicators, close, stop, "BUY")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"ST R:R {rr:.2f} below minimum")
                conf = 0.80 + (0.05 if adx > 35 else 0)
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=conf,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Bearish: SuperTrend bearish + EMA alignment + ADX > 20
            if st_dir == -1 and close < e20 < e50 and adx > 20:
                signals.append("SuperTrend bearish (price below SuperTrend line)")
                signals.append("EMA 20 < EMA 50 confirmation")
                if adx > 25: signals.append(f"ADX={adx:.1f} confirms strong downtrend")
                stop = self._atr_stop(indicators, close, "SELL", 2.0)
                targets = self._atr_targets(indicators, close, stop, "SELL")
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"ST R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "PULLBACK")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.80,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RSI DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

class RSIDivergenceStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "RSI Divergence"
    @property
    def description(self) -> str:
        return "Detects bullish/bearish divergence between price and RSI"
    @property
    def category(self) -> str: return "REVERSAL"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            if len(df) < 20:
                return self._no_trade(ticker, timeframe)

            close = df["close"].iloc[-1]
            from app.core.analysis.indicators import rsi as calc_rsi
            rsi_series = calc_rsi(df, 14)

            # Look at last 10 candles
            price_slice = df["close"].tail(10).values
            rsi_slice   = rsi_series.tail(10).values

            if len(price_slice) < 8 or np.isnan(rsi_slice).any():
                return self._no_trade(ticker, timeframe)

            price_low1  = price_slice[:5].min()
            price_low2  = price_slice[5:].min()
            rsi_low1    = rsi_slice[:5][price_slice[:5].argmin()]
            rsi_low2    = rsi_slice[5:][price_slice[5:].argmin()]

            price_high1 = price_slice[:5].max()
            price_high2 = price_slice[5:].max()
            rsi_high1   = rsi_slice[:5][price_slice[:5].argmax()]
            rsi_high2   = rsi_slice[5:][price_slice[5:].argmax()]

            signals = []

            # Bullish Divergence: price makes LL but RSI makes HL
            if price_low2 < price_low1 and rsi_low2 > rsi_low1 and rsi_low2 < 45:
                signals.append("Bullish RSI Divergence: lower price low, higher RSI low")
                signals.append(f"RSI at {rsi_slice[-1]:.1f} — oversold territory")
                stop = self._atr_stop(indicators, close, "BUY")
                targets = self._atr_targets(indicators, close, stop, "BUY")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.76,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )

            # Bearish Divergence: price makes HH but RSI makes LH
            if price_high2 > price_high1 and rsi_high2 < rsi_high1 and rsi_high2 > 55:
                signals.append("Bearish RSI Divergence: higher price high, lower RSI high")
                signals.append(f"RSI at {rsi_slice[-1]:.1f} — overbought territory")
                stop = self._atr_stop(indicators, close, "SELL")
                targets = self._atr_targets(indicators, close, stop, "SELL")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.75,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MACD MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

class MACDMomentumStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "MACD Momentum"
    @property
    def description(self) -> str:
        return "MACD crossover with histogram expansion and trend confirmation"
    @property
    def category(self) -> str: return "MOMENTUM"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close       = indicators.get("close", df["close"].iloc[-1])
            macd        = indicators.get("macd", 0)
            macd_sig    = indicators.get("macd_signal", 0)
            macd_hist   = indicators.get("macd_hist", 0)
            prev_hist   = indicators.get("macd_prev_hist", 0)
            e50         = indicators.get("ema_50", close)

            signals = []

            # Bullish: MACD crossed above signal + hist expanding + above EMA50
            if macd > macd_sig and macd_hist > prev_hist and macd_hist > 0:
                if close > e50:
                    signals.append("MACD bullish crossover confirmed")
                    signals.append("MACD histogram expanding positively")
                    signals.append("Price above EMA 50")
                    conf = 0.74
                    if macd > 0: signals.append("MACD above zero line"); conf = 0.80
                    stop = self._atr_stop(indicators, close, "BUY")
                    targets = self._atr_targets(indicators, close, stop, "BUY")
                    return StrategySignal(
                        strategy_name=self.name, direction="BUY",
                        confidence=conf,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    )

            # Bearish: MACD crossed below signal + hist shrinking
            if macd < macd_sig and macd_hist < prev_hist and macd_hist < 0:
                if close < e50:
                    signals.append("MACD bearish crossover confirmed")
                    signals.append("MACD histogram expanding negatively")
                    stop = self._atr_stop(indicators, close, "SELL")
                    targets = self._atr_targets(indicators, close, stop, "SELL")
                    return StrategySignal(
                        strategy_name=self.name, direction="SELL",
                        confidence=0.74,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SMART MONEY CONCEPTS (SMC)
# ═══════════════════════════════════════════════════════════════════════════════

class SMCStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Smart Money Concepts"
    @property
    def description(self) -> str:
        return "Order Block + FVG + BOS/CHoCH + Premium/Discount institutional analysis"
    @property
    def category(self) -> str: return "INSTITUTIONAL"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            from app.core.analysis.smart_money import analyze_smc
            smc = analyze_smc(df, ticker, timeframe)
            close = indicators.get("close", df["close"].iloc[-1])
            signals = []

            # Bullish SMC setup
            if (smc.bias == "BULLISH" and
                    smc.premium_discount == "DISCOUNT" and
                    smc.nearest_ob_bullish is not None):
                ob = smc.nearest_ob_bullish
                dist_pct = (close - ob.top) / ob.top * 100
                if abs(dist_pct) < 1.5:  # Price near the OB
                    signals.append(f"Price at Bullish Order Block (₹{ob.bottom:.0f}-₹{ob.top:.0f})")
                    signals.append(f"SMC Bias: BULLISH")
                    signals.append(f"Price in DISCOUNT zone")
                    if smc.last_bos == "BULLISH": signals.append("BOS: Bullish confirmed")
                    if smc.nearest_fvg_bullish: signals.append("Bullish FVG nearby as magnet")
                    stop = ob.bottom * 0.998
                    targets = self._atr_targets(indicators, close, stop, "BUY")
                    return StrategySignal(
                        strategy_name=self.name, direction="BUY",
                        confidence=0.82,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                        pattern_context=f"OB: {ob.bottom:.0f}-{ob.top:.0f}",
                    )

            # Bearish SMC setup
            if (smc.bias == "BEARISH" and
                    smc.premium_discount == "PREMIUM" and
                    smc.nearest_ob_bearish is not None):
                ob = smc.nearest_ob_bearish
                dist_pct = (ob.bottom - close) / ob.bottom * 100
                if abs(dist_pct) < 1.5:
                    signals.append(f"Price at Bearish Order Block (₹{ob.bottom:.0f}-₹{ob.top:.0f})")
                    signals.append(f"SMC Bias: BEARISH")
                    signals.append(f"Price in PREMIUM zone")
                    if smc.last_bos == "BEARISH": signals.append("BOS: Bearish confirmed")
                    stop = ob.top * 1.002
                    targets = self._atr_targets(indicators, close, stop, "SELL")
                    return StrategySignal(
                        strategy_name=self.name, direction="SELL",
                        confidence=0.82,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PRICE ACTION
# ═══════════════════════════════════════════════════════════════════════════════

class PriceActionStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Price Action"
    @property
    def description(self) -> str:
        return "Pure candlestick pattern analysis with volume confirmation"
    @property
    def category(self) -> str: return "REVERSAL"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            from app.core.analysis.patterns import detect_candlestick_patterns
            close = indicators.get("close", df["close"].iloc[-1])
            vol_ratio = indicators.get("vol_ratio", 1.0)
            patterns = detect_candlestick_patterns(df)

            bullish_pats = [p for p in patterns if p.type == "BULLISH"]
            bearish_pats = [p for p in patterns if p.type == "BEARISH"]

            if bullish_pats and vol_ratio >= 1.2:
                best = max(bullish_pats, key=lambda x: x.strength)
                signals = [f"Pattern: {best.name} ({best.description})"]
                if vol_ratio >= 1.5: signals.append(f"Volume {vol_ratio:.1f}x average — confirmed")
                stop = self._atr_stop(indicators, close, "BUY")
                targets = self._atr_targets(indicators, close, stop, "BUY")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=best.strength * 0.9,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    pattern_context=best.name,
                )

            if bearish_pats and vol_ratio >= 1.2:
                best = max(bearish_pats, key=lambda x: x.strength)
                signals = [f"Pattern: {best.name} ({best.description})"]
                stop = self._atr_stop(indicators, close, "SELL")
                targets = self._atr_targets(indicators, close, stop, "SELL")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=best.strength * 0.9,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    pattern_context=best.name,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SUPPORT / RESISTANCE BOUNCE
# ═══════════════════════════════════════════════════════════════════════════════

class SupportResistanceBounceStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Support/Resistance Bounce"
    @property
    def description(self) -> str:
        return "Identifies bounces off dynamic S/R levels with confirmation"
    @property
    def category(self) -> str: return "REVERSAL"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close      = indicators.get("close", df["close"].iloc[-1])
            support    = indicators.get("support", [])
            resistance = indicators.get("resistance", [])
            rsi        = indicators.get("rsi_14", 50)
            vol_ratio  = indicators.get("vol_ratio", 1.0)
            atr        = indicators.get("atr_14", close * 0.01)

            signals = []

            # Support bounce — require volume confirmation (≥ 1.3x)
            for s in support[:3]:
                if abs(close - s) / s < 0.008:
                    signals.append(f"Price at support level ₹{s:.0f}")
                    if rsi < 45: signals.append(f"RSI at {rsi:.1f}")
                    if vol_ratio >= 1.3:
                        signals.append(f"Volume confirmation: {vol_ratio:.1f}x")
                    else:
                        return self._no_trade(ticker, timeframe, f"Support level hit but volume too weak ({vol_ratio:.1f}x < 1.3x required)")
                    stop = s - atr * 1.5  # ATR stop below support
                    targets = self._atr_targets(indicators, close, stop, "BUY")
                    rr = self._calculate_rr(close, stop, targets[0])
                    if rr < 2.0: return self._no_trade(ticker, timeframe, f"S/R R:R {rr:.2f} below minimum")
                    z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "SUPPORT")
                    return StrategySignal(
                        strategy_name=self.name, direction="BUY",
                        confidence=0.72,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=rr,
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                        entry_zone_low=z_low, entry_zone_high=z_high,
                    )

            # Resistance rejection — require volume confirmation
            for r in resistance[:3]:
                if abs(close - r) / r < 0.008:
                    signals.append(f"Price at resistance level ₹{r:.0f}")
                    if rsi > 55: signals.append(f"RSI at {rsi:.1f}")
                    if vol_ratio >= 1.3:
                        signals.append(f"Volume confirmation: {vol_ratio:.1f}x")
                    else:
                        return self._no_trade(ticker, timeframe, f"Resistance hit but volume too weak ({vol_ratio:.1f}x < 1.3x required)")
                    stop = r + atr * 1.5
                    targets = self._atr_targets(indicators, close, stop, "SELL")
                    rr = self._calculate_rr(close, stop, targets[0])
                    if rr < 2.0: return self._no_trade(ticker, timeframe, f"S/R R:R {rr:.2f} below minimum")
                    z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "RESISTANCE")
                    return StrategySignal(
                        strategy_name=self.name, direction="SELL",
                        confidence=0.70,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=rr,
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                        entry_zone_low=z_low, entry_zone_high=z_high,
                    )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. GAP TRADING
# ═══════════════════════════════════════════════════════════════════════════════

class GapTradingStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Gap Trading"
    @property
    def description(self) -> str:
        return "Trades gap-up / gap-down openings with trend confirmation"
    @property
    def category(self) -> str: return "MOMENTUM"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            if len(df) < 2:
                return self._no_trade(ticker, timeframe)

            prev_close = df["close"].iloc[-2]
            today_open = df["open"].iloc[-1]
            close = df["close"].iloc[-1]
            gap_pct = (today_open - prev_close) / prev_close * 100
            atr = indicators.get("atr_14", close * 0.01)
            e20 = indicators.get("ema_20", close)

            signals = []

            # Gap Up + continuation (stricter: ≥ 1.5% gap, 2x volume)
            if gap_pct > 1.5 and close > today_open and close > e20 and vol_ratio >= 2.0:
                signals.append(f"Gap Up: +{gap_pct:.2f}% from previous close")
                signals.append("Price holding above gap open — continuation")
                signals.append(f"Volume {vol_ratio:.1f}x confirms institutional gap")
                stop = today_open - atr * 1.5
                targets = self._atr_targets(indicators, close, stop, "BUY", [2.0, 3.0, 5.0])
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Gap R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "BUY", indicators, "BREAKOUT")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=min(0.84, 0.68 + gap_pct * 0.025),
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )

            # Gap Down + continuation (stricter: ≥ 1.5% gap, 2x volume)
            if gap_pct < -1.5 and close < today_open and close < e20 and vol_ratio >= 2.0:
                signals.append(f"Gap Down: {gap_pct:.2f}% from previous close")
                signals.append("Price holding below gap open — continuation")
                signals.append(f"Volume {vol_ratio:.1f}x confirms institutional gap")
                stop = today_open + atr * 1.5
                targets = self._atr_targets(indicators, close, stop, "SELL", [2.0, 3.0, 5.0])
                rr = self._calculate_rr(close, stop, targets[0])
                if rr < 2.0: return self._no_trade(ticker, timeframe, f"Gap R:R {rr:.2f} below minimum")
                z_low, z_high = self._calculate_entry_zone(close, "SELL", indicators, "BREAKOUT")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=min(0.82, 0.68 + abs(gap_pct) * 0.025),
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=rr,
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    entry_zone_low=z_low, entry_zone_high=z_high,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. VOLUME BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════

class VolumeBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "Volume Breakout"
    @property
    def description(self) -> str:
        return "Detects unusual volume spikes with price breakout — institutional activity"
    @property
    def category(self) -> str: return "BREAKOUT"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close      = indicators.get("close", df["close"].iloc[-1])
            vol_ratio  = indicators.get("vol_ratio", 1.0)
            obv        = indicators.get("obv", 0)
            obv_prev   = indicators.get("obv_prev", 0)
            cmf        = indicators.get("cmf_20", 0)
            atr        = indicators.get("atr_14", close * 0.01)
            e20        = indicators.get("ema_20", close)

            signals = []

            if vol_ratio < 2.0:
                return self._no_trade(ticker, timeframe, f"Volume ratio {vol_ratio:.1f}x — below 2x threshold")

            signals.append(f"Unusual volume: {vol_ratio:.1f}x average")

            # OBV rising = buying pressure
            if obv > obv_prev and close > e20 and cmf > 0.05:
                signals.append("OBV rising — accumulation")
                signals.append(f"CMF={cmf:.3f} positive — buying pressure")
                signals.append("Price above EMA 20")
                stop = close - 2 * atr
                targets = self._atr_targets(indicators, close, stop, "BUY", [1.5, 3.0, 5.0])
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=min(0.85, 0.65 + vol_ratio * 0.04),
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )

            # OBV falling = distribution
            if obv < obv_prev and close < e20 and cmf < -0.05:
                signals.append("OBV falling — distribution")
                signals.append(f"CMF={cmf:.3f} negative — selling pressure")
                stop = close + 2 * atr
                targets = self._atr_targets(indicators, close, stop, "SELL", [1.5, 3.0, 5.0])
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=min(0.83, 0.65 + vol_ratio * 0.04),
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )
        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. BOLLINGER BAND SQUEEZE BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════

class BBSqueezeBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "BB Squeeze Breakout"
    @property
    def description(self) -> str:
        return "Detects Bollinger Band contraction (squeeze) and triggers on breakout expansion"
    @property
    def category(self) -> str: return "BREAKOUT"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close      = indicators.get("close", df["close"].iloc[-1])
            bb_upper   = indicators.get("bb_upper")
            bb_lower   = indicators.get("bb_lower")
            bb_width   = indicators.get("bb_width")
            vol_ratio  = indicators.get("vol_ratio", 1.0)
            atr        = indicators.get("atr_14", close * 0.01)

            if bb_upper is None or bb_lower is None or bb_width is None:
                return self._no_trade(ticker, timeframe, "Missing BB indicators")

            # Check rolling 20-period average of bb_width to determine narrow squeeze
            bb_width_cols = [c for c in df.columns if "bb_width" in c]
            if bb_width_cols:
                hist_widths = df[bb_width_cols[0]]
                avg_width = hist_widths.rolling(20).mean().iloc[-1]
            else:
                avg_width = 0.05  # Fallback narrow threshold

            # Squeeze verification (width is narrower than average width over past 20 bars)
            is_squeeze = bb_width < (avg_width * 0.95)

            if not is_squeeze:
                return self._no_trade(ticker, timeframe, f"Width {bb_width:.3f} above squeeze threshold {avg_width:.3f}")

            signals = [f"Bollinger Band Squeeze confirmed: width={bb_width:.3f}"]

            # Breakout crossover checks
            prev_close = df["close"].iloc[-2] if len(df) > 1 else close
            
            bb_upper_cols = [c for c in df.columns if "bb_upper" in c]
            bb_lower_cols = [c for c in df.columns if "bb_lower" in c]
            
            if bb_upper_cols and bb_lower_cols:
                prev_upper = df[bb_upper_cols[0]].iloc[-2] if len(df) > 1 else bb_upper
                prev_lower = df[bb_lower_cols[0]].iloc[-2] if len(df) > 1 else bb_lower
                
                is_bull_breakout = close > bb_upper and prev_close <= prev_upper
                is_bear_breakout = close < bb_lower and prev_close >= prev_lower
            else:
                is_bull_breakout = close > bb_upper
                is_bear_breakout = close < bb_lower

            if is_bull_breakout and vol_ratio > 1.2:
                signals.append("Price breakout above upper Bollinger Band")
                signals.append(f"Volume confirms breakout: {vol_ratio:.1f}x average")
                stop = close - 1.5 * atr
                targets = self._atr_targets(indicators, close, stop, "BUY")
                return StrategySignal(
                    strategy_name=self.name, direction="BUY",
                    confidence=0.82,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )

            if is_bear_breakout and vol_ratio > 1.2:
                signals.append("Price breakdown below lower Bollinger Band")
                signals.append(f"Volume confirms breakdown: {vol_ratio:.1f}x average")
                stop = close + 1.5 * atr
                targets = self._atr_targets(indicators, close, stop, "SELL")
                return StrategySignal(
                    strategy_name=self.name, direction="SELL",
                    confidence=0.82,
                    entry_price=close, stop_loss=stop, targets=targets,
                    risk_reward=self._calculate_rr(close, stop, targets[0]),
                    contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                )

        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. DOUBLE EMA CROSSOVER MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleEMACrossoverStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "EMA Crossover"
    @property
    def description(self) -> str:
        return "9 EMA and 20 EMA crossover with MACD & RSI trend verification filters"
    @property
    def category(self) -> str: return "TREND"

    def analyze(self, df, indicators, ticker="", timeframe="1d") -> StrategySignal:
        try:
            close      = indicators.get("close", df["close"].iloc[-1])
            rsi        = indicators.get("rsi_14", 50)
            macd_hist  = indicators.get("macd_hist", 0)
            atr        = indicators.get("atr_14", close * 0.01)

            # Compute EMAs directly to avoid column mismatch
            close_series = df["close"]
            ema9 = close_series.ewm(span=9, adjust=False).mean()
            ema20 = close_series.ewm(span=20, adjust=False).mean()

            e9 = ema9.iloc[-1]
            e20 = ema20.iloc[-1]

            prev_e9 = ema9.iloc[-2] if len(df) > 1 else e9
            prev_e20 = ema20.iloc[-2] if len(df) > 1 else e20

            is_bull_cross = e9 > e20 and prev_e9 <= prev_e20
            is_bear_cross = e9 < e20 and prev_e9 >= prev_e20

            signals = []

            if is_bull_cross:
                # Filter: RSI between 45 and 68, MACD histogram is positive
                if rsi > 45 and rsi < 68 and macd_hist > 0:
                    signals.append("Bullish Crossover: 9 EMA crossed above 20 EMA")
                    signals.append(f"RSI={rsi:.1f} shows strong rising momentum")
                    signals.append(f"MACD Hist={macd_hist:.4f} is positive")
                    stop = close - 1.8 * atr
                    targets = self._atr_targets(indicators, close, stop, "BUY")
                    return StrategySignal(
                        strategy_name=self.name, direction="BUY",
                        confidence=0.84,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    )

            if is_bear_cross:
                # Filter: RSI between 32 and 55, MACD histogram is negative
                if rsi < 55 and rsi > 32 and macd_hist < 0:
                    signals.append("Bearish Crossover: 9 EMA crossed below 20 EMA")
                    signals.append(f"RSI={rsi:.1f} shows strong falling momentum")
                    signals.append(f"MACD Hist={macd_hist:.4f} is negative")
                    stop = close + 1.8 * atr
                    targets = self._atr_targets(indicators, close, stop, "SELL")
                    return StrategySignal(
                        strategy_name=self.name, direction="SELL",
                        confidence=0.84,
                        entry_price=close, stop_loss=stop, targets=targets,
                        risk_reward=self._calculate_rr(close, stop, targets[0]),
                        contributing_signals=signals, ticker=ticker, timeframe=timeframe,
                    )

        except Exception as e:
            logger.warning("{} error: {}", self.name, e)
        return self._no_trade(ticker, timeframe)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

ALL_STRATEGIES = [
    TrendFollowingStrategy,
    BreakoutStrategy,
    PullbackStrategy,
    MeanReversionStrategy,
    ORBStrategy,
    VWAPReversalStrategy,
    SupertrendEMAStrategy,
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    SMCStrategy,
    PriceActionStrategy,
    SupportResistanceBounceStrategy,
    GapTradingStrategy,
    VolumeBreakoutStrategy,
    BBSqueezeBreakoutStrategy,
    DoubleEMACrossoverStrategy,
]

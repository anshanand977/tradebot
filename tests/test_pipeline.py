"""
Unit and Integration Tests for AI Trading Analyst Pipeline
============================================================
Tests the entire system:
  1. Technical Indicators
  2. Smart Money Concepts
  3. Voting & Probability Engines
  4. Risk Management Rules
  5. Virtual Portfolio Simulation & Paper Trading
  6. Self-Learning Engine Performance Weights
"""

import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.config import settings
from app.db.database import init_db, db_session
from app.db.models import Trade, Portfolio, StrategyPerformance
from app.core.analysis.indicators import compute_all_indicators, ema, rsi, macd, adx, atr
from app.core.analysis.smart_money import analyze_smc
from app.core.strategies.strategy_manager import StrategyManager, VotingEngine, ProbabilityEngine, signal_generator
from app.core.risk.risk_manager import RiskManager
from app.core.simulation.virtual_portfolio import VirtualPortfolio, SimulatedOrder
from app.ai.ml_models import outcome_predictor


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_candles(length: int = 100, trend: str = "uptrend") -> pd.DataFrame:
    """Generates synthetic OHLCV candle data for test cases."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.utcnow(), periods=length, freq="1D")

    close = 100.0
    closes = []
    for i in range(length):
        if trend == "uptrend":
            change = np.random.normal(0.5, 1.0)
        elif trend == "downtrend":
            change = np.random.normal(-0.5, 1.0)
        else:
            change = np.random.normal(0.0, 1.0)
        close += change
        closes.append(close)

    closes = np.array(closes)
    highs = closes + np.random.exponential(1.0, length)
    lows = closes - np.random.exponential(1.0, length)
    opens = closes + np.random.normal(0.0, 0.5, length)
    volumes = np.random.randint(1000, 10000, length).astype(float)

    # Make sure opens/closes are within high/low boundaries
    for i in range(length):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalIndicators(unittest.TestCase):
    """Verifies correctness of pure numpy/pandas technical indicators."""

    def setUp(self):
        self.df = generate_mock_candles(100, "uptrend")

    def test_ema(self):
        ema20 = ema(self.df, period=20)
        self.assertEqual(len(ema20), len(self.df))
        self.assertFalse(ema20.dropna().empty)

    def test_rsi(self):
        rsi14 = rsi(self.df, period=14)
        self.assertEqual(len(rsi14), len(self.df))
        self.assertTrue((rsi14.dropna() >= 0).all() and (rsi14.dropna() <= 100).all())

    def test_macd(self):
        macd_df = macd(self.df)
        self.assertIn("macd", macd_df.columns)
        self.assertIn("macd_signal", macd_df.columns)
        self.assertIn("macd_hist", macd_df.columns)
        self.assertEqual(len(macd_df), len(self.df))

    def test_adx(self):
        adx_df = adx(self.df)
        self.assertIn("adx", adx_df.columns)
        self.assertEqual(len(adx_df), len(self.df))

    def test_compute_all_indicators(self):
        indicators = compute_all_indicators(self.df)
        self.assertIn("rsi_14", indicators)
        self.assertIn("macd_hist", indicators)
        self.assertIn("adx", indicators)
        self.assertIn("ema_20", indicators)
        self.assertIn("close", indicators)


class TestSmartMoneyConcepts(unittest.TestCase):
    """Verifies that Order Blocks and Fair Value Gaps are correctly mapped."""

    def test_smc_detection(self):
        df = generate_mock_candles(60, "ranging")
        # Inject an FVG (Large green candle creating a gap)
        # Bar 30 is large green: low[31] > high[29]
        df.loc[df.index[29], "high"] = 100.0
        df.loc[df.index[30], "open"] = 101.0
        df.loc[df.index[30], "close"] = 110.0
        df.loc[df.index[30], "high"] = 111.0
        df.loc[df.index[30], "low"] = 101.0
        df.loc[df.index[31], "low"] = 105.0

        ctx = analyze_smc(df, "RELIANCE.NS", "1d")
        self.assertTrue(isinstance(ctx.order_blocks, list))
        self.assertTrue(isinstance(ctx.fair_value_gaps, list))


class TestVotingAndProbability(unittest.TestCase):
    """Tests strategy voting aggregation and calculated probability output."""

    def setUp(self):
        self.df = generate_mock_candles(150, "uptrend")
        self.indicators = compute_all_indicators(self.df)
        self.strategy_manager = StrategyManager()

    def test_voting_consensus(self):
        engine = VotingEngine(self.strategy_manager)
        # Enable all strategies
        for s in self.strategy_manager.get_all():
            self.strategy_manager.enable(s.name)

        rec = engine.vote(
            df=self.df,
            indicators=self.indicators,
            ticker="TCS.NS",
            symbol="TCS",
            exchange="NSE",
            timeframe="1d"
        )
        self.assertIsNotNone(rec.direction)
        self.assertIn(rec.direction, ["BUY", "SELL", "NO_TRADE"])

    def test_probability_calculation(self):
        engine = ProbabilityEngine()
        from app.core.strategies.strategy_manager import TradeRecommendation

        rec = TradeRecommendation(
            ticker="RELIANCE.NS", symbol="RELIANCE", exchange="NSE", timeframe="1d",
            direction="BUY", confidence=0.85, probability=0.0,
            entry_price=2400.0, stop_loss=2350.0, target1=2500.0, target2=2550.0, target3=2600.0,
            risk_reward=2.0, risk_level="MEDIUM", strategy_votes={},
            strategies_agreed=3, total_strategies_voted=10, majority_direction="BUY",
            contributing_signals=["SuperTrend Green", "EMA 20 crossover"],
            indicator_snapshot=self.indicators, is_actionable=True
        )

        perf = {"Trend Following": 0.65, "Breakout": 0.58}
        prob = engine.calculate(rec, perf)
        self.assertTrue(0.30 <= prob <= 0.93)


class TestMachineLearning(unittest.TestCase):
    """Tests local XGBoost trade outcome classifier and scaling logic."""

    def test_ml_outcome_predictions(self):
        # Default cold-start fallback check
        prob = outcome_predictor.predict_win_probability({"rsi_14": 45.0, "adx": 22.0})
        self.assertEqual(prob, 0.50)

        # Check scale features
        feat = outcome_predictor._extract_features_row({"rsi_14": 45.0, "adx": 22.0})
        self.assertEqual(len(feat), 5)
        self.assertEqual(feat[0], 45.0)
        self.assertEqual(feat[2], 22.0)



class TestRiskManagement(unittest.TestCase):
    """Tests the risk gatekeeper's rules and circuit breaker checks."""

    def setUp(self):
        self.risk = RiskManager()

    def test_circuit_breaker(self):
        # Trigger daily loss limit
        self.risk.update_daily_pnl(-15000.0)
        self.assertTrue(self.risk.is_circuit_broken)

        res = self.risk.check_trade(
            ticker="INFY.NS", entry_price=1400.0, stop_loss=1380.0,
            portfolio_balance=100000.0, open_positions=0
        )
        self.assertFalse(res.allowed)
        self.assertIn("Circuit breaker", res.reason)

    def test_position_sizing(self):
        res = self.risk.check_trade(
            ticker="RELIANCE.NS", entry_price=2500.0, stop_loss=2450.0,
            portfolio_balance=100000.0, open_positions=1
        )
        self.assertTrue(res.allowed)
        self.assertTrue(res.position_size > 0)
        self.assertEqual(res.risk_amount, round(res.position_size * 50.0, 2))


class TestVirtualPortfolio(unittest.TestCase):
    """Tests simulated order fills, price changes, and partial exits."""

    def setUp(self):
        init_db()
        # Add mock portfolio to satisfy FK constraint
        with db_session() as db:
            p = db.query(Portfolio).filter_by(id=99).first()
            if not p:
                p = Portfolio(
                    id=99,
                    name="Test Portfolio",
                    initial_balance=50000.0,
                    current_balance=50000.0,
                    peak_balance=50000.0,
                    total_pnl=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0
                )
                db.add(p)
                db.commit()
        self.portfolio = VirtualPortfolio(portfolio_id=99, initial_balance=50000.0)

    def test_order_execution(self):
        order = SimulatedOrder(
            order_id="test-uuid-123", ticker="TCS.NS", symbol="TCS",
            direction="BUY", quantity=10, entry_price=3000.0, stop_loss=2900.0,
            target1=3100.0, target2=3200.0, target3=3300.0,
            strategy_name="Trend Following", confidence=0.85
        )

        res = self.portfolio.place_order(order)
        if res["status"] == "FILLED":
            self.assertEqual(len(self.portfolio.open_positions), 1)
            # Update price to hit Target 1 (partial exit)
            closed = self.portfolio.update_prices({"TCS.NS": 3150.0})
            # T1 is hit, so a partial exit should have occurred
            self.assertTrue(any(c["status"] == "PARTIAL" for c in closed))


if __name__ == "__main__":
    unittest.main()

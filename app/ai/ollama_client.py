"""
Ollama Local AI Client
========================
Connects to a locally running Ollama instance for AI-powered market analysis.
Gracefully falls back if Ollama is not available.
Supports: Llama 3, Mistral, Qwen, Phi, DeepSeek, Gemma
"""

import json
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger

from app.config import settings


class OllamaClient:
    """
    Interfaces with the local Ollama API (http://localhost:11434).
    All inference is local — zero cloud dependency.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_model = settings.OLLAMA_DEFAULT_MODEL
        self._available = None
        self._models: List[str] = []

    # ─── Connection ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama is running (cached for 15s to avoid blocking)."""
        import time
        now = time.time()
        if hasattr(self, "_last_checked") and now - self._last_checked < 15:
            return self._available

        self._last_checked = now
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=0.5)
            self._available = resp.status_code == 200
            if self._available:
                data = resp.json()
                self._models = [m["name"] for m in data.get("models", [])]
        except Exception:
            self._available = False
            self._models = []
        return self._available

    def get_available_models(self) -> List[str]:
        """Returns list of models installed in Ollama."""
        if self.is_available():
            return self._models
        return []

    def get_best_model(self) -> Optional[str]:
        """
        Auto-select the best available model.
        Preference order: Llama3 > Mistral > Qwen > Phi > Gemma > DeepSeek > any
        """
        models = self.get_available_models()
        if not models:
            return None

        preference = ["llama3", "mistral", "qwen", "phi", "gemma", "deepseek"]
        models_lower = [m.lower() for m in models]

        for pref in preference:
            for i, m in enumerate(models_lower):
                if pref in m:
                    return models[i]

        return models[0]  # Fallback to any available model

    # ─── Chat / Analysis ──────────────────────────────────────────────────────

    def analyze_market(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        recommendation_summary: str,
        context: str = "",
        model: Optional[str] = None,
    ) -> str:
        """
        Ask the local LLM to analyze a stock based on indicator data.
        Returns a human-readable analysis string.
        Falls back to rule-based analysis if Ollama is unavailable.
        """
        if not self.is_available():
            return self._rule_based_analysis(ticker, indicators, recommendation_summary)

        model = model or self.get_best_model() or self.default_model

        prompt = self._build_analysis_prompt(ticker, indicators, recommendation_summary, context)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 500,
                    },
                },
                timeout=settings.OLLAMA_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json().get("response", "Analysis unavailable")
            else:
                logger.warning("Ollama returned {}", response.status_code)
                return self._rule_based_analysis(ticker, indicators, recommendation_summary)
        except Exception as e:
            logger.warning("Ollama error: {}", e)
            return self._rule_based_analysis(ticker, indicators, recommendation_summary)

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        model: Optional[str] = None,
        context: str = "",
    ) -> str:
        """
        General-purpose chat with the local LLM.
        Includes a system prompt focused on Indian market trading.
        """
        if not self.is_available():
            return ("⚠️ Ollama is not running. Please install Ollama and pull a model. "
                    "Run: `ollama pull mistral` in your terminal.")

        model = model or self.get_best_model() or self.default_model

        system = (
            "You are an expert quantitative trader and financial analyst specializing "
            "in the Indian stock market (NSE, BSE). You have deep knowledge of "
            "technical analysis, Smart Money Concepts, F&O, SEBI regulations, and "
            "Indian market microstructure. Always be specific, cite indicator values, "
            "and never fabricate data. If you don't know, say so. "
            "Current date: " + datetime.now().strftime("%Y-%m-%d") + ". "
            + context
        )

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.4, "num_predict": 800},
                },
                timeout=settings.OLLAMA_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "No response")
            return "LLM response error. Check Ollama logs."
        except Exception as e:
            logger.error("Ollama chat error: {}", e)
            return f"Ollama connection failed: {e}"

    # ─── Prompt Builder ───────────────────────────────────────────────────────

    def _build_analysis_prompt(
        self, ticker, indicators, recommendation_summary, context
    ) -> str:
        ind = indicators
        supertrend_label = "BULLISH" if ind.get('supertrend_dir', 0) == 1 else "BEARISH"
        context_section = ("=== ADDITIONAL CONTEXT ===\n" + context) if context else ""
        return f"""
You are an expert Indian stock market analyst. Analyze the following data for {ticker}:

=== INDICATOR SNAPSHOT ===
Price: Rs.{ind.get('close', 0):.2f}
RSI (14): {ind.get('rsi_14', 0):.1f}
MACD: {ind.get('macd', 0):.4f} | Signal: {ind.get('macd_signal', 0):.4f} | Hist: {ind.get('macd_hist', 0):.4f}
ADX: {ind.get('adx', 0):.1f} | +DI: {ind.get('plus_di', 0):.1f} | -DI: {ind.get('minus_di', 0):.1f}
EMA 20: {ind.get('ema_20', 0):.2f} | EMA 50: {ind.get('ema_50', 0):.2f} | EMA 200: {ind.get('ema_200', 0):.2f}
VWAP: {ind.get('vwap', 0):.2f}
Volume ratio: {ind.get('vol_ratio', 1.0):.2f}x avg
SuperTrend Dir: {supertrend_label}
BB %: {ind.get('bb_pct', 0.5):.2f}
Stoch K: {ind.get('stoch_k', 50):.1f} | D: {ind.get('stoch_d', 50):.1f}

=== AI RECOMMENDATION ===
{recommendation_summary}

{context_section}

Please provide:
1. A clear, concise view of the current technical setup (2-3 sentences)
2. Key risk factors to watch
3. What would invalidate this trade setup
4. One actionable insight specific to Indian market conditions

Keep the response under 400 words. Be precise, not generic.
"""

    # ─── Rule-Based Fallback ──────────────────────────────────────────────────

    def _rule_based_analysis(
        self, ticker: str, indicators: Dict, recommendation_summary: str
    ) -> str:
        """
        Generates a structured analysis without the LLM.
        Used when Ollama is not available.
        """
        rsi = indicators.get("rsi_14", 50)
        adx = indicators.get("adx", 20)
        close = indicators.get("close", 0)
        ema20 = indicators.get("ema_20", close)
        above_vwap = indicators.get("above_vwap", False)
        vol_ratio = indicators.get("vol_ratio", 1.0)

        rsi_analysis = (
            "RSI is in oversold territory — potential for bounce." if rsi < 30 else
            "RSI is in overbought territory — watch for exhaustion." if rsi > 70 else
            f"RSI at {rsi:.1f} — neutral momentum zone."
        )

        trend_analysis = (
            "Strong trending market detected." if adx > 30 else
            "Weak trend or ranging market." if adx < 20 else
            "Moderate trend strength."
        )

        price_position = (
            f"Price is above EMA 20 (₹{ema20:.2f}) — bullish structure."
            if close > ema20 else
            f"Price is below EMA 20 (₹{ema20:.2f}) — bearish structure."
        )

        volume_note = (
            f"Volume is {vol_ratio:.1f}x average — institutional participation likely."
            if vol_ratio > 2 else
            "Volume is near average."
        )

        return (
            f"📊 **Rule-Based Analysis for {ticker}** (Ollama offline)\n\n"
            f"**Recommendation**: {recommendation_summary}\n\n"
            f"**RSI**: {rsi_analysis}\n"
            f"**Trend**: {trend_analysis}\n"
            f"**Price Structure**: {price_position}\n"
            f"**Volume**: {volume_note}\n"
            f"**VWAP**: {'Price above VWAP — intraday bias bullish.' if above_vwap else 'Price below VWAP — intraday bias bearish.'}\n\n"
            f"_Install Ollama and run `ollama pull mistral` for AI-powered analysis._"
        )

    def get_status(self) -> Dict:
        available = self.is_available()
        return {
            "available": available,
            "base_url": self.base_url,
            "models": self._models if available else [],
            "best_model": self.get_best_model() if available else None,
            "default_model": self.default_model,
        }


ollama_client = OllamaClient()

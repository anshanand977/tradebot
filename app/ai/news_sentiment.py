"""
AI News & Sentiment Analyst
=============================
Fetches market/ticker news and analyzes trends/sentiment using local Ollama.
Includes a dictionary-based fallback if Ollama is unavailable.
"""

import json
from typing import Dict, List, Any, Optional
import yfinance as yf
from loguru import logger

from app.config import settings
from app.ai.ollama_client import ollama_client


class NewsSentimentAnalyst:
    """
    Downloads news articles for symbols or general markets,
    and analyzes overall sentiment and trend drivers.
    """

    def fetch_ticker_news(self, ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches latest news articles for a given ticker from Yahoo Finance."""
        try:
            # Resolve symbol names for yfinance
            if ticker.startswith("^"):
                resolved = ticker
            elif "." in ticker:
                resolved = ticker
            else:
                resolved = f"{ticker}.NS"
            logger.info("📰 Fetching news for ticker: {}", resolved)
            t = yf.Ticker(resolved)
            raw_news = t.news or []
            
            normalized = []
            for item in raw_news[:limit]:
                # yfinance returns can have a 'content' nested dict or flat fields
                content = item.get("content", item)
                title = content.get("title", "")
                summary = content.get("summary", "")
                publisher = content.get("provider", {}).get("displayName", content.get("publisher", "Unknown"))
                pub_date = content.get("pubDate", content.get("displayTime", ""))
                link = content.get("canonicalUrl", {}).get("url", content.get("link", ""))

                if title:
                    normalized.append({
                        "title": title,
                        "summary": summary,
                        "publisher": publisher,
                        "date": pub_date,
                        "link": link
                    })
            return normalized
        except Exception as e:
            logger.error("Failed to fetch news for {}: {}", ticker, e)
            return []

    def fetch_market_trends_news(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetches general market news trends by looking at NIFTY 50 and SENSEX news."""
        news = []
        # Attempt Nifty 50 and Sensex tickers
        for ticker in ["^NSEI", "^BSESN", "RELIANCE.NS"]:
            fetched = self.fetch_ticker_news(ticker, limit=limit // 2)
            for item in fetched:
                # Avoid duplicates
                if not any(n["title"] == item["title"] for n in news):
                    news.append(item)
            if len(news) >= limit:
                break
        return news[:limit]

    def analyze_sentiment(self, news_items: List[Dict[str, Any]], ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes sentiment of news items.
        Returns a dict: {sentiment, score, confidence, trends, summary}
        """
        if not news_items:
            return {
                "sentiment": "NEUTRAL",
                "score": 0.0,
                "confidence": 0.5,
                "trends": ["No recent news available to analyze."],
                "summary": "No news articles found for analysis."
            }

        # Build list of headlines and summaries
        text_corpus = ""
        for i, item in enumerate(news_items, 1):
            text_corpus += f"{i}. Title: {item['title']}\n"
            if item['summary']:
                text_corpus += f"   Summary: {item['summary']}\n"
            text_corpus += "\n"

        target_subject = ticker if ticker else "General Market Trends"

        if ollama_client.is_available():
            model = ollama_client.get_best_model() or settings.OLLAMA_DEFAULT_MODEL
            prompt = f"""
You are an expert financial analyst. Analyze the following news articles for {target_subject} and evaluate the market sentiment.

=== NEWS ARTICLES ===
{text_corpus}

Analyze the tone, implications, and consensus of these articles.
Provide your response in raw JSON format matching this EXACT schema:
{{
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "score": float, // Float from -1.0 (extremely bearish) to +1.0 (extremely bullish)
  "confidence": float, // Float from 0.0 to 1.0 representing analysis confidence
  "trends": [string, string, string], // 3-4 key market drivers or news trends identified
  "summary": string // 1-2 sentence overall summary of the sentiment
}}

Do NOT include any introduction, conversational filler, markdown formatting (like ```json), or trailing notes. Output ONLY valid, parsable JSON.
"""
            try:
                # Call Ollama
                response_text = ollama_client.chat(message=prompt, model=model).strip()
                # Try parsing JSON
                cleaned_text = self._clean_json_response(response_text)
                result = json.loads(cleaned_text)
                
                # Validation
                sentiment = result.get("sentiment", "NEUTRAL").upper()
                if sentiment not in ["BULLISH", "BEARISH", "NEUTRAL"]:
                    sentiment = "NEUTRAL"
                    
                return {
                    "sentiment": sentiment,
                    "score": round(float(result.get("score", 0.0)), 2),
                    "confidence": round(float(result.get("confidence", 0.5)), 2),
                    "trends": result.get("trends", ["Unable to parse trends."])[:4],
                    "summary": result.get("summary", "Analysis completed successfully.")
                }
            except Exception as e:
                logger.warning("Ollama news analysis parse failed, falling back to rule-based: {}", e)

        # Fallback to rule-based keyword sentiment analysis
        return self._rule_based_sentiment(news_items, target_subject)

    def _clean_json_response(self, text: str) -> str:
        """Removes code blocks or conversational text to isolate JSON content."""
        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text

    def _rule_based_sentiment(self, news_items: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        """Dictionary-based keyword analyzer for offline fallback."""
        bullish_words = [
            "profit", "gain", "rise", "grow", "positive", "partnership", "up",
            "data center", "buy", "bull", "rally", "outperform", "dividend", 
            "record high", "beat", "strong", "advance", "expand", "merger", "deal"
        ]
        bearish_words = [
            "loss", "fall", "drop", "decline", "negative", "investigation", "down",
            "sell", "bear", "plunge", "underperform", "fine", "penalty", "debt", 
            "crash", "weak", "skid", "miss", "alert", "concern", "dispute", "lawsuit"
        ]

        score_sum = 0.0
        trends = []
        
        # Analyze titles and summaries
        for item in news_items:
            title_lower = item["title"].lower()
            summary_lower = item["summary"].lower() if item["summary"] else ""
            combined = title_lower + " " + summary_lower
            
            item_score = 0.0
            for w in bullish_words:
                if w in combined:
                    item_score += 0.25
            for w in bearish_words:
                if w in combined:
                    item_score -= 0.25
            
            # Clamp individual article score
            item_score = max(-1.0, min(1.0, item_score))
            score_sum += item_score
            
            # Simple trend identifier
            if item_score > 0.3:
                trends.append(f"Positive momentum: {item['title'][:60]}...")
            elif item_score < -0.3:
                trends.append(f"Risk alert: {item['title'][:60]}...")

        # Average score
        avg_score = score_sum / len(news_items)
        # Clamp average
        avg_score = max(-1.0, min(1.0, avg_score))

        if avg_score > 0.15:
            sentiment = "BULLISH"
        elif avg_score < -0.15:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        # Unique trends or fallback
        if not trends:
            trends = [item["title"] for item in news_items[:3]]
        else:
            # Deduplicate
            trends = list(set(trends))[:3]

        return {
            "sentiment": sentiment,
            "score": round(avg_score, 2),
            "confidence": 0.65,
            "trends": trends,
            "summary": f"Keyword-based analysis of {len(news_items)} news articles indicates a {sentiment.lower()} outlook for {target}."
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
news_sentiment_analyst = NewsSentimentAnalyst()

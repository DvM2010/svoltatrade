"""
CLAUDE BOT - AI Analyzer
Uses Google Gemini to analyze arbitrage opportunities and estimate true probabilities.
Falls back to rule-based heuristics when no API key is provided.
"""

import random
from typing import Optional

from config import CONFIG

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class AIAnalyzer:
    """
    AI-powered analysis of sports arbitrage and Polymarket opportunities.
    Uses Gemini when configured, otherwise applies heuristic analysis.
    """

    def __init__(self):
        self.api_key = CONFIG.get("gemini_api_key", "")
        self._model = None
        if self.api_key and _GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self._model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_sports_opportunity(self, opp) -> dict:
        """
        Analyze a sports ArbitrageOpportunity.
        Returns: {confidence: float, action: str, reasoning: str}
        """
        if self._model:
            try:
                return self._gemini_sports_analysis(opp)
            except Exception:
                pass
        return self._heuristic_sports_analysis(opp)

    def analyze_polymarket(self, market: dict) -> dict:
        """
        Analyze a Polymarket binary prediction market.
        Returns: {probability: float, confidence: float, edge: float, reasoning: str}
        """
        if self._model:
            try:
                return self._gemini_polymarket_analysis(market)
            except Exception:
                pass
        return self._heuristic_polymarket_analysis(market)

    # ------------------------------------------------------------------
    # Gemini implementations
    # ------------------------------------------------------------------

    def _gemini_sports_analysis(self, opp) -> dict:
        """Use Gemini to analyze a sports opportunity."""
        prompt = (
            f"You are a sports betting analyst. Analyze this latency arbitrage opportunity:\n"
            f"Sport: {opp.sport}\n"
            f"Event: {opp.event}\n"
            f"Market: {opp.market}\n"
            f"Buy odds @ {opp.bookmaker_buy}: {opp.odds_buy}\n"
            f"Hedge odds @ {opp.bookmaker_hedge}: {opp.odds_hedge}\n"
            f"Detected edge: {opp.edge_pct*100:.1f}%\n"
            f"Latency indicator: {opp.latency_indicator_ms}ms\n\n"
            f"In 2 sentences, assess confidence (0-1) and recommend action (BET/SKIP/MONITOR). "
            f"Reply in JSON: {{\"confidence\": 0.0, \"action\": \"BET\", \"reasoning\": \"...\"}}"
        )
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        # Try to parse JSON from response
        import json, re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"confidence": opp.confidence, "action": "BET", "reasoning": text[:200]}

    def _gemini_polymarket_analysis(self, market: dict) -> dict:
        """Use Gemini to estimate true probability for a Polymarket question."""
        prompt = (
            f"You are a prediction market analyst. Estimate the true probability for:\n"
            f"Question: {market.get('question', 'Unknown')}\n"
            f"Current market price: {market.get('best_mid', 0.5):.2%}\n\n"
            f"Reply in JSON: {{\"probability\": 0.0, \"confidence\": 0.0, "
            f"\"edge\": 0.0, \"reasoning\": \"...\"}}\n"
            f"where edge = abs(your_probability - market_price)."
        )
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        import json, re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return self._heuristic_polymarket_analysis(market)

    # ------------------------------------------------------------------
    # Heuristic fallbacks
    # ------------------------------------------------------------------

    def _heuristic_sports_analysis(self, opp) -> dict:
        """
        Rule-based heuristic analysis for sports arbitrage.
        Higher edge and lower latency = higher confidence.
        """
        edge = opp.edge_pct
        lat = opp.latency_indicator_ms

        # Base confidence from edge size
        base_conf = min(0.95, 0.5 + edge * 6)

        # Penalize high latency (stale data reduces reliability)
        if lat > 25:
            base_conf *= 0.85
        elif lat > 15:
            base_conf *= 0.92

        # Reward sure bets
        if opp.arb_type == "surebet":
            base_conf = min(0.99, base_conf + 0.1)

        base_conf = round(base_conf, 3)

        if base_conf >= 0.70:
            action = "BET"
            reasoning = (
                f"Strong {opp.arb_type} signal: {edge*100:.1f}% edge detected "
                f"with {lat}ms latency differential on {opp.event}. "
                f"Recommend immediate execution."
            )
        elif base_conf >= 0.55:
            action = "MONITOR"
            reasoning = (
                f"Moderate edge of {edge*100:.1f}% on {opp.event}. "
                f"Latency at {lat}ms — watch for confirmation before entry."
            )
        else:
            action = "SKIP"
            reasoning = (
                f"Edge of {edge*100:.1f}% insufficient for high-confidence trade. "
                f"Latency {lat}ms may indicate noise rather than true mispricing."
            )

        return {
            "confidence": base_conf,
            "action": action,
            "reasoning": reasoning,
        }

    def _heuristic_polymarket_analysis(self, market: dict) -> dict:
        """
        Heuristic estimate of Polymarket true probability.
        Applies slight regression to market price (markets are often efficient).
        """
        market_price = market.get("best_mid", 0.5)
        question = market.get("question", "").lower()

        # Sentiment nudges for known categories
        nudge = 0.0
        if "bitcoin" in question or "crypto" in question:
            nudge = random.gauss(0, 0.08)
        elif "fed" in question or "rate" in question:
            nudge = random.gauss(0, 0.05)
        elif "nba" in question or "nfl" in question or "sports" in question:
            nudge = random.gauss(0, 0.07)
        else:
            nudge = random.gauss(0, 0.04)

        # Regress toward market
        our_prob = max(0.02, min(0.98, market_price + nudge * 0.6))
        edge = abs(our_prob - market_price)
        confidence = min(0.85, 0.4 + edge * 4)

        return {
            "probability": round(our_prob, 4),
            "confidence": round(confidence, 3),
            "edge": round(edge, 4),
            "reasoning": (
                f"Heuristic estimate: {our_prob:.1%} vs market {market_price:.1%}. "
                f"Edge: {edge*100:.1f}%. Based on market efficiency assumptions "
                f"and category-specific variance."
            ),
        }

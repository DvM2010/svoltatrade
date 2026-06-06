"""
CLAUDE BOT - Polymarket Client
Interfaces with the Polymarket CLOB API for prediction market opportunities.
Falls back to mock data when no API credentials are provided.
"""

import random
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import CONFIG, POLYMARKET_CLOB_URL


# ---------------------------------------------------------------------------
# Mock Polymarket markets
# ---------------------------------------------------------------------------

_MOCK_MARKETS = [
    {
        "question": "Will the Fed cut rates in Q2 2025?",
        "category": "Finance",
        "base_prob": 0.62,
    },
    {
        "question": "Will Bitcoin hit $100K before July 2025?",
        "category": "Crypto",
        "base_prob": 0.38,
    },
    {
        "question": "Will Lakers win the 2025 NBA Championship?",
        "category": "Sports",
        "base_prob": 0.12,
    },
    {
        "question": "Will Chiefs win Super Bowl LIX?",
        "category": "Sports",
        "base_prob": 0.28,
    },
    {
        "question": "Will Elon Musk remain Twitter CEO through 2025?",
        "category": "Politics",
        "base_prob": 0.71,
    },
    {
        "question": "Will US enter recession in 2025?",
        "category": "Finance",
        "base_prob": 0.33,
    },
    {
        "question": "Will OpenAI release GPT-5 in 2025?",
        "category": "Tech",
        "base_prob": 0.55,
    },
    {
        "question": "Will Djokovic win 2025 French Open?",
        "category": "Sports",
        "base_prob": 0.44,
    },
    {
        "question": "Will ETH flip BTC by market cap in 2025?",
        "category": "Crypto",
        "base_prob": 0.08,
    },
    {
        "question": "Will there be a US government shutdown in 2025?",
        "category": "Politics",
        "base_prob": 0.41,
    },
]


class PolymarketOpportunity:
    def __init__(
        self,
        token_id: str,
        question: str,
        category: str,
        market_price: float,   # Current best bid/ask midpoint
        ai_probability: float, # Our estimated true probability
        edge: float,
        side: str,             # "YES" or "NO"
        recommended_usdc: float,
        expected_profit_usdc: float,
        timestamp: Optional[datetime] = None,
    ):
        self.token_id = token_id
        self.question = question
        self.category = category
        self.market_price = market_price
        self.ai_probability = ai_probability
        self.edge = edge
        self.side = side
        self.recommended_usdc = recommended_usdc
        self.expected_profit_usdc = expected_profit_usdc
        self.timestamp = timestamp or datetime.now(timezone.utc)


class PolymarketClient:
    """
    Fetches active Polymarket prediction markets and identifies edge
    opportunities via probability mismatch analysis.
    """

    def __init__(self):
        self.clob_url = POLYMARKET_CLOB_URL
        self.api_key = CONFIG.get("polymarket_api_key", "")
        self.private_key = CONFIG.get("private_key", "")
        self.min_edge = CONFIG.get("polymarket_min_edge", 0.05)
        self.max_position = CONFIG.get("polymarket_max_position_usdc", 50)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_active_markets(self, limit: int = 100) -> list:
        """
        Fetch active Polymarket markets.
        Returns list of market dicts.
        """
        if self.api_key:
            try:
                return self._fetch_real_markets(limit)
            except Exception:
                pass
        return self._generate_mock_markets(limit)

    def get_order_book(self, token_id: str) -> dict:
        """
        Fetch the CLOB order book for a specific market token.
        Returns dict with bids/asks or mock data.
        """
        if self.api_key:
            try:
                return self._fetch_real_book(token_id)
            except Exception:
                pass
        return self._generate_mock_book(token_id)

    def find_opportunities(self, ai_probabilities: Optional[dict] = None) -> list:
        """
        Scan markets and return PolymarketOpportunity objects where
        our estimated probability differs from market price by >= min_edge.
        """
        markets = self.get_active_markets(50)
        opportunities = []

        for market in markets:
            token_id = market.get("token_id", "")
            question = market.get("question", "")
            category = market.get("category", "Other")
            market_price = market.get("best_mid", 0.5)

            # Use AI probability if provided, else slight nudge from mock
            if ai_probabilities and token_id in ai_probabilities:
                ai_prob = ai_probabilities[token_id]
            else:
                # Simulate AI estimate: market_price ± random offset
                noise = random.gauss(0, 0.06)
                ai_prob = max(0.01, min(0.99, market_price + noise))

            # YES side: buy YES if AI prob > market price
            yes_edge = ai_prob - market_price
            # NO side: buy NO if AI prob < market price (i.e., NO edge = market_price - ai_prob)
            no_edge = market_price - ai_prob

            if yes_edge >= self.min_edge:
                usdc = min(
                    self.max_position,
                    self.max_position * (yes_edge / 0.2),
                )
                profit = usdc * yes_edge
                opportunities.append(
                    PolymarketOpportunity(
                        token_id=token_id,
                        question=question,
                        category=category,
                        market_price=market_price,
                        ai_probability=round(ai_prob, 4),
                        edge=round(yes_edge, 4),
                        side="YES",
                        recommended_usdc=round(usdc, 2),
                        expected_profit_usdc=round(profit, 2),
                    )
                )
            elif no_edge >= self.min_edge:
                usdc = min(
                    self.max_position,
                    self.max_position * (no_edge / 0.2),
                )
                profit = usdc * no_edge
                opportunities.append(
                    PolymarketOpportunity(
                        token_id=token_id,
                        question=question,
                        category=category,
                        market_price=market_price,
                        ai_probability=round(ai_prob, 4),
                        edge=round(no_edge, 4),
                        side="NO",
                        recommended_usdc=round(usdc, 2),
                        expected_profit_usdc=round(profit, 2),
                    )
                )

        opportunities.sort(key=lambda o: o.expected_profit_usdc, reverse=True)
        return opportunities

    def execute_paper_trade(self, opp: PolymarketOpportunity) -> dict:
        """
        Simulate a Polymarket trade (paper trading).
        Returns a simulated fill result.
        """
        fill_price = opp.market_price + random.uniform(-0.002, 0.002)
        fill_price = max(0.01, min(0.99, fill_price))
        shares = opp.recommended_usdc / fill_price
        return {
            "status": "filled",
            "token_id": opp.token_id,
            "side": opp.side,
            "shares": round(shares, 2),
            "fill_price": round(fill_price, 4),
            "usdc_spent": opp.recommended_usdc,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Real API calls
    # ------------------------------------------------------------------

    def _fetch_real_markets(self, limit: int) -> list:
        """Fetch active markets from Polymarket CLOB API."""
        url = f"{self.clob_url}/markets"
        params = {"active": "true", "limit": limit}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        markets = data if isinstance(data, list) else data.get("data", [])

        enriched = []
        for m in markets:
            token_id = m.get("condition_id", m.get("id", ""))
            question = m.get("question", "Unknown")
            category = m.get("category", "Other")
            # Get mid price from order book
            try:
                book = self._fetch_real_book(token_id)
                best_bid = book.get("best_bid", 0.48)
                best_ask = book.get("best_ask", 0.52)
                mid = (best_bid + best_ask) / 2
            except Exception:
                mid = 0.5
            enriched.append({
                "token_id": token_id,
                "question": question,
                "category": category,
                "best_mid": round(mid, 4),
            })
        return enriched

    def _fetch_real_book(self, token_id: str) -> dict:
        """Fetch order book for a token."""
        url = f"{self.clob_url}/book"
        params = {"token_id": token_id}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        book = resp.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.48
        best_ask = float(asks[0]["price"]) if asks else 0.52
        return {"best_bid": best_bid, "best_ask": best_ask}

    # ------------------------------------------------------------------
    # Mock data
    # ------------------------------------------------------------------

    def _generate_mock_markets(self, limit: int) -> list:
        """Generate mock Polymarket markets."""
        markets = []
        pool = _MOCK_MARKETS * 3  # repeat pool for variety
        random.shuffle(pool)
        for i, tmpl in enumerate(pool[:limit]):
            # Add noise to base probability
            noise = random.gauss(0, 0.05)
            mid = max(0.05, min(0.95, tmpl["base_prob"] + noise))
            markets.append({
                "token_id": f"mock_token_{i:04d}",
                "question": tmpl["question"],
                "category": tmpl["category"],
                "best_mid": round(mid, 4),
            })
        return markets

    def _generate_mock_book(self, token_id: str) -> dict:
        """Generate a mock order book."""
        mid = random.uniform(0.2, 0.8)
        spread = random.uniform(0.005, 0.02)
        return {
            "best_bid": round(mid - spread / 2, 4),
            "best_ask": round(mid + spread / 2, 4),
        }

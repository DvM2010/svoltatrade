"""
CLAUDE BOT - Bookmaker Client
Fetches live odds from The Odds API (the-odds-api.com).
Falls back to realistic mock data when no API key is configured.
"""

import random
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import CONFIG, ODDS_API_BASE_URL, BOOKMAKER_LATENCY_PROFILE, BOOKMAKER_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

_MOCK_EVENTS = {
    "basketball_nba": [
        ("Lakers", "Celtics"),
        ("Warriors", "Nets"),
        ("Bucks", "Heat"),
        ("Suns", "Nuggets"),
        ("Clippers", "76ers"),
    ],
    "soccer_epl": [
        ("Arsenal", "Man City"),
        ("Liverpool", "Chelsea"),
        ("Tottenham", "Man United"),
        ("Newcastle", "Aston Villa"),
    ],
    "icehockey_nhl": [
        ("Rangers", "Bruins"),
        ("Avalanche", "Oilers"),
        ("Panthers", "Lightning"),
        ("Maple Leafs", "Canadiens"),
    ],
    "mma_mixed_martial_arts": [
        ("Adesanya", "Pereira"),
        ("Jones", "Miocic"),
        ("Poirier", "McGregor"),
        ("Ngannou", "Aspinall"),
    ],
    "americanfootball_nfl": [
        ("Cowboys", "Packers"),
        ("Chiefs", "Eagles"),
        ("Bills", "Dolphins"),
        ("49ers", "Seahawks"),
    ],
    "baseball_mlb": [
        ("Yankees", "Red Sox"),
        ("Dodgers", "Giants"),
        ("Astros", "Rangers"),
        ("Braves", "Mets"),
    ],
    "soccer_spain_la_liga": [
        ("Real Madrid", "Barcelona"),
        ("Atletico Madrid", "Sevilla"),
        ("Valencia", "Villarreal"),
    ],
    "tennis_atp_french_open": [
        ("Djokovic", "Alcaraz"),
        ("Medvedev", "Zverev"),
        ("Nadal", "Tsitsipas"),
    ],
}


def _generate_mock_odds_for_event(
    sport: str,
    home: str,
    away: str,
    bookmakers: list,
    now: datetime,
) -> dict:
    """
    Generate a mock odds payload that mimics The Odds API response format.
    Introduces realistic variance + occasional stale timestamps to simulate
    latency arbitrage windows.
    """
    base_home_prob = random.uniform(0.42, 0.62)
    base_away_prob = 1.0 - base_home_prob
    vig = random.uniform(1.04, 1.07)

    book_outcomes = []
    for book in bookmakers:
        stale = random.random() < 0.15
        latency_lo, latency_hi = BOOKMAKER_LATENCY_PROFILE.get(book, (10, 30))
        latency_ms = random.randint(latency_lo, latency_hi)

        if stale:
            last_update = (now - timedelta(seconds=random.randint(20, 60))).isoformat() + "Z"
            drift = random.uniform(0.04, 0.10)
            home_odds = round(1.0 / (base_home_prob * vig) + drift, 3)
            away_odds = round(1.0 / (base_away_prob * vig) - drift * 0.8, 3)
        else:
            last_update = (now - timedelta(seconds=random.randint(0, 10))).isoformat() + "Z"
            noise = random.uniform(-0.02, 0.02)
            home_odds = round(1.0 / (base_home_prob * vig) + noise, 3)
            away_odds = round(1.0 / (base_away_prob * vig) - noise * 0.5, 3)

        home_odds = max(1.01, home_odds)
        away_odds = max(1.01, away_odds)

        book_outcomes.append({
            "key": book,
            "title": BOOKMAKER_DISPLAY_NAMES.get(book, book),
            "last_update": last_update,
            "latency_ms": latency_ms,
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": home, "price": home_odds},
                        {"name": away, "price": away_odds},
                    ],
                }
            ],
        })

    return {
        "id": f"mock_{sport}_{home}_{away}".replace(" ", "_").lower(),
        "sport_key": sport,
        "sport_title": sport.replace("_", " ").title(),
        "commence_time": (now + timedelta(hours=random.randint(1, 48))).isoformat() + "Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": book_outcomes,
    }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class BookmakerClient:
    """
    Fetches sports odds from The Odds API or falls back to mock data.
    """

    def __init__(self):
        self.api_key = CONFIG.get("odds_api_key", "")
        self.base_url = ODDS_API_BASE_URL
        self._latency_cache: dict = {}

    def get_live_odds(self, sport: str, bookmakers: list) -> list:
        """
        Return a list of event dicts for the given sport, each containing
        odds from the requested bookmakers.
        """
        if self.api_key:
            try:
                return self._fetch_real_odds(sport, bookmakers)
            except Exception:
                pass
        return self._generate_mock_sport_odds(sport, bookmakers)

    def get_all_sports_odds(self, bookmakers: list) -> dict:
        """
        Return odds for all configured sports.
        Returns: {sport_key: [event, ...]}
        """
        results = {}
        for sport in CONFIG["sports"]:
            try:
                results[sport] = self.get_live_odds(sport, bookmakers)
            except Exception:
                results[sport] = []
        return results

    def get_bookmaker_latency_ms(self, bookmaker: str) -> int:
        """
        Return simulated latency for a bookmaker in ms.
        """
        if bookmaker in self._latency_cache:
            return self._latency_cache[bookmaker]
        lo, hi = BOOKMAKER_LATENCY_PROFILE.get(bookmaker, (10, 30))
        return random.randint(lo, hi)

    def _fetch_real_odds(self, sport: str, bookmakers: list) -> list:
        """Fetch from The Odds API."""
        books_param = ",".join(bookmakers)
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h",
            "bookmakers": books_param,
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        events = resp.json()

        now = datetime.now(timezone.utc)
        for event in events:
            for book in event.get("bookmakers", []):
                lu = book.get("last_update", "")
                try:
                    lu_dt = datetime.fromisoformat(lu.replace("Z", "+00:00"))
                    age_ms = int((now - lu_dt).total_seconds() * 1000)
                    book["latency_ms"] = max(0, age_ms)
                    self._latency_cache[book["key"]] = book["latency_ms"]
                except Exception:
                    book["latency_ms"] = self.get_bookmaker_latency_ms(book["key"])
        return events

    def _generate_mock_sport_odds(self, sport: str, bookmakers: list) -> list:
        """Generate realistic mock odds for a sport."""
        events_pool = _MOCK_EVENTS.get(sport, [("Team A", "Team B")])
        count = random.randint(2, min(4, len(events_pool)))
        selected = random.sample(events_pool, count)
        now = datetime.now(timezone.utc)
        return [
            _generate_mock_odds_for_event(sport, home, away, bookmakers, now)
            for home, away in selected
        ]

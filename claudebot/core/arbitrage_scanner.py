"""
CLAUDE BOT - Arbitrage Scanner
Detects latency arbitrage and sure-bet opportunities across bookmakers.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import CONFIG, SPORT_DISPLAY_NAMES, BOOKMAKER_DISPLAY_NAMES
from core.bookmaker_client import BookmakerClient


@dataclass
class ArbitrageOpportunity:
    sport: str
    event: str               # "Cowboys vs Packers"
    market: str              # "Live ML", "Props", etc.
    bookmaker_buy: str       # Book with favorable odds (we buy here)
    bookmaker_hedge: str     # Book with opposing odds (we hedge here)
    odds_buy: float
    odds_hedge: float
    delta_odds: float        # Absolute odds difference
    edge_pct: float          # Estimated edge as a fraction
    latency_indicator_ms: int
    confidence: float        # 0..1
    recommended_stake_usd: float
    expected_profit_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    arb_type: str = "latency"  # "latency" or "surebet"


# ---------------------------------------------------------------------------
# Market labels for variety in the UI
# ---------------------------------------------------------------------------
_MARKET_LABELS = [
    "Live ML", "Props", "1H ML", "Spread", "Alt Line",
    "2H ML", "Game Props", "Player Props",
]


class ArbitrageScanner:
    """
    Scans sports odds feeds for latency arbitrage and sure-bet opportunities.
    """

    def __init__(self, bookmaker_client: Optional[BookmakerClient] = None):
        self.client = bookmaker_client or BookmakerClient()
        self.threshold = CONFIG["min_edge_threshold"]
        self.bookmakers = CONFIG["bookmakers"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan_all_sports(self) -> list:
        """
        Fetch odds across all configured sports and return a list of
        ArbitrageOpportunity objects sorted by expected_profit_usd descending.
        """
        all_odds = self.client.get_all_sports_odds(self.bookmakers)
        opportunities = []
        for sport, events in all_odds.items():
            latency_opps = self.find_latency_arb({sport: events})
            sure_opps = self.find_sure_bets({sport: events})
            opportunities.extend(latency_opps)
            opportunities.extend(sure_opps)

        # Sort best first
        opportunities.sort(key=lambda o: o.expected_profit_usd, reverse=True)
        return opportunities

    def find_latency_arb(self, sport_odds: dict) -> list:
        """
        Detect latency arbitrage: one bookmaker's odds are significantly
        higher than consensus AND their last_update timestamp is older.
        """
        opportunities = []
        now = datetime.now(timezone.utc)

        for sport, events in sport_odds.items():
            sport_label = SPORT_DISPLAY_NAMES.get(sport, sport)
            for event in events:
                home = event.get("home_team", "Home")
                away = event.get("away_team", "Away")
                event_name = f"{home} vs {away}"
                bookmakers_data = event.get("bookmakers", [])

                if len(bookmakers_data) < 2:
                    continue

                # Build a table of {team: [(odds, book_key, last_update, latency_ms)]}
                team_odds: dict = {}
                for book in bookmakers_data:
                    for market in book.get("markets", []):
                        if market["key"] != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            team = outcome["name"]
                            price = outcome["price"]
                            lu_str = book.get("last_update", "")
                            try:
                                lu_dt = datetime.fromisoformat(
                                    lu_str.replace("Z", "+00:00")
                                )
                                age_s = (now - lu_dt).total_seconds()
                            except Exception:
                                age_s = 0
                            if team not in team_odds:
                                team_odds[team] = []
                            team_odds[team].append({
                                "odds": price,
                                "book": book["key"],
                                "age_s": age_s,
                                "latency_ms": book.get("latency_ms", 15),
                            })

                # For each team, find if highest-odds book is also oldest
                for team, entries in team_odds.items():
                    if len(entries) < 2:
                        continue
                    entries.sort(key=lambda x: x["odds"], reverse=True)
                    best = entries[0]
                    second = entries[1]

                    delta = best["odds"] - second["odds"]
                    # Consensus odds = median of all
                    all_odds_vals = [e["odds"] for e in entries]
                    consensus = sorted(all_odds_vals)[len(all_odds_vals) // 2]
                    edge = (best["odds"] - consensus) / consensus

                    # Only flag if: significant price discrepancy AND stale
                    staleness_threshold = 15  # seconds
                    is_stale = best["age_s"] > staleness_threshold

                    if edge >= self.threshold and is_stale:
                        stake = self._kelly_stake(edge, best["odds"])
                        profit = stake * edge
                        lat_ms = max(
                            best["latency_ms"],
                            second["latency_ms"],
                        )
                        opp = ArbitrageOpportunity(
                            sport=sport_label,
                            event=event_name,
                            market=random.choice(_MARKET_LABELS),
                            bookmaker_buy=best["book"],
                            bookmaker_hedge=second["book"],
                            odds_buy=best["odds"],
                            odds_hedge=second["odds"],
                            delta_odds=round(delta, 4),
                            edge_pct=round(edge, 4),
                            latency_indicator_ms=lat_ms,
                            confidence=min(0.99, 0.5 + edge * 5),
                            recommended_stake_usd=round(stake, 2),
                            expected_profit_usd=round(profit, 2),
                            arb_type="latency",
                        )
                        opportunities.append(opp)

        return opportunities

    def find_sure_bets(self, sport_odds: dict) -> list:
        """
        Detect sure-bets (risk-free arb): sum of inverse odds < 1 across books.
        """
        opportunities = []
        for sport, events in sport_odds.items():
            sport_label = SPORT_DISPLAY_NAMES.get(sport, sport)
            for event in events:
                home = event.get("home_team", "Home")
                away = event.get("away_team", "Away")
                event_name = f"{home} vs {away}"
                bookmakers_data = event.get("bookmakers", [])

                # Best odds per team across all books
                best_per_team: dict = {}
                for book in bookmakers_data:
                    for market in book.get("markets", []):
                        if market["key"] != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            team = outcome["name"]
                            price = outcome["price"]
                            if team not in best_per_team or price > best_per_team[team]["odds"]:
                                best_per_team[team] = {
                                    "odds": price,
                                    "book": book["key"],
                                    "latency_ms": book.get("latency_ms", 15),
                                }

                if len(best_per_team) < 2:
                    continue

                teams = list(best_per_team.keys())
                implied_sum = sum(1.0 / best_per_team[t]["odds"] for t in teams)

                if implied_sum < 1.0:
                    arb_pct = 1.0 - implied_sum
                    if arb_pct >= self.threshold * 0.5:
                        # Two-way sure bet
                        t1, t2 = teams[0], teams[1]
                        b1 = best_per_team[t1]
                        b2 = best_per_team[t2]
                        total_stake = CONFIG["max_bet_usd"]
                        profit = total_stake * arb_pct
                        opp = ArbitrageOpportunity(
                            sport=sport_label,
                            event=event_name,
                            market="Sure Bet",
                            bookmaker_buy=b1["book"],
                            bookmaker_hedge=b2["book"],
                            odds_buy=b1["odds"],
                            odds_hedge=b2["odds"],
                            delta_odds=round(b1["odds"] - b2["odds"], 4),
                            edge_pct=round(arb_pct, 4),
                            latency_indicator_ms=max(
                                b1["latency_ms"], b2["latency_ms"]
                            ),
                            confidence=min(0.99, 0.7 + arb_pct * 3),
                            recommended_stake_usd=round(total_stake / 2, 2),
                            expected_profit_usd=round(profit, 2),
                            arb_type="surebet",
                        )
                        opportunities.append(opp)

        return opportunities

    def calculate_arb_profit(self, odds1: float, odds2: float) -> float:
        """
        Calculate the guaranteed profit percentage for a two-way sure bet.
        Returns 0 if not profitable.
        """
        implied_sum = (1.0 / odds1) + (1.0 / odds2)
        if implied_sum >= 1.0:
            return 0.0
        return round(1.0 - implied_sum, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kelly_stake(self, edge: float, odds: float) -> float:
        """Fractional Kelly criterion stake."""
        if odds <= 1.0:
            return 0.0
        b = odds - 1.0          # net odds on winning
        p = 0.5 + edge / 2.0    # estimated win probability
        q = 1.0 - p
        kelly = (b * p - q) / b
        kelly = max(0, kelly) * CONFIG["kelly_fraction"]
        wallet = CONFIG["wallet_balance"]
        stake = wallet * kelly
        return min(stake, CONFIG["max_bet_usd"])

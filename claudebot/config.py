"""
CLAUDE BOT - Configuration
All settings for the sport latency arbitrage engine.
"""

import os

CONFIG = {
    # API Keys (set via environment variables or edit directly)
    "odds_api_key": os.environ.get("ODDS_API_KEY", ""),
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    "polymarket_api_key": os.environ.get("POLYMARKET_API_KEY", ""),
    "polymarket_api_secret": os.environ.get("POLYMARKET_API_SECRET", ""),
    "polymarket_api_passphrase": os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
    "private_key": os.environ.get("POLYGON_PRIVATE_KEY", ""),

    # Arbitrage Settings
    "min_edge_threshold": 0.03,  # 3% minimum edge
    "max_latency_ms": 30,        # Max acceptable latency window
    "scan_interval_seconds": 5,

    # Sports to scan
    "sports": [
        "basketball_nba",
        "soccer_epl",
        "icehockey_nhl",
        "mma_mixed_martial_arts",
        "americanfootball_nfl",
        "baseball_mlb",
        "soccer_spain_la_liga",
        "tennis_atp_french_open",
    ],

    # Bookmakers (10 books)
    "bookmakers": [
        "betmgm", "draftkings", "fanduel", "pointsbetus",
        "williamhill_us", "betrivers", "unibet", "bovada",
        "mybookieag", "betonlineag"
    ],

    # Risk Management
    "wallet_balance": 1204.50,   # Starting simulated balance
    "max_bet_usd": 100,
    "kelly_fraction": 0.25,      # Fractional Kelly
    "max_concurrent_bets": 5,
    "daily_loss_limit": 200,

    # Polymarket
    "polymarket_enabled": True,
    "polymarket_min_edge": 0.05,
    "polymarket_max_position_usdc": 50,

    # Mode
    "paper_trading": True,       # Simulate trades, no real money
    "verbose_logging": True,
}

# Sport display names for UI
SPORT_DISPLAY_NAMES = {
    "basketball_nba": "NBA",
    "soccer_epl": "EPL",
    "icehockey_nhl": "NHL",
    "mma_mixed_martial_arts": "UFC",
    "americanfootball_nfl": "NFL",
    "baseball_mlb": "MLB",
    "soccer_spain_la_liga": "La Liga",
    "tennis_atp_french_open": "Tennis",
}

# Bookmaker display names
BOOKMAKER_DISPLAY_NAMES = {
    "betmgm": "BetMGM",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "pointsbetus": "PointsBet",
    "williamhill_us": "Caesars",
    "betrivers": "BetRivers",
    "unibet": "Unibet",
    "bovada": "Bovada",
    "mybookieag": "MyBookie",
    "betonlineag": "BetOnline",
}

# Latency profile: simulated latency ranges (ms) per bookmaker
# Reflects real-world observed update speeds (fastest -> slowest)
BOOKMAKER_LATENCY_PROFILE = {
    "draftkings":     (2, 6),
    "fanduel":        (3, 8),
    "betmgm":         (5, 12),
    "pointsbetus":    (6, 14),
    "betrivers":      (8, 18),
    "williamhill_us": (10, 22),
    "unibet":         (12, 24),
    "bovada":         (15, 28),
    "mybookieag":     (18, 32),
    "betonlineag":    (20, 35),
}

# Polymarket CLOB API base URL
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"

# The Odds API base URL
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

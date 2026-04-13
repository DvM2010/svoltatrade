"""
CLAUDE BOT - Wallet
Tracks the paper trading (or real) wallet balance and trade history.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from config import CONFIG

TRADES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trades.json")


class Wallet:
    """
    Manages the simulated or real balance for CLAUDE BOT.
    Persists trade history to data/trades.json.
    """

    def __init__(self, starting_balance: Optional[float] = None):
        self._balance = starting_balance or CONFIG["wallet_balance"]
        self._starting_balance = self._balance
        self._trades: list = []
        self._session_pnl: float = 0.0
        self._wins: int = 0
        self._losses: int = 0
        self._load_trades()

    # ------------------------------------------------------------------
    # Balance management
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        return round(self._balance, 2)

    def get_session_pnl(self) -> float:
        return round(self._session_pnl, 2)

    def get_total_pnl(self) -> float:
        return round(self._balance - self._starting_balance, 2)

    def update_balance(self, pnl: float):
        """Adjust balance by pnl amount (positive = profit, negative = loss)."""
        self._balance += pnl
        self._session_pnl += pnl
        if pnl > 0:
            self._wins += 1
        elif pnl < 0:
            self._losses += 1

    def get_win_rate(self) -> float:
        """Return win rate as a float 0..1."""
        total = self._wins + self._losses
        if total == 0:
            return 1.0
        return self._wins / total

    def get_trade_count(self) -> int:
        return len(self._trades)

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict):
        """Add a trade record and persist to disk."""
        trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._trades.append(trade)
        self._save_trades()

    def get_recent_trades(self, n: int = 20) -> list:
        return list(reversed(self._trades[-n:]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_trades(self):
        """Load existing trade history from disk."""
        try:
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, "r") as f:
                    data = json.load(f)
                    self._trades = data.get("trades", [])
                    saved_balance = data.get("balance")
                    if saved_balance is not None:
                        self._balance = saved_balance
                        self._starting_balance = data.get(
                            "starting_balance", saved_balance
                        )
                    # Recalculate stats
                    for t in self._trades:
                        pnl = t.get("pnl", 0)
                        if pnl > 0:
                            self._wins += 1
                        elif pnl < 0:
                            self._losses += 1
        except Exception:
            pass

    def _save_trades(self):
        """Persist balance and trades to disk."""
        try:
            os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
            with open(TRADES_FILE, "w") as f:
                json.dump(
                    {
                        "balance": self._balance,
                        "starting_balance": self._starting_balance,
                        "trades": self._trades,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

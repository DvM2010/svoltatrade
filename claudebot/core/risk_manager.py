"""
CLAUDE BOT - Risk Manager
Position sizing using fractional Kelly criterion.
Enforces daily loss limits, concurrent bet caps, and drawdown protection.
"""

from datetime import datetime, timezone, date
from typing import Optional

from config import CONFIG


class RiskManager:
    """
    Manages risk across all open and pending positions.
    Enforces:
    - Max concurrent bets
    - Daily loss limit
    - Kelly-based position sizing
    - Max single bet size
    """

    def __init__(self, wallet=None):
        self.wallet = wallet
        self.max_bet = CONFIG["max_bet_usd"]
        self.kelly_fraction = CONFIG["kelly_fraction"]
        self.max_concurrent = CONFIG["max_concurrent_bets"]
        self.daily_loss_limit = CONFIG["daily_loss_limit"]
        self._open_bets: list = []
        self._daily_pnl: float = 0.0
        self._daily_reset_date: date = date.today()

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------

    def can_place_bet(self) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        """
        self._maybe_reset_daily()

        if len(self._open_bets) >= self.max_concurrent:
            return False, f"Max concurrent bets reached ({self.max_concurrent})"

        if self._daily_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss limit hit (${-self._daily_pnl:.2f})"

        return True, "OK"

    def approve_opportunity(self, opp) -> tuple:
        """
        Gate an ArbitrageOpportunity through risk checks.
        Returns (approved: bool, stake_usd: float, reason: str).
        """
        allowed, reason = self.can_place_bet()
        if not allowed:
            return False, 0.0, reason

        if opp.confidence < 0.50:
            return False, 0.0, f"Confidence too low ({opp.confidence:.0%})"

        stake = self._size_position(opp)
        if stake < 1.0:
            return False, 0.0, "Stake below minimum ($1)"

        return True, stake, "Approved"

    def register_open_bet(self, bet_id: str, stake: float):
        """Record a new open bet."""
        self._open_bets.append({"id": bet_id, "stake": stake})

    def settle_bet(self, bet_id: str, pnl: float):
        """Mark a bet as settled and update daily P&L."""
        self._open_bets = [b for b in self._open_bets if b["id"] != bet_id]
        self._daily_pnl += pnl
        if self.wallet:
            self.wallet.update_balance(pnl)

    def get_stats(self) -> dict:
        """Return current risk stats."""
        self._maybe_reset_daily()
        return {
            "open_bets": len(self._open_bets),
            "daily_pnl": round(self._daily_pnl, 2),
            "remaining_daily_loss": round(
                self.daily_loss_limit + self._daily_pnl, 2
            ),
            "at_daily_limit": self._daily_pnl <= -self.daily_loss_limit,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _size_position(self, opp) -> float:
        """
        Compute position size using fractional Kelly criterion.
        Falls back to the pre-calculated recommended_stake_usd if available.
        """
        if hasattr(opp, "recommended_stake_usd") and opp.recommended_stake_usd > 0:
            return min(opp.recommended_stake_usd, self.max_bet)

        odds = opp.odds_buy
        if odds <= 1.0:
            return 0.0

        edge = opp.edge_pct
        b = odds - 1.0
        p = 0.5 + edge / 2.0
        q = 1.0 - p

        kelly = max(0, (b * p - q) / b) * self.kelly_fraction
        balance = self.wallet.get_balance() if self.wallet else CONFIG["wallet_balance"]
        stake = balance * kelly
        return round(min(stake, self.max_bet), 2)

    def _maybe_reset_daily(self):
        """Reset daily P&L at midnight."""
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_reset_date = today

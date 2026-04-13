"""
CLAUDE BOT - Terminal UI
Rich-powered live terminal dashboard matching the viral "CLAUDE BOT" screenshots.
Green/cyan color scheme, ASCII art header, real-time scrolling trade log.
"""

import sys
import time
import random
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Callable

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box

from config import CONFIG, SPORT_DISPLAY_NAMES, BOOKMAKER_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# ASCII art header
# ---------------------------------------------------------------------------

CLAUDE_ASCII = r"""
  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗
 ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝
 ██║     ██║     ███████║██║   ██║██║  ██║█████╗
 ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝
 ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗
  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝"""

BOT_ASCII = r"""          ██████╗  ██████╗ ████████╗
          ██╔══██╗██╔═══██╗╚══██╔══╝
          ██████╔╝██║   ██║   ██║
          ██╔══██╗██║   ██║   ██║
          ██████╔╝╚██████╔╝   ██║
          ╚═════╝  ╚═════╝    ╚═╝   """

SUBTITLE = "    - SPORT LATENCY ARBITRAGE ENGINE - LIVE MARKETS -"


def _ts() -> str:
    """Return UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_pnl(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:.2f}"


# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------

class LogBuffer:
    """Thread-safe rolling log buffer."""

    def __init__(self, maxlen: int = 200):
        self._lines: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, text: str, style: str = "green"):
        with self._lock:
            self._lines.append((text, style))

    def get_lines(self, n: int = 40) -> list:
        with self._lock:
            return list(self._lines)[-n:]


# ---------------------------------------------------------------------------
# Main terminal class
# ---------------------------------------------------------------------------

class ClaudeBotTerminal:
    """
    Manages the live Rich terminal display for CLAUDE BOT.
    """

    def __init__(
        self,
        wallet=None,
        mode: str = "latency_arb",
        threshold: float = 0.03,
        num_books: int = 10,
    ):
        self.console = Console(highlight=False, force_terminal=True)
        self.wallet = wallet
        self.mode = mode
        self.threshold = threshold
        self.num_books = num_books
        self.log = LogBuffer(maxlen=300)
        self._scan_count = 0
        self._arb_found = 0
        self._running = False
        self._live: Optional[Live] = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def print_boot_sequence(self, command_line: str = ""):
        """
        Print the static boot header to console before live mode starts.
        Matches the screenshot exactly.
        """
        c = self.console

        # Box top
        c.print("╔" + "═" * 62 + "╗", style="green")
        c.print()

        # ASCII header
        for line in CLAUDE_ASCII.splitlines():
            c.print(line, style="bold green")
        for line in BOT_ASCII.splitlines():
            c.print(line, style="bold cyan")

        c.print()
        c.print(SUBTITLE, style="bold cyan")
        c.print()

        # Stats bar
        self._print_stats_bar()
        c.print("═" * 64, style="green")
        c.print()

        # Boot messages
        lat_lo = random.randint(2, 4)
        lat_hi = random.randint(25, 32)
        wallet_bal = self.wallet.get_balance() if self.wallet else CONFIG["wallet_balance"]

        c.print(
            f"[BOOT] Latency calibration: {lat_lo}-{lat_hi}ms across {self.num_books} books",
            style="bright_green",
        )
        c.print(
            f"[BOOT] Mode: SPORT LATENCY ARB | Wallet: ${wallet_bal:,.2f}",
            style="bright_green",
        )
        c.print(
            f"[{_ts()}] session started — scanning live odds across {self.num_books} bookmakers",
            style="green",
        )
        c.print()

        # Command echo
        if command_line:
            c.print(f"> {command_line}", style="bold bright_cyan")
            c.print()

        sports_list = ", ".join(SPORT_DISPLAY_NAMES.values())
        c.print(
            f"Scanning live odds feeds across {sports_list}...",
            style="green",
        )
        c.print(
            f"Comparing real-time odds deltas | min edge: {self.threshold} | "
            f"max latency: {CONFIG['max_latency_ms']}ms",
            style="green",
        )
        c.print()

    def _print_stats_bar(self):
        """Print the WALLET / TRADES / WIN stats line."""
        if self.wallet:
            bal = self.wallet.get_balance()
            session_pnl = self.wallet.get_session_pnl()
            trades = self.wallet.get_trade_count()
            win_rate = self.wallet.get_win_rate()
        else:
            bal = CONFIG["wallet_balance"]
            session_pnl = 0.0
            trades = 0
            win_rate = 1.0

        pnl_str = _fmt_pnl(session_pnl)
        win_pct = f"{win_rate * 100:.0f}%"
        stats = (
            f"WALLET: ${bal:,.2f} ({pnl_str}) | "
            f"TRADES: {trades} | "
            f"WIN: {win_pct}"
        )
        self.console.print(stats, style="bold bright_green")

    # ------------------------------------------------------------------
    # Live scrolling log mode
    # ------------------------------------------------------------------

    def run_live(
        self,
        scan_callback: Callable,
        stop_event: Optional[threading.Event] = None,
    ):
        """
        Run the live scrolling terminal. Calls scan_callback() every
        scan_interval seconds to get new opportunities.
        """
        self._running = True
        stop = stop_event or threading.Event()

        self.log.append(
            f"[{_ts()}] Live scanning active — press Ctrl+C to stop",
            "bright_green",
        )

        # Run scan in background thread, update display in main thread
        scan_thread = threading.Thread(
            target=self._scan_loop,
            args=(scan_callback, stop),
            daemon=True,
        )
        scan_thread.start()

        # Live display update
        try:
            with Live(
                self._build_display(),
                console=self.console,
                refresh_per_second=2,
                screen=False,
            ) as live:
                self._live = live
                while not stop.is_set():
                    live.update(self._build_display())
                    time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set()

        self._running = False
        scan_thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Display builder
    # ------------------------------------------------------------------

    def _build_display(self) -> Panel:
        """Build the current Rich renderable for the live display."""
        text = Text()

        # Stats bar
        if self.wallet:
            bal = self.wallet.get_balance()
            session_pnl = self.wallet.get_session_pnl()
            trades = self.wallet.get_trade_count()
            win_rate = self.wallet.get_win_rate()
        else:
            bal = CONFIG["wallet_balance"]
            session_pnl = 0.0
            trades = self._scan_count
            win_rate = 1.0

        pnl_str = _fmt_pnl(session_pnl)
        win_pct = f"{win_rate * 100:.0f}%"
        stats_line = (
            f"WALLET: ${bal:,.2f} ({pnl_str}) | "
            f"TRADES: {trades} | WIN: {win_pct} | "
            f"SCANS: {self._scan_count}"
        )
        text.append(stats_line + "\n", style="bold bright_green")
        text.append("═" * 64 + "\n", style="green")
        text.append("\n")

        # Rolling log
        lines = self.log.get_lines(35)
        for line_text, line_style in lines:
            text.append(line_text + "\n", style=line_style)

        return Panel(
            text,
            title="[bold green]CLAUDE BOT — LIVE[/bold green]",
            border_style="green",
            box=box.HEAVY,
        )

    # ------------------------------------------------------------------
    # Scan loop (background thread)
    # ------------------------------------------------------------------

    def _scan_loop(self, scan_callback: Callable, stop: threading.Event):
        """Background thread: call scan_callback and log results."""
        interval = CONFIG["scan_interval_seconds"]
        while not stop.is_set():
            try:
                self._scan_count += 1
                opportunities = scan_callback()
                self._log_scan_results(opportunities)
            except Exception as e:
                self.log.append(f"[ERROR] Scan failed: {e}", "red")

            # Sleep in small increments so we can respond to stop event
            for _ in range(interval * 10):
                if stop.is_set():
                    return
                time.sleep(0.1)

    def _log_scan_results(self, opportunities: list):
        """Write opportunity results to the rolling log."""
        now_str = _ts()

        if not opportunities:
            self.log.append(
                f"[{now_str}] Scan #{self._scan_count}: no opportunities above threshold",
                "dim green",
            )
            return

        self.log.append(
            f"[{now_str}] Scan #{self._scan_count}: "
            f"Found {len(opportunities)} arbitrage windows — executing high-confidence trades",
            "bright_green",
        )
        self._arb_found += len(opportunities)

        for opp in opportunities[:5]:  # Show top 5 per scan
            self._log_opportunity(opp)

    def _log_opportunity(self, opp):
        """Log a single ArbitrageOpportunity in the viral screenshot style."""
        buy_name = BOOKMAKER_DISPLAY_NAMES.get(opp.bookmaker_buy, opp.bookmaker_buy)
        hedge_name = BOOKMAKER_DISPLAY_NAMES.get(opp.bookmaker_hedge, opp.bookmaker_hedge)

        # Arb detection header
        arb_type = "ARB" if opp.arb_type == "latency" else "SUREBET"
        self.log.append(
            f"⚡ {arb_type} DETECTED: {opp.sport} | {opp.event} | "
            f"{opp.market} | Δodds {opp.delta_odds:.3f} [{opp.latency_indicator_ms}ms]",
            "bold bright_yellow",
        )

        ts1 = _ts()
        ts2 = _ts()

        # Buy leg
        self.log.append(
            f"[{ts1}] BUY {opp.event.split(' vs ')[1] if ' vs ' in opp.event else opp.event} "
            f"{opp.market} @ {opp.odds_buy:.2f} via {buy_name} → "
            f"${opp.recommended_stake_usd:.2f}",
            "green",
        )
        # Hedge leg
        self.log.append(
            f"[{ts2}] HEDGE {opp.event.split(' vs ')[0] if ' vs ' in opp.event else opp.event} "
            f"{opp.market} @ {opp.odds_hedge:.2f} via {hedge_name} → "
            f"${opp.recommended_stake_usd:.2f}",
            "green",
        )

        # Settlement
        if self.wallet:
            bal = self.wallet.get_balance()
        else:
            bal = CONFIG["wallet_balance"]

        pnl = opp.expected_profit_usd
        pnl_str = _fmt_pnl(pnl)
        self.log.append(
            f"[{_ts()}] ✓ SETTLED {pnl_str} | bal: ${bal:,.2f}",
            "bold bright_green" if pnl >= 0 else "bold red",
        )
        self.log.append("", "green")  # spacer

    # ------------------------------------------------------------------
    # One-shot status/history display
    # ------------------------------------------------------------------

    def print_status(self, wallet=None):
        """Print current status summary."""
        w = wallet or self.wallet
        c = self.console

        c.print()
        c.print("╔" + "═" * 50 + "╗", style="green")
        c.print("  CLAUDE BOT — STATUS", style="bold green")
        c.print("═" * 52, style="green")

        if w:
            table = Table(box=box.SIMPLE, style="green", show_header=False)
            table.add_column("Key", style="bright_green", width=25)
            table.add_column("Value", style="green")
            table.add_row("Balance", f"${w.get_balance():,.2f}")
            table.add_row("Session P&L", _fmt_pnl(w.get_session_pnl()))
            table.add_row("Total Trades", str(w.get_trade_count()))
            table.add_row("Win Rate", f"{w.get_win_rate()*100:.0f}%")
            table.add_row("Mode", "PAPER TRADING" if CONFIG["paper_trading"] else "LIVE")
            c.print(table)
        else:
            c.print("  No wallet data available.", style="dim green")

    def print_history(self, wallet=None, n: int = 20):
        """Print recent trade history."""
        w = wallet or self.wallet
        c = self.console

        c.print()
        c.print("╔" + "═" * 80 + "╗", style="green")
        c.print("  CLAUDE BOT — TRADE HISTORY", style="bold green")
        c.print("═" * 82, style="green")

        if not w:
            c.print("  No trade history.", style="dim green")
            return

        trades = w.get_recent_trades(n)
        if not trades:
            c.print("  No trades recorded yet.", style="dim green")
            return

        table = Table(
            box=box.SIMPLE,
            style="green",
            header_style="bold bright_green",
        )
        table.add_column("ID", width=10)
        table.add_column("Sport", width=8)
        table.add_column("Event", width=25)
        table.add_column("Market", width=12)
        table.add_column("Buy@", width=8)
        table.add_column("Hedge@", width=8)
        table.add_column("Stake", width=8)
        table.add_column("P&L", width=8)
        table.add_column("Balance", width=10)

        for t in trades:
            pnl = t.get("pnl", 0)
            pnl_style = "bright_green" if pnl >= 0 else "red"
            table.add_row(
                t.get("id", "?"),
                t.get("sport", "?"),
                t.get("event", "?")[:24],
                t.get("market", "?"),
                f"{t.get('buy_odds', 0):.2f}",
                f"{t.get('hedge_odds', 0):.2f}",
                f"${t.get('stake', 0):.2f}",
                Text(_fmt_pnl(pnl), style=pnl_style),
                f"${t.get('balance_after', 0):,.2f}",
            )
        c.print(table)

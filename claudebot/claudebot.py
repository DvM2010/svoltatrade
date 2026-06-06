#!/usr/bin/env python3
"""
CLAUDE BOT - Sport Latency Arbitrage Engine
============================================
A viral-style CLI terminal that scans 10 bookmakers for latency arbitrage
and Polymarket prediction market opportunities.

Usage:
    python claudebot.py scan --mode latency_arb --sports all --books 10 --threshold 0.03
    python claudebot.py scan --mode polymarket --threshold 0.05
    python claudebot.py scan --mode all --threshold 0.03
    python claudebot.py status
    python claudebot.py history
"""

import sys
import os
import argparse
import threading
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, SPORT_DISPLAY_NAMES
from core.wallet import Wallet
from core.risk_manager import RiskManager
from core.bookmaker_client import BookmakerClient
from core.arbitrage_scanner import ArbitrageScanner
from core.polymarket_client import PolymarketClient
from core.ai_analyzer import AIAnalyzer
from core.trade_executor import TradeExecutor
from ui.terminal import ClaudeBotTerminal


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claudebot",
        description="CLAUDE BOT — Sport Latency Arbitrage Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python claudebot.py scan --mode latency_arb --sports all --books 10 --threshold 0.03
  python claudebot.py scan --mode polymarket --threshold 0.05
  python claudebot.py scan --mode all --threshold 0.03
  python claudebot.py status
  python claudebot.py history --n 50
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # scan
    scan = sub.add_parser("scan", help="Start live arbitrage scanning")
    scan.add_argument(
        "--mode",
        choices=["latency_arb", "surebet", "polymarket", "all"],
        default="latency_arb",
        help="Scanning mode (default: latency_arb)",
    )
    scan.add_argument(
        "--sports",
        default="all",
        help="Sports to scan: all, or comma-separated list (e.g. nba,nfl)",
    )
    scan.add_argument(
        "--books",
        type=int,
        default=10,
        help="Number of bookmakers to scan (default: 10)",
    )
    scan.add_argument(
        "--threshold",
        type=float,
        default=CONFIG["min_edge_threshold"],
        help="Minimum edge threshold (default: 0.03 = 3%%)",
    )
    scan.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default: True)",
    )
    scan.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Use real money (requires API keys configured)",
    )

    # status
    status = sub.add_parser("status", help="Show current bot status and wallet")

    # history
    history = sub.add_parser("history", help="Show trade history")
    history.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of recent trades to show (default: 20)",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_scan(args):
    """Run the live arbitrage scanner."""
    # Apply args to config
    if args.live:
        CONFIG["paper_trading"] = False
    CONFIG["min_edge_threshold"] = args.threshold

    num_books = min(args.books, len(CONFIG["bookmakers"]))
    bookmakers = CONFIG["bookmakers"][:num_books]
    mode = args.mode

    # Filter sports if specified
    if args.sports != "all":
        requested = [s.strip().lower() for s in args.sports.split(",")]
        # Match against sport keys or display names
        filtered = []
        for sk, sname in SPORT_DISPLAY_NAMES.items():
            if sk in requested or sname.lower() in requested:
                filtered.append(sk)
        if filtered:
            CONFIG["sports"] = filtered

    # Initialize components
    wallet = Wallet()
    risk_mgr = RiskManager(wallet=wallet)
    book_client = BookmakerClient()
    arb_scanner = ArbitrageScanner(bookmaker_client=book_client)
    poly_client = PolymarketClient()
    ai_analyzer = AIAnalyzer()
    executor = TradeExecutor(wallet=wallet, risk_manager=risk_mgr)

    # Terminal UI
    terminal = ClaudeBotTerminal(
        wallet=wallet,
        mode=mode,
        threshold=args.threshold,
        num_books=num_books,
    )

    # Build the command line string for display
    cmd_str = (
        f"claudebot scan --mode {mode} --sports {args.sports} "
        f"--books {num_books} --threshold {args.threshold}"
    )

    # Print boot sequence
    terminal.print_boot_sequence(command_line=cmd_str)

    # Build scan callback based on mode
    def do_scan():
        results = []

        if mode in ("latency_arb", "surebet", "all"):
            opportunities = arb_scanner.scan_all_sports()
            # Filter by mode
            if mode == "latency_arb":
                opportunities = [o for o in opportunities if o.arb_type == "latency"]
            elif mode == "surebet":
                opportunities = [o for o in opportunities if o.arb_type == "surebet"]

            # AI-filter: only execute high-confidence opps
            approved = []
            for opp in opportunities:
                analysis = ai_analyzer.analyze_sports_opportunity(opp)
                opp.confidence = analysis.get("confidence", opp.confidence)
                action = analysis.get("action", "BET")
                if action in ("BET", "MONITOR"):
                    approved.append(opp)

            # Execute trades
            for opp in approved[:CONFIG["max_concurrent_bets"]]:
                trade_result = executor.execute_sports_trade(opp)
                if trade_result and trade_result.settled:
                    # Update opp's expected profit with actual
                    opp.expected_profit_usd = trade_result.pnl

            results.extend(approved)

        if mode in ("polymarket", "all") and CONFIG.get("polymarket_enabled"):
            poly_opps = poly_client.find_opportunities()
            for poly_opp in poly_opps[:3]:
                executor.execute_polymarket_trade(poly_opp)
            # Wrap poly opps in a light adapter for the terminal
            for po in poly_opps[:3]:
                results.append(_wrap_poly_opp(po))

        return results

    # Run live
    stop_event = threading.Event()
    try:
        terminal.run_live(scan_callback=do_scan, stop_event=stop_event)
    except KeyboardInterrupt:
        stop_event.set()

    # Final summary
    terminal.console.print()
    terminal.console.print("Session ended.", style="bold bright_green")
    terminal._print_stats_bar()


def _wrap_poly_opp(po):
    """Wrap a PolymarketOpportunity in a duck-type shim for the terminal."""
    from core.arbitrage_scanner import ArbitrageOpportunity
    return ArbitrageOpportunity(
        sport="POLY",
        event=po.question[:50],
        market=f"{po.side} @ {po.market_price:.2%}",
        bookmaker_buy="Polymarket",
        bookmaker_hedge="Polymarket",
        odds_buy=round(1 / max(0.01, po.market_price), 3),
        odds_hedge=round(1 / max(0.01, 1 - po.market_price), 3),
        delta_odds=round(po.edge, 4),
        edge_pct=po.edge,
        latency_indicator_ms=0,
        confidence=po.edge * 5,
        recommended_stake_usd=po.recommended_usdc,
        expected_profit_usd=po.expected_profit_usdc,
        arb_type="latency",
    )


def cmd_status(args):
    """Show current status."""
    wallet = Wallet()
    terminal = ClaudeBotTerminal(wallet=wallet)
    terminal.print_boot_sequence()
    terminal.print_status(wallet=wallet)


def cmd_history(args):
    """Show trade history."""
    wallet = Wallet()
    terminal = ClaudeBotTerminal(wallet=wallet)
    terminal.print_history(wallet=wallet, n=args.n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "history":
        cmd_history(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

"""pm_edges command line."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time
from pathlib import Path

from rich.console import Console

from .config import load_config
from .fair import DEVIG
from .httpclient import ApiError
from .ledger import Ledger
from .pipeline import GameSpec, analyze, collect
from .report import print_edges, print_markets, write_csv, write_html
from .store import Store

LEGAL = ("Read-only research tool; it places no bets. Check that Polymarket and sportsbook "
         "use are legal where you live.")


def _game_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("game")
    g.add_argument("--league", default="nfl")
    g.add_argument("--away", required=True, help="away team code, e.g. WAS")
    g.add_argument("--home", required=True, help="home team code, e.g. SF")
    g.add_argument("--date", required=True, type=date.fromisoformat,
                   help="local game date YYYY-MM-DD (the UTC next-day slug is also tried)")
    g.add_argument("--kickoff", type=lambda s: time.fromisoformat(s),
                   help="local kickoff HH:MM, to pick the UTC slug date first (optional)")
    g.add_argument("--tz", default="America/New_York", help="time zone of --date/--kickoff")
    o = p.add_argument_group("run")
    o.add_argument("--offline", action="store_true",
                   help="re-run from saved raw JSON in data/raw instead of the network")
    o.add_argument("--offline-run", help="raw run folder to replay (default: newest copy of each request)")
    o.add_argument("--data-dir", type=Path, default=Path("data"))
    o.add_argument("--min-liquidity", type=float, help="ignore markets below this liquidityNum (default 100)")
    o.add_argument("--no-clob", action="store_true", help="use Gamma bestBid/bestAsk only, skip order books")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pm_edges", description=__doc__, epilog=LEGAL)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("markets", help="list every Polymarket market for a game", epilog=LEGAL)
    _game_args(m)
    m.add_argument("--all", action="store_true", help="also show markets below the liquidity filter")

    tr = sub.add_parser("trends", help="historical situational angles for a game (NBA, NHL, NFL)", epilog=LEGAL)
    _game_args(tr)
    tr.add_argument("--spread", type=float, help="home spread, e.g. -4.5 if the home team is favoured by 4.5 "
                    "(NFL: read from the nflverse schedule when omitted; not used for hockey)")
    tr.add_argument("--total", type=float, help="game total (NFL: read from the schedule when omitted)")
    tr.add_argument("--home-ml", type=float, help="home moneyline, e.g. -150 (used for moneyline angles)")
    tr.add_argument("--away-ml", type=float, help="away moneyline, e.g. 130")
    tr.add_argument("--wind", type=float, help="forecast wind in mph (NFL; the schedule has no weather until after the game)")
    tr.add_argument("--temp", type=float, help="forecast temperature in F (NFL)")
    tr.add_argument("--angle", help='a TrendsCenter-style filter, e.g. "at_home == 1 and spread > -7 and game_num == 1"')
    tr.add_argument("--bet", choices=["under", "over", "ats", "ml"], default="under")
    tr.add_argument("--leave-one-out", action="store_true", help="show which conditions of --angle carry its record")

    lg = sub.add_parser("ledger", help="paper-trade ledger: closing-line value and results of flagged edges", epilog=LEGAL)
    lg.add_argument("--settle", action="store_true", help="look up Polymarket resolutions for finished games first")
    lg.add_argument("--csv", type=Path, help="also write every entry to this CSV file")
    lg.add_argument("--data-dir", type=Path, default=Path("data"))

    s = sub.add_parser("snapshot", help="store both sources in SQLite (run every 10-15 min)", epilog=LEGAL)
    _game_args(s)
    s.add_argument("--history", action="store_true", help="also pull CLOB price history")
    s.add_argument("--scope", choices=["full", "halves", "all"], help="periods to request odds for")
    s.add_argument("--bookmakers", help="Odds API bookmaker keys, comma-separated")

    e = sub.add_parser("edges", help="compare Polymarket asks with sportsbook fair values", epilog=LEGAL)
    _game_args(e)
    e.add_argument("--min-edge", type=float, help="flag threshold in probability (default 0.03 = 3 pts)")
    e.add_argument("--devig", choices=sorted(DEVIG), help="de-vig method (default power)")
    e.add_argument("--allow-model", action="store_true",
                   help="allow model-converted lines to be flagged (lower confidence)")
    e.add_argument("--confirm-sharp", action="store_true",
                   help="only flag when Pinnacle's own no-vig price also clears --min-edge")
    e.add_argument("--fee-rate", type=float,
                   help="override the Polymarket taker fee rate (fee/share = rate*p*(1-p))")
    e.add_argument("--bookmakers", help="Odds API bookmaker keys, comma-separated (default: 10 books "
                   "incl. Pinnacle; 10 books cost the same as 1 region)")
    e.add_argument("--regions", help="use Odds API regions instead of --bookmakers, e.g. us,eu")
    e.add_argument("--no-alternates", action="store_true", help="never request alternate lines")
    e.add_argument("--no-props", action="store_true", help="never request player props (saves Odds API credits)")
    e.add_argument("--scope", choices=["full", "halves", "all"],
                   help="periods to request odds for (default all; fewer = fewer credits)")
    e.add_argument("--no-history", action="store_true", help="skip CLOB price history (drift)")
    e.add_argument("--snapshot", action="store_true", help="also store this run as a snapshot")
    e.add_argument("--ledger", action="store_true", help="paper-trade the flagged edges: record them and refresh earlier entries "
                   "(run again before kickoff so the last scan stands in for the closing line)")
    e.add_argument("--max-stake", type=float, help="largest order the model considers, in dollars (default 50)")
    e.add_argument("--kelly", action="store_true", help="show a capped quarter-Kelly stake")
    e.add_argument("--bankroll", type=float, help="bankroll for --kelly")
    e.add_argument("--html", action="store_true", help="also write a self-contained HTML report")
    e.add_argument("--no-csv", action="store_true")
    e.add_argument("--quiet-unpriced", action="store_true", help="hide the no-fair-value table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    cfg = load_config(
        data_dir=args.data_dir,
        min_liquidity=getattr(args, "min_liquidity", None),
        min_edge=getattr(args, "min_edge", None),
        devig_method=getattr(args, "devig", None),
        fee_rate_override=getattr(args, "fee_rate", None),
        regions=getattr(args, "regions", None),
        scope=getattr(args, "scope", None),
        max_stake=getattr(args, "max_stake", None),
    )
    if getattr(args, "allow_model", False):
        cfg.allow_model = True
    if getattr(args, "confirm_sharp", False):
        cfg.confirm_sharp = True
    if getattr(args, "no_alternates", False):
        cfg.fetch_alternates = False
    if getattr(args, "no_props", False):
        cfg.include_props = False
    if getattr(args, "bookmakers", None):
        cfg.bookmakers = args.bookmakers
    elif getattr(args, "regions", None):
        cfg.bookmakers = None

    if args.cmd == "ledger":
        return _ledger(console, cfg, args)

    try:
        spec = GameSpec(args.league, args.away, args.home, args.date, args.kickoff, args.tz)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        return 2

    try:
        if args.cmd == "trends":
            from .trends import run as run_trends
            run_trends(console, spec.league, spec.away, spec.home, args.date, args.spread, args.total,
                       args.data_dir, args.angle, args.bet, args.leave_one_out,
                       args.home_ml, args.away_ml, args.wind, args.temp)
        elif args.cmd == "markets":
            run = collect(spec, cfg, offline=args.offline, offline_run=args.offline_run,
                          books=False, use_clob=not args.no_clob)
            _header(console, run)
            print_markets(console, run.markets, cfg, show_all=args.all)
        elif args.cmd == "snapshot":
            run = collect(spec, cfg, offline=args.offline, offline_run=args.offline_run,
                          use_clob=not args.no_clob, history=args.history)
            _header(console, run)
            if args.offline:
                console.print("[yellow]offline: snapshot is built from saved responses[/]")
            store = Store(cfg.db_path)
            try:
                sid = store.write(spec.game_id, run.fetched_at, run.book_rows, run.markets)
                n = len(store.snapshots(spec.game_id))
            finally:
                store.close()
            console.print(f"snapshot {sid} saved: {len(run.markets)} Polymarket markets, "
                          f"{len(run.book_rows)} sportsbook prices ({n} snapshots for {spec.game_id}; "
                          f"the first is treated as the open)")
        elif args.cmd == "edges":
            if args.kelly and not args.bankroll:
                console.print("[red]--kelly needs --bankroll[/]")
                return 2
            run = collect(spec, cfg, offline=args.offline, offline_run=args.offline_run,
                          use_clob=not args.no_clob, history=not args.no_history)
            _header(console, run)
            store = Store(cfg.db_path)
            try:
                if args.snapshot:
                    store.write(spec.game_id, run.fetched_at, run.book_rows, run.markets)
                result = analyze(run, store, bankroll=args.bankroll if args.kelly else None)
                if args.ledger:
                    added, refreshed = Ledger(store.conn).record(spec.game_id, result.edges, cfg, run.fetched_at)
                    console.print(f"ledger: {added} new paper trade(s), {refreshed} earlier one(s) refreshed")
            finally:
                store.close()
            print_edges(console, result.edges, len(result.checked), cfg, result.signal_notes,
                        show_unpriced=not args.quiet_unpriced, kelly=args.kelly)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            if not args.no_csv:
                path = write_csv(result.edges, cfg.reports_dir / f"{spec.game_id}-{stamp}.csv")
                console.print(f"CSV: {path}")
            if args.html:
                path = write_html(result.edges, len(result.checked),
                                  cfg.reports_dir / f"{spec.game_id}-{stamp}.html",
                                  f"{spec.away} @ {spec.home} {spec.date}", cfg, run.fetched_at)
                console.print(f"HTML: {path}")
    except ApiError as exc:
        console.print(f"[red]error:[/] {exc}")
        return 1
    return 0


def _ledger(console: Console, cfg, args) -> int:
    from datetime import timezone
    from rich.table import Table
    store = Store(cfg.db_path)
    try:
        led = Ledger(store.conn)
        if args.settle:
            n = led.settle_open(datetime.now(timezone.utc))
            console.print(f"settled {n} entr{'y' if n == 1 else 'ies'} from Polymarket resolutions")
        rows, s = led.rows(), led.summary()
        if not rows:
            console.print("The ledger is empty. Run `pm_edges edges ... --ledger` before kickoff to paper-trade the flagged edges.")
            return 0
        t = Table(title="Paper trades", title_justify="left")
        for c in ("game", "bet", "ask", "fair now", "clv", "status", "result", "pnl $"):
            t.add_column(c, justify="left" if c in ("game", "bet", "status") else "right")
        for r in rows:
            clv = (r["clv"] if r["clv"] is not None else (r["fair_last"] - r["ask"] if r["fair_last"] is not None else None))
            t.add_row(r["game_id"], f"{r['question'][:40]} / {r['outcome']}", f"{r['ask']:.3f}",
                      "-" if r["fair_last"] is None else f"{r['fair_last']:.3f}", "-" if clv is None else f"{clv * 100:+.1f}",
                      r["status"], "-" if r["result"] is None else f"{r['result']:g}", "-" if r["pnl"] is None else f"{r['pnl']:+.2f}")
        console.print(t)
        console.print(f"{s.entries} paper trades, {s.settled} settled. Closing-line value (points per share): "
                      + ("n/a" if s.mean_clv is None else f"{s.mean_clv * 100:+.2f} ± {((s.clv_se or 0) * 100):.2f}"
                         + (f" (t = {s.clv_t:.1f})" if s.clv_t is not None else ""))
                      + ("" if s.roi is None else f"; ROI {s.roi:+.1%} on ${s.staked:.0f} staked"))
        if s.settled < 30:
            console.print("[dim]Fewer than 30 settled trades: treat these numbers as anecdotes.[/]")
        if args.csv:
            import csv
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            console.print(f"CSV: {args.csv}")
    finally:
        store.close()
    return 0


def _header(console: Console, run) -> None:
    spec = run.spec
    console.print(f"[bold]{spec.away} @ {spec.home}[/] {spec.date}  "
                  f"slug: {run.base_slug or '-'}  data as of {run.fetched_at:%Y-%m-%d %H:%M:%S} UTC")
    if run.markets:
        console.print(f"Polymarket: {len(run.markets)} markets across "
                      f"{len({m.event_slug for m in run.markets})} event(s); "
                      f"{run.clob_priced} outcomes priced from the CLOB order book")
    if run.odds_calls:
        console.print(f"Odds API: {len(run.odds_calls)} call(s), {len(run.book_rows)} prices; "
                      f"credits remaining: {run.quota_remaining or 'unknown'}")
    for w in run.warnings[:15]:
        console.print(f"[yellow]warning:[/] {w}")
    if len(run.warnings) > 15:
        console.print(f"[yellow]... {len(run.warnings) - 15} more warnings[/]")
    console.print(f"[dim]raw JSON: {run.raw_dir}[/]")


if __name__ == "__main__":
    sys.exit(main())

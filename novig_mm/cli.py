"""novig_mm command line: check | record | report | quotes.  Read-only: nothing here can place an order."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import analyze, recorder
from .book import D, touch_from_market
from .client import NovigClient, NovigError, load_config
from .quoter import QuoteConfig, make_quotes

console = Console()


def cmd_check(a) -> int:
    cfg = load_config()
    console.print(f"config: {cfg}")
    c = NovigClient(cfg)
    now = int(time.time() * 1000)
    ev = c.get_public("/v3/public/catalog/events", {"league": a.league, "status": "OPEN_PREGAME", "limit": 3})
    console.print(f"public catalog OK: {len(ev['items'])} {a.league} pregame events, e.g. {ev['items'][0]['description'] if ev['items'] else '-'}")
    if cfg.key_id and cfg.pem:
        try:
            console.print(f"signed echo: {c.echo()}")
        except NovigError as e:
            console.print(f"[red]signed echo failed: {e}[/red]  (wrong host/key pair, clock skew > 30 s, or a bad PEM)")
            return 1
    else:
        console.print("no NOVIG_KEY_ID / NOVIG_PEM set: signed calls skipped (public data only)")
    return 0


def cmd_record(a) -> int:
    c = NovigClient(load_config())
    recorder.run(c, [x.strip().upper() for x in a.leagues.split(",")], a.types, a.interval, a.duration,
                 a.window_hours, Path(a.out))
    return 0


def cmd_report(a) -> int:
    paths = sorted(Path(a.dir).glob("touch_*.jsonl.gz"))
    if not paths:
        console.print(f"no touch_*.jsonl.gz in {a.dir}; run `novig_mm record` first")
        return 1
    rows = analyze.summarize(analyze.read_rows(paths))
    t = Table(title=f"Novig top of book ({len(paths)} file(s)); spreads in cents, qty = smaller side of the touch")
    cols = list(rows[0]) if rows else []
    for col in cols:
        t.add_column(col, justify="left" if col in ("league", "type", "line", "to_start") else "right")
    for r in rows:
        t.add_row(*[str(r[c]) for c in cols])
    console.print(t)
    return 0


def cmd_flow(a) -> int:
    from . import flow                                   # needs pandas/numpy
    m = flow.run(Path(a.touch_dir), Path(a.trades_dir), a.days)
    if m.empty:
        console.print("no joined trades: record touches first (`novig_mm record`) and download the public trade files "
                      "for the same days (data.novig.com/reporting/trade-data/<date>/trades.csv)")
        return 1
    s = flow.summarize(m, f"maker_{a.horizon}s")
    t = Table(title=f"Maker result vs price {a.horizon}s later, cents per contract (+ = makers earned); live credit not included")
    for col in s.columns:
        t.add_column(col, justify="left" if col in ("league", "type", "line", "phase") else "right")
    for _, r in s.iterrows():
        t.add_row(*[str(r[c]) for c in s.columns])
    console.print(t)
    return 0


def _load_fair(path: str) -> dict[str, tuple[Decimal, Decimal]]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["outcome_id"]] = (D(row["fair"]), D(row.get("sd") or "0.01"))
    return out


def cmd_quotes(a) -> int:
    c = NovigClient(load_config())
    cfg = QuoteConfig(half_spread=D(a.half_spread), size=a.size, pull_minutes=a.pull_minutes)
    fair_map = _load_fair(a.fair) if a.fair else {}
    now = int(time.time() * 1000)
    markets = c.markets(a.league.upper(), market_types=a.types, starts_before_ms=now + int(a.window_hours * 3600_000))
    t = Table(title="DRY RUN: quotes that would be posted (post-only buys). Nothing is sent.")
    for col in ("market", "type", "outcome", "fair", "quote", "qty", "vs touch (ticks)", "skipped"):
        t.add_column(col)
    out_rows = []
    for m in markets:
        touch = touch_from_market(m)
        if touch is None or m.get("status") != "OPEN":
            continue
        if a.fair_from_mid:                         # plumbing demo only: circular, not a real fair value
            mid = touch.mid(0)
            if mid is None:
                continue
            fair, sd = mid, D("0.01")
        else:
            hit = [(i, fair_map[o]) for i, o in enumerate(touch.outcome_ids) if o in fair_map]
            if not hit:
                continue
            i, (f, sd) = hit[0]
            fair = f if i == 0 else 1 - f
        qs = make_quotes(touch, fair, sd, 0, (m["startsTs"] - now) / 60000, cfg)
        names = {o["outcomeId"]: o["name"] for o in m["outcomes"]}
        if qs.skipped:
            t.add_row(m["description"][:38], m["marketType"], "-", f"{fair:.3f}", "-", "-", "-", qs.skipped)
        for q in qs.quotes:
            t.add_row(m["description"][:38], m["marketType"], names[q.outcome_id][:18], f"{fair:.3f}", str(q.price), str(q.qty),
                      str(q.ticks_from_best), "")
        out_rows.append({"ts": now, "market": m["marketId"], "fair_a": str(fair), "skipped": qs.skipped,
                         "quotes": [{"outcome": q.outcome_id, "price": str(q.price), "qty": q.qty, "tif": q.tif,
                                     "ticks_from_best": q.ticks_from_best} for q in qs.quotes]})
    console.print(t)
    if out_rows:
        p = Path(a.out) / f"dryrun_quotes_{time.strftime('%Y-%m-%d', time.gmtime(now / 1000))}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        console.print(f"logged {len(out_rows)} markets to {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="novig_mm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="verify connectivity (public) and, if keys are set, the signature via /v3/echo")
    p.add_argument("--league", default="NBA")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("record", help="poll the public catalog and store top-of-book snapshots")
    p.add_argument("--leagues", default="NBA")
    p.add_argument("--types", default=recorder.DEFAULT_TYPES, help="comma-separated market types (default MONEY,SPREAD,TOTAL)")
    p.add_argument("--interval", type=float, default=30.0, help="seconds between polls (CDN caches ~15 s)")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds (default: run until Ctrl-C)")
    p.add_argument("--window-hours", type=float, default=24.0, help="only events starting within this many hours")
    p.add_argument("--out", default="data/novig")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("report", help="spread and depth summary of recorded touches")
    p.add_argument("--dir", default="data/novig")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("flow", help="join recorded touches with public trade files: maker result by game phase")
    p.add_argument("--touch-dir", default="data/novig")
    p.add_argument("--trades-dir", required=True, help="folder holding <YYYY-MM-DD>/trades.csv from data.novig.com")
    p.add_argument("--days", type=int, default=None, help="use only the newest N trade files")
    p.add_argument("--horizon", type=int, choices=[60, 300], default=300, help="seconds after the trade")
    p.set_defaults(fn=cmd_flow)

    p = sub.add_parser("quotes", help="DRY RUN: print the quotes the engine would post")
    p.add_argument("--league", default="NBA")
    p.add_argument("--types", default="MONEY,SPREAD,TOTAL")
    p.add_argument("--fair", help="CSV with columns outcome_id,fair[,sd]  (fair = probability of that outcome)")
    p.add_argument("--fair-from-mid", action="store_true", help="plumbing demo only: use Novig's own mid as fair (circular)")
    p.add_argument("--half-spread", default="0.015")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--pull-minutes", type=float, default=45.0)
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--out", default="data/novig")
    p.set_defaults(fn=cmd_quotes)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except NovigError as e:
        console.print(f"[red]Novig error: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Terminal tables, CSV and a self-contained HTML report."""
from __future__ import annotations

import csv
import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape as rx
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .edges import EdgeRow
from .models import MARKET_TYPES, PMMarket
from .polymarket import is_stale


def _p(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _pts(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:+.1f}"


def _n(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f}"


# --------------------------------------------------------------------------
# markets
# --------------------------------------------------------------------------


def print_markets(console: Console, markets: list[PMMarket], cfg: Config, show_all: bool = False) -> int:
    active = [m for m in markets if m.active and not m.closed]
    shown = active if show_all else [m for m in active if m.liquidity >= cfg.min_liquidity]
    hidden = len(active) - len(shown)
    groups: dict[str, list[PMMarket]] = defaultdict(list)
    for m in shown:
        groups[m.market_type].append(m)
    for mt in MARKET_TYPES:
        ms = groups.get(mt)
        if not ms:
            continue
        t = Table(title=f"{mt} ({len(ms)})", title_justify="left", show_lines=False)
        for col in ("market", "outcome", "bid", "ask", "mid", "last", "ask size", "liquidity", "src", "flags"):
            t.add_column(col, justify="right" if col not in ("market", "outcome", "flags", "src") else "left")
        for m in sorted(ms, key=lambda m: (m.event_slug, m.question)):
            flags = ", ".join(is_stale(m, cfg) + ([m.parse_note] if m.parse_note else []))
            for i, o in enumerate(m.outcomes):
                t.add_row(
                    rx(m.question) if i == 0 else "", o.name, _p(o.bid), _p(o.ask),
                    _p(m.mid) if i == 0 else "", _p(m.last_trade_price) if i == 0 else "",
                    _n(o.ask_size), _n(m.liquidity) if i == 0 else "", o.price_source,
                    ("[yellow]" + rx(flags) + "[/]") if (flags and i == 0) else "")
        console.print(t)
    console.print(f"{len(shown)} active markets shown"
                  + (f", {hidden} hidden below liquidity {cfg.min_liquidity:g} (use --all)" if hidden else "")
                  + f", {len(markets) - len(active)} closed/inactive skipped.")
    return len(shown)


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

EDGE_COLUMNS = [
    "market", "outcome", "selection", "ask", "fee", "fair", "edge_pts", "flagged", "why_not",
    "size_at_ask", "size_with_edge", "fair_source", "books", "pinnacle_fair", "fair_note",
    "stale", "signals", "kelly_stake", "fee_rate", "price_source", "liquidity", "slug",
]


def edge_record(r: EdgeRow) -> dict:
    return {
        "market": r.market.question,
        "outcome": r.outcome.name,
        "selection": r.outcome.selection.label() if r.outcome.selection else "",
        "ask": r.ask,
        "fee": None if r.fee is None else round(r.fee, 4),
        "fair": None if r.fair.prob is None else round(r.fair.prob, 4),
        "edge_pts": None if r.edge is None else round(r.edge * 100, 2),
        "flagged": r.flagged,
        "why_not": r.why_not,
        "size_at_ask": r.size_at_ask,
        "size_with_edge": r.size_with_edge,
        "fair_source": r.fair.source,
        "books": r.fair.books_label,
        "pinnacle_fair": r.fair.book_prob("pinnacle"),
        "fair_note": r.fair.reason,
        "stale": "; ".join(r.stale),
        "signals": " | ".join(r.signals),
        "kelly_stake": r.kelly_stake,
        "fee_rate": r.fee_note,
        "price_source": r.outcome.price_source,
        "liquidity": r.market.liquidity,
        "slug": r.market.slug,
    }


def summary_line(rows: list[EdgeRow], n_checked: int) -> tuple[int, str]:
    flagged_markets = {r.market.market_id for r in rows if r.flagged}
    k = len(flagged_markets)
    return k, f"{n_checked} markets checked, {k} flagged"


def print_edges(console: Console, rows: list[EdgeRow], n_checked: int, cfg: Config,
                signal_notes: list[str], show_unpriced: bool = True, kelly: bool = False) -> int:
    priced = [r for r in rows if r.fair.prob is not None]
    priced.sort(key=lambda r: (not r.flagged, -(r.edge if r.edge is not None else -9)))
    t = Table(title=f"Polymarket vs sportsbook fair (min edge {cfg.min_edge * 100:.1f} pts, "
                    f"de-vig {cfg.devig_method})", title_justify="left")
    cols = ["", "market", "side", "ask", "fee", "fair", "edge", "size@ask", "size w/ edge",
            "fair source", "books", "signals / notes"]
    if kelly:
        cols.insert(-1, "¼-Kelly $")
    for c in cols:
        t.add_column(c, justify="right" if c in ("ask", "fee", "fair", "edge", "size@ask",
                                                 "size w/ edge", "¼-Kelly $") else "left",
                     overflow="fold")
    for r in priced:
        mark = "[bold green]EDGE[/]" if r.flagged else ""
        edge_txt = _pts(r.edge)
        if r.flagged:
            edge_txt = f"[bold green]{edge_txt}[/]"
        elif r.edge is not None and r.edge > 0:
            edge_txt = f"[green]{edge_txt}[/]"
        src = r.fair.source + ("" if r.fair.source != "model" else " [yellow](low conf.)[/]")
        notes = [rx(x) for x in r.signals]
        if r.stale:
            notes.insert(0, "[yellow]stale: " + "; ".join(r.stale) + "[/]")
        if r.why_not and not r.why_not.startswith("edge below"):
            notes.insert(0, f"[dim]{rx(r.why_not)}[/]")
        if r.fair.reason:
            notes.append(f"[dim]{rx(r.fair.reason)}[/]")
        row = [mark, rx(r.market.question), r.outcome.selection.label() if r.outcome.selection else r.outcome.name,
               _p(r.ask), _p(r.fee, 4), _pct(r.fair.prob), edge_txt, _n(r.size_at_ask),
               _n(r.size_with_edge), src, r.fair.books_label, "\n".join(notes)]
        if kelly:
            row.insert(-1, "-" if r.kelly_stake is None else f"{r.kelly_stake:,.2f}")
        t.add_row(*row)
    console.print(t)

    if show_unpriced:
        unpriced = [r for r in rows if r.fair.prob is None]
        if unpriced:
            u = Table(title=f"No fair value ({len(unpriced)} outcomes)", title_justify="left")
            u.add_column("market", overflow="fold")
            u.add_column("outcome")
            u.add_column("ask", justify="right")
            u.add_column("reason", overflow="fold")
            for r in sorted(unpriced, key=lambda r: (r.market.market_type, r.market.question, r.outcome.index)):
                u.add_row(rx(r.market.question), rx(r.outcome.name), _p(r.ask), rx(r.fair.reason or r.why_not))
            console.print(u)

    for note in signal_notes:
        console.print(f"[dim]signals: {note}[/]")
    fee_notes = sorted({r.fee_note for r in rows})
    console.print(f"[dim]taker fee rate(s) used: {', '.join(fee_notes) or '-'}; "
                  f"fee per share = rate x p x (1-p)[/]")

    k, line = summary_line(rows, n_checked)
    if k == 0:
        console.print(Panel(f"{line}\n[bold]No edge found.[/]", style="bold yellow", expand=False))
    else:
        console.print(Panel(f"{line}\nFlagged = fair - ask - fee >= {cfg.min_edge * 100:.1f} pts "
                            f"against the sportsbook consensus. An estimate, not a sure thing.",
                            style="bold green", expand=False))
    return k


def write_csv(rows: list[EdgeRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(edge_record(r))
    return path


def write_html(rows: list[EdgeRow], n_checked: int, path: Path, title: str, cfg: Config,
               generated: datetime) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    k, line = summary_line(rows, n_checked)
    cols = ["market", "selection", "ask", "fee", "fair", "edge_pts", "size_at_ask", "fair_source",
            "books", "why_not", "stale", "signals"]
    body = []
    for r in sorted(rows, key=lambda r: (not r.flagged, -(r.edge if r.edge is not None else -9))):
        rec = edge_record(r)
        cls = "flag" if r.flagged else ("none" if r.fair.prob is None else "")
        cells = "".join(f"<td>{html.escape('' if rec[c] is None else str(rec[c]))}</td>" for c in cols)
        body.append(f'<tr class="{cls}">{cells}</tr>')
    verdict = "No edge found." if k == 0 else f"{k} market(s) flagged."
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --flag:#e6f6ea; --none:#f6f6f6; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141414; --fg:#eee; --muted:#999; --line:#333; --flag:#183d24; --none:#1d1d1d; }}
}}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui,sans-serif; margin:16px; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }}
tr.flag {{ background:var(--flag); }} tr.none {{ background:var(--none); color:var(--muted); }}
.wrap {{ overflow-x:auto; }} .summary {{ font-size:18px; font-weight:600; margin:12px 0; }}
.muted {{ color:var(--muted); }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="muted">Generated {generated:%Y-%m-%d %H:%M UTC}. Min edge {cfg.min_edge * 100:.1f} pts,
de-vig {cfg.devig_method}. Edge = fair probability - Polymarket ask - taker fee. Fair values are
estimates from sportsbook prices; nothing here is certain.</p>
<p class="summary">{html.escape(line)}. {verdict}</p>
<div class="wrap"><table><thead><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
</body></html>
"""
    path.write_text(doc)
    return path

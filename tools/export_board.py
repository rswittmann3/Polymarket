"""Export the latest analysis of one game as JSON datasets for the edge board.

    python tools/export_board.py --away WAS --home SF --date 2026-10-19 --kickoff 20:15 \
        [--run 20261019T231508Z] [--scope full] [--out data/board]

Reads saved raw responses only (no network, no Odds API credits).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, time
from pathlib import Path

from pm_edges.config import load_config
from pm_edges.pipeline import GameSpec, analyze, collect
from pm_edges.store import Store


def reason_bucket(text: str) -> str:
    t = text or ""
    for needle, label in (
        ("no sportsbook equivalent", "Polymarket-only market (no sportsbook twin)"),
        ("one-sided", "Books quote only one side"),
        ("no book lists this market", "No book lists it"),
        ("full-game only", "Half/quarter line not quoted by books"),
        ("no line model", "Prop line not quoted by books"),
        ("not requested", "Period not requested (scope)"),
        ("no two-way", "No two-way price at any book"),
        ("liquidity", "Below liquidity filter"),
    ):
        if needle in t:
            return label
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--date", required=True, type=date.fromisoformat)
    ap.add_argument("--kickoff", type=time.fromisoformat)
    ap.add_argument("--run", help="raw run folder to analyse (default: newest)")
    ap.add_argument("--scope", default="full")
    ap.add_argument("--confirm-sharp", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/board"))
    args = ap.parse_args()

    cfg = load_config(data_dir=args.data_dir, scope=args.scope)
    cfg.confirm_sharp = args.confirm_sharp
    spec = GameSpec("nfl", args.away, args.home, args.date, args.kickoff)
    run = collect(spec, cfg, offline=True, offline_run=args.run, history=True)
    store = Store(cfg.db_path)
    try:
        a = analyze(run, store)
        n_snaps = len(store.snapshots(spec.game_id))
    finally:
        store.close()

    edges = []
    gaps = Counter()
    for e in a.edges:
        if e.fair.prob is None:
            gaps[reason_bucket(e.fair.reason or e.why_not)] += 1
            continue
        sel = e.outcome.selection
        if e.flagged:
            status = "flagged"
        elif e.why_not.startswith("model-converted"):
            status = "model"
        elif e.why_not.startswith("Pinnacle does not"):
            status = "unconfirmed"
        elif e.edge is not None and e.edge > 0:
            status = "below"
        else:
            status = "none"
        edges.append({
            "id": f"{e.market.market_id}-{e.outcome.index}",
            "market": e.market.question,
            "selection": sel.label() if sel else e.outcome.name,
            "kind": e.market.market_type,
            "period": sel.period if sel else "full",
            "ask": e.ask, "fee": None if e.fee is None else round(e.fee, 4),
            "fair": None if e.fair.prob is None else round(e.fair.prob, 4),
            "edge_pts": None if e.edge is None else round(e.edge * 100, 2),
            "pin_edge_pts": None if e.pin_edge is None else round(e.pin_edge * 100, 2),
            "fair_source": e.fair.source,
            "n_books": len(e.fair.books),
            "books": e.fair.books_label,
            "size_at_ask": e.size_at_ask, "size_with_edge": e.size_with_edge,
            "status": status, "why_not": e.why_not,
            "stale": "; ".join(e.stale),
            "liquidity": round(e.market.liquidity),
            "signals": " | ".join(e.signals),
        })

    seen, signals = set(), []
    for s in a.signals:
        if s.text in seen:
            continue
        seen.add(s.text)
        signals.append({"id": f"s{len(signals)}", "kind": s.kind, "text": s.text,
                        "books": ", ".join(s.books), "value": None if s.value is None else round(s.value, 4)})

    flagged = [e for e in a.edges if e.flagged]
    fees = sorted({e.fee_note for e in a.edges})
    run_row = {
        "id": "run",
        "as_of": run.fetched_at.isoformat(),
        "game": f"{spec.away} @ {spec.home}",
        "game_date": spec.date.isoformat(),
        "snapshots": n_snaps,
        "markets_checked": len(a.checked),
        "outcomes_priced": len(edges),
        "outcomes_unpriced": sum(gaps.values()),
        "flagged": len({e.market.market_id for e in flagged}),
        "credits_remaining": None if run.quota_remaining is None else int(run.quota_remaining),
        "min_edge_pts": cfg.min_edge * 100,
        "devig": cfg.devig_method,
        "fee_rates": ", ".join(fees),
        "confirm_sharp": cfg.confirm_sharp,
        "signal_notes": " | ".join(a.signal_notes),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("edges", edges), ("signals", signals), ("run", [run_row]),
                       ("gaps", [{"id": f"g{i}", "reason": r, "outcomes": n} for i, (r, n) in enumerate(gaps.most_common())])):
        (args.out / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")))
    print({k: len(v) for k, v in (("edges", edges), ("signals", signals), ("gaps", gaps))}, run_row["as_of"])


if __name__ == "__main__":
    main()

"""Scan the upcoming slate for Polymarket edges and write one file the dashboard can show.

For each game starting within --hours, fetch the Polymarket markets and the sportsbook prices (3 Odds API credits per
game with --no-props), compare them, and keep every outcome whose edge is positive. An edge is "material" when it is
flagged: at least the minimum edge after the taker fee, at least two books agree, Pinnacle's own price confirms it,
the market is liquid, and the price is not stale. The file also says how many markets were checked, so "nothing found"
is a result the page can show.

    python tools/scan_slate.py --league nba --hours 36
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pm_edges import slate as S
from pm_edges.config import load_config
from pm_edges.httpclient import ApiError
from pm_edges.pipeline import GameSpec, analyze, collect
from pm_edges.store import Store


def classify(e) -> str:
    if e.flagged:
        return "flagged"
    if e.fair.prob is None or e.edge is None:
        return "unpriced"
    if e.why_not.startswith("model-converted"):
        return "model"
    if e.why_not.startswith("Pinnacle does not"):
        return "unconfirmed"
    return "below" if e.edge > 0 else "none"


def confidence(e) -> str:
    """High: Pinnacle alone clears the bar and at least three books agree. Medium: flagged. Low: anything else."""
    if not e.flagged:
        return "low"
    return "high" if (e.pin_edge or 0) >= 0.03 and len(e.fair.books) >= 3 else "medium"


def edge_rows(game: dict, analysis, cfg) -> list[dict]:
    out = []
    for e in analysis.edges:
        status = classify(e)
        if status in ("unpriced", "none") or (e.edge or 0) < 0.01:
            continue
        sel = e.outcome.selection
        out.append({
            "id": f"{game['league']}|{game['date']}|{game['away']}|{game['home']}|{e.market.market_id}-{e.outcome.index}",
            "league": game["league"].upper(), "game": f"{game['away']} @ {game['home']}", "date": game["date"], "kickoff": game.get("kickoff"),
            "market": e.market.question, "bet": sel.label() if sel else e.outcome.name, "kind": e.market.market_type,
            "ask": e.ask, "fee": e.fee, "fair": e.fair.prob, "edge_pts": 100 * e.edge,
            "pin_edge_pts": None if e.pin_edge is None else 100 * e.pin_edge, "books": len(e.fair.books), "book_names": e.fair.books_label,
            "dollars_at_ask": None if e.size_at_ask is None else e.size_at_ask * e.ask,
            "dollars_with_edge": None if e.size_with_edge is None else min(e.size_with_edge * e.ask, cfg.max_stake),
            "liquidity": e.market.liquidity, "status": status, "confidence": confidence(e),
            "why_not": e.why_not, "signals": " | ".join(e.signals),
        })
    return out


def scan(league: str, hours: float, data_dir: Path, now: datetime | None = None) -> dict:
    warnings.filterwarnings("ignore")
    now = now or datetime.now(timezone.utc)
    cfg = load_config(data_dir=data_dir, scope="full")
    cfg.include_props, cfg.fetch_alternates, cfg.confirm_sharp = False, False, True
    games = [g for g in S.slate(league, now.date(), 3 if league != "nfl" else 8)
             if g.get("kickoff") is None or now < datetime.fromisoformat(g["kickoff"]) < now + timedelta(hours=hours)]
    store = Store(cfg.db_path)
    rows, summary, errors = [], [], []
    try:
        for g in games:
            try:
                spec = GameSpec(league, g["away"], g["home"], date.fromisoformat(g["date"]))
                run = collect(spec, cfg, history=False)
                a = analyze(run, store)
            except ApiError as exc:
                errors.append({"game": f"{g['away']} @ {g['home']}", "error": str(exc)[:160]})
                continue
            er = edge_rows(g, a, cfg)
            rows += er
            summary.append({"game": f"{g['away']} @ {g['home']}", "markets_checked": len(a.checked),
                            "outcomes_priced": sum(1 for e in a.edges if e.fair.prob is not None),
                            "flagged": sum(1 for r in er if r["status"] == "flagged"), "credits_left": run.quota_remaining})
    finally:
        store.close()
    return {"scanned_at": now.isoformat(timespec="seconds"), "league": league, "hours": hours, "min_edge_pts": 100 * cfg.min_edge,
            "max_stake": cfg.max_stake, "games": summary, "errors": errors, "edges": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", default="nba", choices=["nba", "nfl"])
    ap.add_argument("--hours", type=float, default=36)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/scan"))
    a = ap.parse_args()
    res = scan(a.league, a.hours, a.data_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.league}.json").write_text(json.dumps(res, indent=1))
    flagged_now = [r for r in res["edges"] if r["status"] == "flagged"]
    with (a.out / "history.jsonl").open("a") as f:
        f.write(json.dumps({"scanned_at": res["scanned_at"], "league": a.league, "games": len(res["games"]),
                            "markets": sum(g["markets_checked"] for g in res["games"]), "flagged": len(flagged_now),
                            "best_edge_pts": max((r["edge_pts"] for r in res["edges"]), default=None),
                            "best_flagged_pts": max((r["edge_pts"] for r in flagged_now), default=None)}) + "\n")
    flagged = [r for r in res["edges"] if r["status"] == "flagged"]
    print(f"{len(res['games'])} games, {sum(g['markets_checked'] for g in res['games'])} markets, {len(flagged)} flagged, "
          f"{len(res['edges'])} positive-edge outcomes; credits left {res['games'][-1]['credits_left'] if res['games'] else '?'}")
    for r in sorted(flagged, key=lambda r: -r["edge_pts"])[:10]:
        print(f"  {r['game']}  {r['bet']}  ask {r['ask']:.3f}  edge {r['edge_pts']:+.1f}  pin {r['pin_edge_pts']:+.1f}  books {r['books']}")

"""Score frozen NBA line-move predictions against the lines that actually closed.

A frozen file (snapshots/frozen/nba_*.json) holds, for each game, the opening line, the line at prediction time,
and the model's predicted closing total and spread (from opening-line inputs only). Closing lines come from
data/lag/timeline.jsonl: the last Pinnacle row before tip, and the last ESPN/DraftKings row.

Two questions, each for totals and spreads:
  * from open: did the line move the way the model said?      sign(pred - open)  vs  sign(close - open)
  * from the line at prediction time (the tradable question):  sign(pred - then)  vs  sign(close - then)
CLV is the signed move in points. Compare against zero: a coin-flip model has none.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA = Path("data/lag/timeline.jsonl")


def closing_lines(path: Path = DATA) -> dict[tuple, dict]:
    """{(away, home): {"pin_total", "pin_spread", "dk_total", "dk_spread", "tip"}} from the last row before tip."""
    out: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if '"src": "pinnacle"' not in line and '"src": "espn_dk"' not in line:
            continue
        r = json.loads(line)
        tip = r.get("commence") or r.get("kickoff")
        if not tip or r["ts"] >= tip.replace("Z", "+00:00"):
            continue
        key = (r["away"], r["home"])
        d = out.setdefault(key, {"tip": tip})
        if r["src"] == "pinnacle":
            tot = next((o["point"] for o in r.get("totals", []) if o["name"] == "Over"), None)
            spr = next((o["point"] for o in r.get("spreads", []) if o["name"] and _is_home(o["name"], r["home"])), None)
            if tot is not None:
                d["pin_total"] = tot
            if spr is not None:
                d["pin_spread"] = spr
        else:
            d["dk_total"], d["dk_spread"] = r.get("total"), r.get("home_spread")
    return out


def _is_home(name: str, home: str) -> bool:
    from pm_edges.teams import league
    return league("nba").team_from_text(name) == home


def score(frozen: dict, closes: dict[tuple, dict], book: str = "pin", now: datetime | None = None) -> dict:
    """Only games that have tipped are scored: before that the "last row" is not a close."""
    now = now or datetime.now(timezone.utc)
    rows = {"total": [], "spread": []}
    for g in frozen["games"]:
        c = closes.get((g["away"], g["home"]))
        if not c or datetime.fromisoformat(c["tip"].replace("Z", "+00:00")) > now:
            continue
        for kind, open_k, now_k, pred_k, close_k in (("total", "total_open", "total", "pred_close_total", f"{book}_total"),
                                                     ("spread", "home_spread_open", "home_spread", "pred_close_spread", f"{book}_spread")):
            vals = [g.get(open_k), g.get(now_k), g.get(pred_k), c.get(close_k)]
            if any(v is None for v in vals):
                continue
            o, n, p, cl = vals
            rows[kind].append({"game": f"{g['away']}@{g['home']}", "open": o, "then": n, "pred": p, "close": cl})
    out = {}
    for kind, rs in rows.items():
        if not rs:
            continue
        stat = {"games": len(rs)}
        for label, base in (("from_open", "open"), ("from_then", "then")):
            pred = np.array([r["pred"] - r[base] for r in rs])
            act = np.array([r["close"] - r[base] for r in rs])
            clv = np.sign(pred) * act
            moved = act != 0
            stat[label] = {"mean_clv_points": float(clv.mean()), "se": float(clv.std(ddof=1) / math.sqrt(len(clv))) if len(clv) > 1 else None,
                           "direction_right": int(((np.sign(pred) == np.sign(act)) & moved).sum()), "moved": int(moved.sum()),
                           "corr": float(np.corrcoef(pred, act)[0, 1]) if len(rs) > 2 and pred.std() > 0 and act.std() > 0 else None}
        stat["detail"] = rs
        out[kind] = stat
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frozen", type=Path, nargs="?", help="a snapshots/frozen/nba_*.json file (default: all, newest first)")
    ap.add_argument("--book", choices=["pin", "dk"], default="pin")
    a = ap.parse_args()
    closes = closing_lines()
    for path in ([a.frozen] if a.frozen else sorted(Path("snapshots/frozen").glob("nba_*.json"), reverse=True)):
        res = score(json.loads(path.read_text()), closes, a.book)
        print(path.name, json.dumps({k: {x: y for x, y in v.items() if x != "detail"} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

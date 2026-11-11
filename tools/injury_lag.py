"""How fast does Novig's NBA spread react after an injury report changes? (Event study over the logged data.)

Needs two logs, both written by the loggers started with the daily routine:
  * data/lag/timeline.jsonl            injury rows from tools/line_poller.py ("src": "injury": status, athlete, team, ESPN `date`)
  * data/novig/touch_*.jsonl.gz, meta_*  Novig top of book from `python -m novig_mm record`

For every game we rebuild Novig's *fair home spread* through time: for each spread market take the home outcome's mid price, then
interpolate the line at which the home side covers 50%. For each injury report that lands on a team playing within `--horizon` hours we
then ask: of the eventual move from just before the report to the last pre-tip quote, how much had happened after 5, 15, 30, 60 minutes?
A lag shows up as a fraction well below 1 after a few minutes on games that move a lot. No lag means the market is already efficient to injury news.

    python tools/injury_lag.py                 # prints the table
    python tools/injury_lag.py --min-move 1.0
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

NOVIG = Path("data/novig")
TIMELINE = Path("data/lag/timeline.jsonl")
STATUS_RANK = {"Out": 3, "Doubtful": 2, "Questionable": 1, "Day-To-Day": 1}


def _read_gz(paths):
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def load_novig(d: Path = NOVIG) -> tuple[pd.DataFrame, dict, dict]:
    """(touch rows for SPREAD markets, {event: (away, home, start_ms)}, {market: {"event", "n", "line"}}) for NBA."""
    meta_rows = list(_read_gz(sorted(d.glob("meta_*.jsonl.gz"))))
    events = {r["e"]: r for r in meta_rows if r.get("k") == "ev" and r.get("lg") == "NBA"}
    teams: dict[str, tuple[str, str]] = {}
    for r in meta_rows:
        if r.get("k") == "mk" and r["e"] in events and r.get("mt") == "TOTAL":
            m = re.match(r"(\S+) @ (\S+) t", r["d"])
            if m:
                teams[r["e"]] = (m.group(1), m.group(2))
    mk = {}
    for r in meta_rows:
        if r.get("k") == "mk" and r["e"] in teams and r.get("mt") == "SPREAD":
            away, home = teams[r["e"]]
            home_i = next((i for i, n in enumerate(r["n"]) if n.split()[0] == home), None)
            if home_i is None:
                continue
            line = float(r["n"][home_i].split()[-1])
            mk[r["m"]] = {"event": r["e"], "home_i": home_i, "line": line}
    ev = {e: (*teams[e], events[e]["st"]) for e in teams}
    rows = [r for r in _read_gz(sorted(d.glob("touch_*.jsonl.gz"))) if r.get("m") in mk and r.get("lg") == "NBA"]
    df = pd.DataFrame(rows)
    return df, ev, mk


def fair_line_series(df: pd.DataFrame, mk: dict, event: str, times_ms: list[int]) -> list[float | None]:
    """Fair home spread (line where the home side covers 50%) at each time, from the latest touch of every spread market."""
    sub = df[df.m.map(lambda m: mk[m]["event"] == event)].sort_values("ts")
    out = []
    for t in times_ms:
        pts = []
        for m, g in sub[sub.ts <= t].groupby("m"):
            r = g.iloc[-1]
            b = r["b"]
            if b[0] is None or b[1] is None:
                continue
            i = mk[m]["home_i"]
            p = (float(b[i]) + 1 - float(b[1 - i])) / 2
            pts.append((mk[m]["line"], p))
        pts = sorted(p for p in pts if 0.05 < p[1] < 0.95)
        if len(pts) < 2:
            out.append(None)
            continue
        xs, ps = zip(*pts)
        ps = np.maximum.accumulate(np.array(ps))                 # cover probability must rise with the home line
        out.append(float(np.interp(0.5, ps, xs)) if ps[0] <= 0.5 <= ps[-1] else None)
    return out


def event_study(df, ev, mk, injuries: list[dict], horizon_h: float = 24, minutes=(5, 15, 30, 60), min_move: float = 1.0) -> pd.DataFrame:
    rows = []
    for inj in injuries:
        if inj.get("baseline") or inj.get("status") not in STATUS_RANK:
            continue
        t0 = datetime.fromisoformat(inj["date"].replace("Z", "+00:00")).timestamp() * 1000
        for e, (away, home, st) in ev.items():
            if inj["team"] not in (away, home) or not (t0 < st <= t0 + horizon_h * 3600e3):
                continue
            times = [t0 - 600e3] + [t0 + m * 60e3 for m in minutes] + [st - 60e3]
            line = fair_line_series(df, mk, e, [int(t) for t in times])
            if line[0] is None or line[-1] is None:
                continue
            total = line[-1] - line[0]
            row = {"game": f"{away}@{home}", "team": inj["team"], "player": inj.get("name"), "status": inj["status"], "reported": inj["date"],
                   "total_move": round(total, 2)}
            for m, v in zip(minutes, line[1:-1]):
                row[f"done_{m}m"] = None if v is None or abs(total) < min_move else round((v - line[0]) / total, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--novig-dir", type=Path, default=NOVIG)
    ap.add_argument("--timeline", type=Path, default=TIMELINE)
    ap.add_argument("--horizon", type=float, default=24.0, help="hours from the report to tip")
    ap.add_argument("--min-move", type=float, default=1.0, help="only measure games whose spread moved at least this much")
    a = ap.parse_args()
    if not list(a.novig_dir.glob("touch_*.jsonl.gz")) or not a.timeline.exists():
        raise SystemExit("need data/novig/touch_*.jsonl.gz and data/lag/timeline.jsonl: run the loggers first")
    df, ev, mk = load_novig(a.novig_dir)
    injuries = []
    for line in a.timeline.read_text().splitlines():
        if '"src": "injury"' in line:
            r = json.loads(line)
            if r.get("league") == "nba":
                injuries.append(r)
    res = event_study(df, ev, mk, injuries, a.horizon, min_move=a.min_move)
    if res.empty:
        print("no injury reports with Novig data on both sides yet")
        return
    pd.set_option("display.width", 200)
    print(res.sort_values("reported").to_string(index=False))
    big = res[res.total_move.abs() >= a.min_move]
    print(f"\n{len(res)} reports matched to a game; {len(big)} on games that moved >= {a.min_move} pts. Share of the move completed after:")
    for m in (5, 15, 30, 60):
        v = big[f"done_{m}m"].dropna()
        if len(v):
            print(f"  {m:>2} min: median {v.median():.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()

"""Freeze the NBA player-prop model's view of tonight's Novig prop markets (read-only; nothing is ever ordered).

For every prop market with a quote: model mean and P(over) (negative binomial, form + new teammate absences), Novig's bid/ask on both sides,
and the model's edge versus the price you would actually pay (the ask = 1 - opposite bid). Written to snapshots/props_frozen/<UTC date>.json.gz (each run appends one freeze).
Score with tools/score_props.py after the games.

    python tools/freeze_props.py                  # all upcoming NBA props in data/props
    python tools/freeze_props.py --top 25         # also print the largest edges
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pm_edges import nba_props as NP  # noqa: E402

PROPS, TIMELINE, OUT = Path("data/props"), Path("data/lag/timeline.jsonl"), Path("snapshots/props_frozen")


def _rows(paths):
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def load_quotes(now_ms: int, d: Path = PROPS):
    """Latest touch per upcoming prop market + meta: list of dicts."""
    meta = {}
    for r in _rows(sorted(d.glob("meta_*.jsonl.gz"))):
        if r.get("k") == "mk":
            meta[r["m"]] = r
    last = {}
    for r in _rows(sorted(d.glob("touch_*.jsonl.gz"))):
        if r["m"] in meta and r["mt"] in NP.MARKET_STAT and r["st"] > now_ms and (r["m"] not in last or r["ts"] >= last[r["m"]]["ts"]):
            last[r["m"]] = r
    out = []
    for m, t in last.items():
        mk = meta[m]
        names = mk["n"]
        if len(names) != 2 or not names[0].startswith("Over"):
            continue
        player = mk["d"].rsplit(" ", 2)[0] if mk.get("d") else None   # "Kevin Durant 44.5 POINTS_REBOUNDS"
        try:
            line = float(t["k"])
        except (TypeError, ValueError):
            continue
        if line == int(line):
            continue
        out.append({"market": m, "event": t["e"], "type": t["mt"], "stat": NP.MARKET_STAT[t["mt"]], "player": player, "line": line, "start": t["st"],
                    "quote_ts": t["ts"], "bid_over": None if t["b"][0] is None else float(t["b"][0]), "bid_under": None if t["b"][1] is None else float(t["b"][1]),
                    "qty_over": t["q"][0], "qty_under": t["q"][1]})
    return out


def latest_out(timeline: Path = TIMELINE) -> set[tuple[str, str]]:
    """(team, normalised name) of NBA players listed Out on ESPN's live injury report right now (falls back to the logged timeline)."""
    try:
        from pm_edges import injuries as I
        live = I.report("nba")
        if live:
            return {(NP.FIX.get(r["team"], r["team"]), NP.norm_name(r.get("name") or "")) for r in live if r.get("status") == "Out"}
    except Exception as e:                                   # network or schema trouble: use what the poller logged
        print("live injury report failed, using timeline:", type(e).__name__)
    last = {}
    for line in timeline.read_text().splitlines():
        if '"src": "injury"' not in line:
            continue
        r = json.loads(line)
        if r.get("league") == "nba":
            last[(r["team"], r.get("athlete_id"))] = r
    return {(NP.FIX.get(k[0], k[0]), NP.norm_name(r.get("name") or "")) for k, r in last.items() if r.get("status") == "Out"}


def build_rows(quotes: list[dict], p: pd.DataFrame, today: pd.Timestamp):
    """Pseudo player rows for tonight (one synthetic game per Novig event) appended to the box history, so the trailing features come out of the
    same code path as training. Absences: players with an Out status, plus baseline-20+ minute players with no prop market."""
    last_team = p.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    name_to = {}
    for pid, r in last_team.iterrows():
        name_to.setdefault(NP.norm_name(r.player), []).append((pid, r.team))
    amb = {k for k, v in name_to.items() if len(v) > 1}

    def find(key: str):
        if key in name_to and key not in amb:
            return name_to[key][0]
        first, _, last = key.partition(" ")[0], None, key.split(" ")[-1]
        hits = [v[0] for k, v in name_to.items() if k.split(" ")[-1] == last and k not in amb and (k.startswith(first[:4]) or first.startswith(k.split(" ")[0][:4]))]
        return hits[0] if len(hits) == 1 else None
    out_set = latest_out()
    by_event: dict[str, dict[str, str]] = {}
    unmatched = set()
    for q in quotes:
        hit = find(NP.norm_name(q["player"]))
        if hit is None:
            unmatched.add(q["player"]); continue
        pid, team = hit
        by_event.setdefault(q["event"], {})[pid] = team
        q["pid"], q["team"] = pid, team
    synth = []
    recent = p[p.played].sort_values("date")
    for e, players in by_event.items():
        teams = pd.Series(list(players.values())).value_counts()
        keep = set(teams.index[:2])
        roster = dict(players)
        for team in keep:                                    # recent rotation players without a prop market are still playing unless listed Out
            gids = recent[recent.team == team].game_id.drop_duplicates().tail(10)
            for pid in recent[(recent.team == team) & recent.game_id.isin(gids)].player_id.unique():
                roster.setdefault(pid, team)
        for pid, team in roster.items():
            if team in keep and not any((team, NP.norm_name(last_team.loc[pid].player)) == o for o in out_set):
                synth.append({"game_id": f"NEW-{e}", "date": today, "season": 2027, "stype": 2, "team": team, "player_id": pid, "player": last_team.loc[pid].player,
                              "starter": False, "dnp": False, "reason": "", "min": np.nan, "pts": np.nan, "reb": np.nan, "ast": np.nan, "tpm": np.nan,
                              "stl": np.nan, "blk": np.nan, "played": True, "pf": np.nan, "oreb": np.nan, "dreb": np.nan})
    return pd.concat([p, pd.DataFrame(synth)], ignore_index=True), unmatched, out_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=0)
    a = ap.parse_args()
    now_ms = int(time.time() * 1000)
    quotes = load_quotes(now_ms)
    if not quotes:
        raise SystemExit("no upcoming prop quotes in data/props")
    p = NP.load_player_rows()
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - p.date.max()).days
    if age > 2:
        print(f"WARNING: player box scores end {p.date.max().date()} ({age} days old): run `python tools/fetch_box.py` first, form features are stale")
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    pp, unmatched, out_set = build_rows(quotes, p, today)
    d = NP.player_frame(pp)
    hist = d[d.min.notna() if hasattr(d, "min") and False else d["min"].notna()]
    hist = hist[(hist.f_games >= 5) & (hist.f_min_20 >= 12)]
    new = d[d.game_id.astype(str).str.startswith("NEW-") & (d.f_games >= 5) & (d.f_min_20 >= 12)]
    tr = NP.train_all(hist)
    new = new.assign(**{f"mu_{s}": NP.predict_mean(tr["models"][s], new, s) for s in NP.TARGETS})
    key = new.set_index("player_id")
    rows = []
    for q in quotes:
        if q.get("pid") not in key.index:
            continue
        r = key.loc[[q["pid"]]].iloc[0] if isinstance(key.loc[q["pid"]], pd.DataFrame) else key.loc[q["pid"]]
        mu = float(r[f"mu_{q['stat']}"]); po = float(NP.p_over(mu, tr["alpha"][q["stat"]], q["line"]))
        ask_o = None if q["bid_under"] is None else 1 - q["bid_under"]; ask_u = None if q["bid_over"] is None else 1 - q["bid_over"]
        two = q["bid_over"] is not None and q["bid_under"] is not None
        rows.append({**{k: q[k] for k in ("market", "event", "type", "stat", "player", "pid", "team", "line", "start", "quote_ts", "bid_over", "bid_under", "qty_over", "qty_under")},
                     "mu": round(mu, 3), "p_over": round(po, 4), "alpha": round(tr["alpha"][q["stat"]], 4),
                     "mid_over": round((q["bid_over"] + 1 - q["bid_under"]) / 2, 4) if two else None,
                     "ask_over": ask_o, "ask_under": ask_u,
                     "edge_over": None if ask_o is None else round(po - ask_o, 4), "edge_under": None if ask_u is None else round(1 - po - ask_u, 4),
                     "miss_min": float(r.miss_min), "miss_n": float(r.miss_n), "form_min": round(float(r.f_min_20), 1)})
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    path = OUT / f"{stamp[:10]}.json.gz"
    old = json.load(gzip.open(path, "rt")) if path.exists() else {"freezes": []}
    all_rows = rows
    rows = [r for r in all_rows if r["mid_over"] is not None or max([x for x in (r["edge_over"], r["edge_under"]) if x is not None], default=-1) >= 0.03]   # keep two-sided + any one-sided with a real gap
    old["freezes"].append({"frozen_at": stamp, "unmatched_players": sorted(unmatched), "out": sorted(f"{t}:{n}" for t, n in out_set), "rows": rows})
    with gzip.open(path, "wt") as f:
        json.dump(old, f, separators=(",", ":"))
    best = [(max(x for x in (r["edge_over"], r["edge_under"]) if x is not None), r) for r in all_rows if r["edge_over"] is not None or r["edge_under"] is not None]
    print(f"froze {len(rows)} of {len(all_rows)} prop markets ({sum(1 for r in rows if r['mid_over'] is not None)} two-sided); {len(unmatched)} player names unmatched; -> {path}")
    for e, r in sorted(best, key=lambda x: -x[0])[: a.top]:
        side = "OVER" if (r["edge_over"] or -9) >= (r["edge_under"] or -9) else "UNDER"
        print(f"  {r['team']} {r['player']:22s} {r['type'][:12]:12s} line {r['line']:>5} model mean {r['mu']:.1f}  P(over) {r['p_over']:.2f}  {side} edge {e:+.3f}  miss_min {r['miss_min']:.0f}")


if __name__ == "__main__":
    main()

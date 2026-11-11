"""Polymarket US vs Novig on NBA player props (read-only): price gaps and "locks".

Polymarket lists ladders "Will <player> record at least k <stat>" = P(stat >= k) = Novig's Over (k - 0.5).
For every Polymarket rung with a Novig counterpart we compute
  gap   = Polymarket mid - Novig mid                      (two-sided Novig quotes only)
  lock1 = PM Yes ask + PM fee + Novig Under ask            (< 1 pays at least $1 for sure)
  lock2 = Novig Over ask + PM No ask (= 1 - PM Yes bid) + PM fee
and, for anything under 1.02, the Polymarket book depth. Polymarket US taker fee ~ 0.0695 p (1-p) per share.
Top-of-book quotes on Polymarket are often dust, so any "lock" is re-checked against the book before it counts.

    python tools/xv_props.py
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pm_edges import nba_props as NP  # noqa: E402
from tools.freeze_props import _rows  # noqa: E402

GW = "https://gateway.polymarket.us"
PM_STAT = {"basketball_player_points": "POINTS", "basketball_player_rebounds": "REBOUNDS", "basketball_player_assists": "ASSISTS",
           "basketball_player_threes": "THREE_POINTERS_MADE", "basketball_player_points_rebounds_assists": "POINTS_REBOUNDS_ASSISTS",
           "basketball_player_steals": "STEALS", "basketball_player_blocks": "BLOCKS"}
fee = lambda p: 0.0695 * p * (1 - p)


def novig_props(now_ms: int, d: Path = Path("data/props")):
    meta, events, last = {}, {}, {}
    for r in _rows(sorted(d.glob("meta_*.jsonl.gz"))):
        if r.get("k") == "mk":
            meta[r["m"]] = r
        elif r.get("k") == "ev":
            events[r["e"]] = r
    for r in _rows(sorted(d.glob("touch_*.jsonl.gz"))):
        if r["m"] in meta and r["st"] > now_ms and (r["m"] not in last or r["ts"] >= last[r["m"]]["ts"]):
            last[r["m"]] = r
    out = {}
    for m, t in last.items():
        mk = meta[m]
        if mk["mt"] not in PM_STAT.values() or len(mk["n"]) != 2 or not mk["n"][0].startswith("Over"):
            continue
        try:
            line = float(t["k"])
        except (TypeError, ValueError):
            continue
        player = mk["d"].rsplit(" ", 2)[0]
        bo, bu = t["b"]
        out[(t["e"], NP.norm_name(player), mk["mt"], line)] = {"bo": None if bo is None else float(bo), "bu": None if bu is None else float(bu), "qo": t["q"][0], "qu": t["q"][1], "start": t["st"]}
    return out, events


def pm_event(client: httpx.Client, ev: dict):
    desc = ev.get("d") or ""
    away, _, home = desc.partition(" @ ")
    q = f"{away.split()[-1]} {home.split()[-1]}"
    r = client.get(f"{GW}/v1/search", params={"query": q}, timeout=30).json()
    for e in r.get("events", []):
        if e["slug"].startswith("nba-"):
            st = e.get("startDate")
            if st and abs(pd.Timestamp(st).timestamp() * 1000 - ev["st"]) < 6 * 3600e3:
                return client.get(f"{GW}/v1/events/slug/{e['slug']}", timeout=30).json()["event"]
    return None


def main() -> None:
    now = int(time.time() * 1000)
    nv, events = novig_props(now)
    rows = []
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (research)"}) as c:
        for eid, ev in events.items():
            if ev.get("lg") != "NBA" or ev["st"] < now:
                continue
            pe = pm_event(c, ev)
            if not pe:
                continue
            for m in pe["markets"]:
                st = PM_STAT.get(m.get("sportsMarketType"))
                mm = re.match(r"Will (.+?) record at least (\d+) ", m.get("question", ""))
                if not st or not mm or not (m.get("bestBidQuote") and m.get("bestAskQuote")):
                    continue
                k = int(mm.group(2)); h = nv.get((eid, NP.norm_name(mm.group(1)), st, k - 0.5))
                if not h:
                    continue
                yb, ya = float(m["bestBidQuote"]["value"]), float(m["bestAskQuote"]["value"])
                ov_ask = None if h["bu"] is None else 1 - h["bu"]; un_ask = None if h["bo"] is None else 1 - h["bo"]
                two = h["bo"] is not None and h["bu"] is not None
                rows.append({"event": ev["d"], "player": mm.group(1), "stat": st, "k": k, "pm_bid": yb, "pm_ask": ya, "pm_mid": (yb + ya) / 2, "slug": m["slug"],
                             "nv_mid": (h["bo"] + 1 - h["bu"]) / 2 if two else None, "nv_over_ask": ov_ask, "nv_under_ask": un_ask, "q_over": h["qo"], "q_under": h["qu"],
                             "lock1": None if un_ask is None else ya + fee(ya) + un_ask, "lock2": None if ov_ask is None else ov_ask + (1 - yb) + fee(1 - yb)})
    D = pd.DataFrame(rows)
    if D.empty:
        raise SystemExit("no overlapping markets")
    D["gap"] = D.pm_mid - D.nv_mid
    D["lock"] = D[["lock1", "lock2"]].min(axis=1)
    live = D[(D.pm_ask < 0.98) & (D.pm_bid > 0.02)]                         # drop 0.99/0.01 dust ladders
    print(f"matched {len(D)} rungs ({len(live)} away from the 1c/99c rails) across {D.event.nunique()} games")
    g = live.dropna(subset=["gap"])
    print(f"PM mid - Novig mid, two-sided rungs n={len(g)}: mean {g.gap.mean():+.4f}  sd {g.gap.std():.4f}  mean|gap| {g.gap.abs().mean():.4f}  share |gap|>0.05: {(g.gap.abs() > .05).mean():.3f}")
    print(g.groupby("stat").gap.agg(["size", "mean", "std"]).round(4).to_string())
    for th in (1.0, 1.01, 1.02, 1.04):
        print(f"lock cost < {th}: {(live.lock < th).sum()}")
    best = live.sort_values("lock").head(8)
    print(best[["event", "player", "stat", "k", "pm_bid", "pm_ask", "nv_over_ask", "nv_under_ask", "q_over", "q_under", "lock1", "lock2"]].round(3).to_string(index=False))
    D.to_pickle("data/model/nba/xv_props.pkl")


if __name__ == "__main__":
    main()

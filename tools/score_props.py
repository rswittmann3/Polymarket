"""Score the frozen NBA prop predictions (snapshots/props_frozen/*.json.gz) against actual box scores. Read-only.

For each frozen market whose game is final:
  * outcome: did the player's stat exceed the line (players who did not play are dropped; they are voided/unknowable here)
  * model vs market on the SAME markets: Brier score of the model's P(over) against the market mid (two-sided quotes only)
  * paper bets: the side with model edge >= threshold, bought at the frozen ask; gain = result - ask, in probability points (clustered by game)
  * market movement: did the closing mid (last touch before tip, from data/props) move toward the model after the freeze?
Uses the EARLIEST freeze that has a quote for each market ("early") and the LATEST before the game ("late").

    python tools/score_props.py            # prints the summary and writes data/model/props_scored.json
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pm_edges import nba_props as NP  # noqa: E402
from tools import fetch_box as FB  # noqa: E402

FROZEN, PROPS = Path("snapshots/props_frozen"), Path("data/props")
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE = Path("data/history/box/props_outcomes.json")


def et_date(ms: int) -> str:
    return (datetime.fromtimestamp(ms / 1000, tz=timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")


def final_boxes(day: str, client: httpx.Client) -> dict[str, dict]:
    """player_id -> stat line for every finished NBA game on ET date `day` (YYYYMMDD)."""
    out = {}
    r = client.get(SB, params={"dates": day, "limit": 100}, timeout=40)
    for e in r.json().get("events", []):
        comp = e["competitions"][0]
        if comp["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        g = {"id": e["id"], "date": e["date"][:10], "season": e["season"]["year"], "stype": e["season"]["type"],
             "teams": [{"homeAway": c["homeAway"], "abbr": c["team"]["abbreviation"]} for c in comp["competitors"]]}
        s = client.get(FB.URL, params={"event": e["id"]}, timeout=40).json()
        p = FB.parse(g, s)
        if not p:
            continue
        for q in p["players"]:
            if q.get("player_id") and not q.get("dnp") and (q.get("min") or 0) > 0:
                out[str(q["player_id"])] = {k: (q.get(k) or 0) for k in ("pts", "reb", "ast", "tpm", "stl", "blk", "min")} | {"game": e["id"]}
    return out


def outcomes(days: set[str]) -> dict[str, dict]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (research; pm_edges)"}) as c:
        for d in sorted(days):
            if d not in cache:
                box = final_boxes(d, c)
                if box:                                               # only cache days that actually finished
                    cache[d] = box
    CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(cache))
    merged = {}
    for d in days:
        merged.update(cache.get(d, {}))
    return merged


def closing_mids(starts: dict[str, int]) -> dict[str, float | None]:
    """Last two-sided mid for each market at or before its start, from the prop touch logs."""
    last = {}
    for f in sorted(glob.glob(str(PROPS / "touch_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                m = r["m"]
                if m in starts and r["ts"] <= starts[m] and (m not in last or r["ts"] >= last[m][0]):
                    last[m] = (r["ts"], r["b"])
    out = {}
    for m, (_, b) in last.items():
        out[m] = None if b[0] is None or b[1] is None else (float(b[0]) + 1 - float(b[1])) / 2
    return out


def load_frozen() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(FROZEN / "*.json.gz"))):
        for fr in json.load(gzip.open(f, "rt"))["freezes"]:
            for r in fr["rows"]:
                rows.append({**r, "frozen_at": fr["frozen_at"]})
    return pd.DataFrame(rows)


def cluster_se(x: pd.Series, groups: pd.Series) -> float:
    d = pd.DataFrame({"x": x - x.mean(), "g": groups}).groupby("g").x.sum()
    return float(np.sqrt((d ** 2).sum()) / len(x))


def summarize(df: pd.DataFrame, label: str) -> dict:
    out = {"which": label, "markets": int(len(df))}
    two = df[df.mid_over.notna()]
    out["two_sided"] = int(len(two))
    if len(two):
        bm = ((two.mid_over - two.over) ** 2).mean(); bp = ((two.p_over - two.over) ** 2).mean()
        d = (two.mid_over - two.over) ** 2 - (two.p_over - two.over) ** 2
        out.update(brier_market=round(float(bm), 4), brier_model=round(float(bp), 4), brier_gain=round(float(d.mean()), 4), brier_gain_se=round(cluster_se(d, two.event), 4))
        for th in (0.03, 0.06, 0.10):
            sel = two[(two.p_over - two.mid_over).abs() >= th]
            if len(sel):
                toward = ((np.sign(sel.p_over - sel.mid_over)) * (sel.over - sel.mid_over))
                out[f"vs_mid_gap{int(th*100)}"] = {"n": int(len(sel)), "mean_result_minus_mid_toward_model": round(float(toward.mean()), 4), "se": round(cluster_se(toward, sel.event), 4)}
    bets = []
    for side in ("over", "under"):
        e = df[f"edge_{side}"]; ask = df[f"ask_{side}"]
        won = (df.over == (1 if side == "over" else 0)).astype(float)
        b = df[e.notna()].assign(side=side, edge=e, ask=ask, won=won)
        bets.append(b[["event", "player", "stat", "line", "side", "edge", "ask", "won", "close_mid", "mid_over"]])
    b = pd.concat(bets)
    for th in (0.05, 0.10, 0.15):
        s = b[b.edge >= th]
        if len(s):
            g = s.won - s.ask
            out[f"bets_edge{int(th*100)}"] = {"n": int(len(s)), "win_rate": round(float(s.won.mean()), 3), "avg_ask": round(float(s.ask.mean()), 3), "mean_gain_prob_pts": round(float(g.mean()), 4),
                                              "se": round(cluster_se(g, s.event), 4), "roi": round(float(g.sum() / s.ask.sum()), 4)}
    mv = df[df.close_mid.notna() & df.mid_over.notna()]
    if len(mv):
        toward = np.sign(mv.p_over - mv.mid_over) * (mv.close_mid - mv.mid_over)
        out["close_moved_toward_model"] = {"n": int(len(mv)), "mean": round(float(toward.mean()), 4), "se": round(cluster_se(toward, mv.event), 4)}
    return out


def main() -> None:
    F = load_frozen()
    if F.empty:
        raise SystemExit("no frozen props yet")
    F = F[F.pid.notna()].copy(); F["pid"] = F.pid.astype(str)
    now_ms = int(time.time() * 1000)
    F = F[F.start < now_ms - 4 * 3600 * 1000]                          # game surely over
    if F.empty:
        raise SystemExit("no frozen games have finished yet")
    box = outcomes({et_date(s) for s in F.start.unique()})
    F["stat_val"] = [sum(box.get(p, {}).get(c, np.nan) for c in NP.TARGETS[s]) if p in box else np.nan for p, s in zip(F.pid, F.stat)]
    F = F[F.stat_val.notna()].copy()                                  # DNP / unfinished: dropped
    F["over"] = (F.stat_val > F.line).astype(float)
    cm = closing_mids(dict(zip(F.market, F.start)))
    F["close_mid"] = F.market.map(cm)
    F = F.sort_values("frozen_at")
    res = {"early": summarize(F.groupby("market", as_index=False).head(1), "early"), "late": summarize(F.groupby("market", as_index=False).tail(1), "late"),
           "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "freeze_days": sorted({x[:10] for x in F.frozen_at})}
    Path("data/model").mkdir(exist_ok=True); Path("data/model/props_scored.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Run the walk-forward models for every league and write data/model/{results,slate}.json.

    python tools/run_model.py                 # all leagues, slate for today and tomorrow
    python tools/run_model.py --league nba --days 3

results.json: per league and task, out-of-sample skill versus the closing line, by-season gains, a shuffled-outcome
placebo, betting/closing-line-value simulation, feature importances, and how well the market itself is calibrated.
slate.json: upcoming games with the current line, the model's view, and how much to trust each model.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from pm_edges import angles as A
from pm_edges import injuries as I
from pm_edges import model as M
from pm_edges import slate as S
from pm_edges.trends import history_frames

PLAN = {"nfl": ["side", "total"], "nba": ["side", "total", "move_side", "move_total"], "nhl": ["total", "win"]}
LABELS = {"side": "Spread residual", "total": "Total residual", "win": "Win probability residual",
          "move_side": "Spread move (open to close)", "move_total": "Total move (open to close)"}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def league_results(lg: str, data_dir: Path) -> tuple[dict, dict]:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):               # one thread per league process: three processes on four cores, no oversubscription
        return _league_results(lg, data_dir)


def _league_results(lg: str, data_dir: Path) -> tuple[dict, dict]:
    games, _ = history_frames(data_dir, lg)
    frame = M.game_frame(A.team_games(games))
    out = {"games": int(len(frame)), "seasons": [int(frame["season"].min()), int(frame["season"].max())],
           "calibration": M.market_calibration(frame), "tasks": {}}
    fitted = {}
    for task in PLAN[lg]:
        ev = M.evaluate_task(frame, task, lg)
        if ev.get("available"):
            ev["placebo_t"] = M.placebo(frame, task, lg)
            ev["importances"] = M.importances(frame, task, lg)
            fitted[task] = M.fit_final(frame, task, lg)
        ev["status"] = M.status(task, ev)
        ev["label"] = LABELS[task]
        out["tasks"][task] = ev
        print(f"  {lg} {task}: {ev['status']['label']}", flush=True)
    return out, {"games": games, "fitted": fitted}


def availability(lg: str, games: list[dict]) -> tuple[dict, dict]:
    """({game id: injury rows}, {team|game: goalie}) for the slate's teams, with a per-game line for the players that matter."""
    teams = {t for g in games for t in (g["home"], g["away"])}
    rows = [r for r in I.report(lg) if r["team"] in teams and r["status"] in I.ACTIVE_STATUSES]
    ids = sorted({r["athlete_id"] for r in rows if r["athlete_id"]})
    values = {}
    if lg in ("nba", "nhl") and ids:
        cache = I._load()
        with ThreadPoolExecutor(6) as pool:
            for aid, v in zip(ids, pool.map(lambda a: I.player_value(lg, a, cache=cache), ids)):
                values[aid] = v
        I.CACHE.parent.mkdir(parents=True, exist_ok=True)
        I.CACHE.write_text(json.dumps(cache))
    inj = {f"{g['away']}@{g['home']}|{g['date']}": I.game_context(lg, g["away"], g["home"], rows, values) for g in games}
    goalies = {}
    if lg == "nhl":
        for day in sorted({g["date"].replace("-", "") for g in games}):
            for (team, eid), gk in I.probable_goalies(day).items():
                v = I.player_value("nhl", gk["athlete_id"], cache=I._load()) if gk.get("athlete_id") else {}
                goalies[f"{team}|{eid}"] = {**gk, "team": team, "save_pct": v.get("save_pct"), "gaa": v.get("gaa"), "games": v.get("games"), "day": day}
    return inj, goalies


def slate_rows(lg: str, ctx: dict, results: dict, start: date, days: int) -> list[dict]:
    games = slate_games = S.slate(lg, start, days)
    games = [g for g in slate_games if g.get("total") is not None]
    if not games:
        return []
    specs = []
    for g in games:
        spec = {"away": g["away"], "home": g["home"], "date": g["date"], "total_close": g["total"],
                "home_spread_close": g.get("home_spread") if lg != "nhl" else None,
                "home_spread_open": g.get("home_spread_open") if lg != "nhl" else None,
                "total_open": g.get("total_open"), "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml"),
                "over_odds": g.get("over_odds"), "under_odds": g.get("under_odds")}
        for k in ("week", "weekday", "hour", "roof", "div_game"):
            if g.get(k) is not None:
                spec[k] = g[k]
        specs.append(spec)
    rows = M.game_rows(A.upcoming_many(ctx["games"], specs))
    inj, goalies = availability(lg, games)
    out = []
    for g, (_, r) in zip(games, rows.iterrows()):
        item = {k: g.get(k) for k in ("league", "away", "home", "date", "kickoff", "provider", "total", "total_open", "home_spread",
                                      "home_spread_open", "home_ml", "away_ml", "over_odds", "under_odds", "week")}
        item["context"] = {"home_rest": _num(r.get("rest_days")), "away_rest": _num(r.get("opp_rest_days")),
                           "home_game_num": _num(r.get("game_num")), "away_form_total_5": _num(r.get("opp_tot_5")),
                           "home_form_total_5": _num(r.get("tot_5"))}
        preds = {}
        for task, fit in ctx["fitted"].items():
            x = rows.loc[[r.name]].copy()
            if task in M.MOVE_TASKS:                 # features come from the opening line; with none, the current line stands in
                if task == "move_total" and pd.isna(x["total_open"]).all():
                    x["total_open"] = x["total"]; x["total_open_rel"] = x["total_rel"]
                    x["total_open_vs_avg"] = x["total"] - (x["prev_avg_points"] + x["opp_prev_avg_points"]) / 2
                if task == "move_side" and ("spread_open" not in x or pd.isna(x["spread_open"]).all()):
                    x["spread_open"] = x["spread"]
            p = float(M.predict_final(fit, x)[0])
            preds[task] = {"pred": p, "status": results["tasks"][task]["status"]["label"], "tone": results["tasks"][task]["status"]["tone"]}
        item["preds"] = preds
        item["injuries"] = inj.get(f"{g['away']}@{g['home']}|{g['date']}", [])
        item["goalies"] = [v for v in goalies.values() if v["team"] in (g["home"], g["away"]) and v["day"] == g["date"].replace("-", "")]
        item["espn_id"] = g.get("espn_id")
        if "total" in preds and g.get("total") is not None:
            item["model_total"] = g["total"] + preds["total"]["pred"]
        if "side" in preds and g.get("home_spread") is not None:
            item["model_home_spread"] = g["home_spread"] - preds["side"]["pred"]
        if "move_side" in preds and g.get("home_spread_open") is not None:
            item["pred_close_spread"] = g["home_spread_open"] + preds["move_side"]["pred"]
        if "move_total" in preds and g.get("total_open") is not None:
            item["pred_close_total"] = g["total_open"] + preds["move_total"]["pred"]
        out.append(item)
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(PLAN), nargs="+", default=list(PLAN))
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--start", type=date.fromisoformat, default=date.today())
    args = ap.parse_args()
    results = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "leagues": {}}
    slate = []
    for lg in args.league:                    # history frames are built (and cached) one league at a time, then the fits run in parallel
        history_frames(args.data_dir, lg)
    with ProcessPoolExecutor(len(args.league)) as pool:
        futures = {lg: pool.submit(league_results, lg, args.data_dir) for lg in args.league}
        done = {lg: f.result() for lg, f in futures.items()}
    for lg in args.league:
        res, ctx = done[lg]
        results["leagues"][lg] = res
        slate += slate_rows(lg, ctx, res, args.start, args.days)
    out = args.data_dir / "model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(clean(results), indent=1))
    (out / "slate.json").write_text(json.dumps(clean({"generated": results["generated"], "games": slate}), indent=1))
    print(f"wrote {out}/results.json and slate.json ({len(slate)} games)")


if __name__ == "__main__":
    main()

"""Freeze today's NBA line-move predictions so they can be scored once the games close.

Fits the two open-to-close models on all history (about five seconds), takes the upcoming NBA slate from ESPN with its
opening and current DraftKings lines, and writes snapshots/frozen/nba_<time>.json. Nothing in the file can be
revised afterwards: it records the time it was made, the lines at that time, and the predicted closing lines.

    python tools/freeze_predictions.py            # today and tomorrow's games that have not tipped
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_history
from pm_edges import angles as A
from pm_edges import model as M
from pm_edges import slate as S
from pm_edges.trends import history_frames


def freeze(data_dir: Path, out_dir: Path, days: int = 2, now: datetime | None = None, refresh: bool = True) -> Path | None:
    warnings.filterwarnings("ignore")
    now = now or datetime.now(timezone.utc)
    if refresh:                                          # without the season's finished games, form and rest days are last spring's
        end_year = now.year + 1 if now.month >= 9 else now.year
        fetch_history.fetch_espn("nba", [end_year], workers=8)
    games, _ = history_frames(data_dir, "nba")
    frame = M.game_frame(A.team_games(games))
    fits = {t: M.fit_final(frame, t, "nba") for t in M.MOVE_TASKS}
    slate = [g for g in S.espn_slate("nba", now.date(), days) if g.get("total") is not None and g.get("total_open") is not None]
    slate = [g for g in slate if datetime.fromisoformat(g["kickoff"]) > now]
    if not slate:
        return None
    specs = [{"away": g["away"], "home": g["home"], "date": g["date"], "total_close": g["total"], "total_open": g["total_open"],
              "home_spread_close": g.get("home_spread"), "home_spread_open": g.get("home_spread_open"),
              "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml"), "over_odds": g.get("over_odds"), "under_odds": g.get("under_odds")}
             for g in slate]
    rows = M.game_rows(A.upcoming_many(games, specs))
    out = []
    for g, (_, r) in zip(slate, rows.iterrows()):
        x = rows.loc[[r.name]]
        pred = {t: float(M.predict_final(fits[t], x)[0]) for t in M.MOVE_TASKS}
        out.append({"league": "nba", "espn_id": g["espn_id"], "away": g["away"], "home": g["home"], "date": g["date"], "kickoff": g["kickoff"],
                    "total": g["total"], "total_open": g["total_open"], "home_spread": g.get("home_spread"),
                    "home_spread_open": g.get("home_spread_open"),
                    "pred_close_total": g["total_open"] + pred["move_total"],
                    "pred_close_spread": (g["home_spread_open"] + pred["move_side"]) if g.get("home_spread_open") is not None else None})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nba_{now:%Y%m%dT%H%MZ}.json"
    played = games[games["kind"].isin(["reg", "post", "playin"]) & games["home_pts"].notna()]
    path.write_text(json.dumps({"saved_at": now.isoformat(timespec="seconds"), "history_through": f"{played['date'].max():%Y-%m-%d}",
                                "games": out}, indent=1))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("snapshots/frozen"))
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--no-refresh", action="store_true", help="do not fetch the latest finished games first")
    a = ap.parse_args()
    p = freeze(a.data_dir, a.out, a.days, refresh=not a.no_refresh)
    print(f"froze {len(json.loads(p.read_text())['games'])} games: {p}" if p else "no upcoming games with opening lines")

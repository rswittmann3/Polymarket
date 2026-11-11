"""Possession-based NBA team ratings built from box scores, fitted walk-forward (each game sees only earlier games).

For every game date D the ratings use the games before D, each weighted by 0.5 ** (age_in_days / half_life):

    points per 100 possessions  =  mu + hca * home + off[team] + allow[opponent]

with a ridge penalty pulling every team toward average. A game's expected margin is the difference of the two teams' expected efficiencies
times their expected pace. Nothing here uses the betting line, so comparing it with the closing line asks whether the line is missing
something a box-score model knows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BOX = Path("data/history/box")
FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NBA_TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
             "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}


def team_rows(box: Path = BOX) -> pd.DataFrame:
    """One row per team-game: efficiency, pace, opponent, home flag. Regular season and playoffs, NBA teams only."""
    t = pd.read_pickle(box / "team_box.pkl")
    t["team"] = t.team.replace(FIX)
    t = t[t.team.isin(NBA_TEAMS)].copy()
    t["date"] = pd.to_datetime(t.date)
    t["pts"] = 2 * (t.fgm - t.tpm) + 3 * t.tpm + t.ftm
    t["poss_own"] = t.fga + 0.44 * t.fta - t.oreb + t.tov
    o = t[["game_id", "team", "pts", "poss_own"]].rename(columns={"team": "opp", "pts": "opp_pts", "poss_own": "poss_opp"})
    m = t.merge(o, on="game_id")
    m = m[m.team != m.opp]
    m = m[m.groupby(["game_id", "team"]).team.transform("size") == 1]
    m["poss"] = (m.poss_own + m.poss_opp) / 2
    m["off"] = 100 * m.pts / m.poss
    m["margin"] = m.pts - m.opp_pts
    return m.sort_values(["date", "game_id"]).reset_index(drop=True)


def fit_ratings(rows: pd.DataFrame, asof: pd.Timestamp, teams: list[str], half_life: float = 60.0, ridge: float = 25.0, max_days: int = 540) -> dict:
    """Ratings as of `asof` using only rows strictly before it."""
    r = rows[(rows.date < asof) & (rows.date >= asof - pd.Timedelta(days=max_days))]
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if len(r) < 60:
        return {}
    w = 0.5 ** ((asof - r.date).dt.days.values / half_life)
    X = np.zeros((len(r), 2 + 2 * n))
    X[:, 0] = 1
    X[:, 1] = r.home.values.astype(float)
    X[np.arange(len(r)), 2 + r.team.map(idx).values] = 1
    X[np.arange(len(r)), 2 + n + r.opp.map(idx).values] = 1
    y = r.off.values
    A = (X * w[:, None]).T @ X
    A[2:, 2:] += ridge * np.eye(2 * n) * (w.sum() / len(r))
    beta = np.linalg.solve(A, (X * w[:, None]).T @ y)
    # pace: each team's weighted mean possessions, shrunk toward the league mean
    lg_pace = np.average(r.poss.values, weights=w)
    pace = {}
    for t in teams:
        s = r[(r.team == t)]
        ww = 0.5 ** ((asof - s.date).dt.days.values / half_life)
        pace[t] = lg_pace + (np.average(s.poss.values, weights=ww) - lg_pace) * (ww.sum() / (ww.sum() + 4)) if len(s) else lg_pace
    return {"mu": beta[0], "hca": beta[1], "off": dict(zip(teams, beta[2:2 + n])), "allow": dict(zip(teams, beta[2 + n:])), "pace": pace, "lg_pace": lg_pace,
            "games": {t: float(w[(r.team == t).values].sum()) for t in teams}}


def expected(rat: dict, home: str, away: str) -> dict | None:
    if not rat:
        return None
    h_eff = rat["mu"] + rat["hca"] + rat["off"][home] + rat["allow"][away]
    a_eff = rat["mu"] + rat["off"][away] + rat["allow"][home]
    pace = rat["lg_pace"] + (rat["pace"][home] - rat["lg_pace"]) + (rat["pace"][away] - rat["lg_pace"])
    return {"margin": (h_eff - a_eff) * pace / 100, "total": (h_eff + a_eff) * pace / 100, "pace": pace, "h_eff": h_eff, "a_eff": a_eff}


def walk_forward(rows: pd.DataFrame, half_life: float = 60.0, ridge: float = 25.0) -> pd.DataFrame:
    """Expected margin/total for every game in `rows`, using only games before its date."""
    teams = sorted(rows.team.unique())
    games = rows[rows.home].rename(columns={"team": "home_team", "opp": "away_team"})[["game_id", "date", "home_team", "away_team", "margin", "season"]]
    games["points"] = rows[rows.home].set_index("game_id").loc[games.game_id, "pts"].values + rows[~rows.home].set_index("game_id").loc[games.game_id, "pts"].values
    out = []
    for d, grp in games.groupby("date"):
        rat = fit_ratings(rows, d, teams, half_life, ridge)
        for g in grp.itertuples():
            e = expected(rat, g.home_team, g.away_team) if rat else None
            if e:
                out.append({"game_id": g.game_id, "date": d, "home": g.home_team, "away": g.away_team, "season": g.season, "margin": g.margin,
                            "points": g.points, "model_margin": e["margin"], "model_total": e["total"], "model_pace": e["pace"]})
    return pd.DataFrame(out)

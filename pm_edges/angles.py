"""Situational betting "angles" over the closing-line history, in the style of TrendsCenter.

An angle is a set of conditions on a team's game (at home, spread, total, which game
of the season, what happened the last time these two teams met, ...) and a bet
(under, over, or the team against the spread). `evaluate` returns the record at flat
-110 and how surprising it is. `null_search` shows how often *random* angles of the
same shape do as well, which is the question that matters for a number like 15-0.

Totals are bet once per game: an angle that both teams in a game satisfy counts that
game once.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PAYOUT = 100 / 110          # profit on a winning -110 bet, per unit risked
BREAKEVEN = 110 / 210       # win rate needed at -110


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

# game-level columns carried onto both teams' rows when the games frame has them
GAME_EXTRAS = ["week", "playoff_round", "ot", "neutral", "roof", "temp", "wind", "div_game", "weekday", "hour",
               "over_odds", "under_odds"]


def _implied(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(o > 0, 100 / (o + 100), -o / (-o + 100)), index=odds.index).where(o.abs() >= 100)


def team_games(games: pd.DataFrame, kinds=("reg", "post", "playin"), keep_unplayed: bool = False) -> pd.DataFrame:
    """One row per team per game, with outcome and situation features.

    `games` has one row per game with lines from the home team's side (home_spread_close,
    total_close, optionally home_ml/away_ml, *_open, over_odds/under_odds, and league extras)."""
    g = games[games["kind"].isin(kinds)].copy()
    if not keep_unplayed:
        g = g.dropna(subset=["home_pts", "away_pts"])
    g = g.reset_index(drop=True)
    g["game_id"] = np.arange(len(g))

    def col(name: str) -> pd.Series:
        return pd.to_numeric(g[name], errors="coerce") if name in g else pd.Series(np.nan, index=g.index)

    def side(home: bool) -> pd.DataFrame:
        me, you, sign = ("home", "away", 1) if home else ("away", "home", -1)
        d = {"game_id": g["game_id"], "date": g["date"], "season": g["season"], "kind": g["kind"],
             "team": g[me], "opp": g[you], "at_home": int(home),
             "pts_for": col(f"{me}_pts"), "pts_against": col(f"{you}_pts"),
             "spread": sign * col("home_spread_close"), "total": col("total_close"),
             "spread_open": sign * col("home_spread_open"), "total_open": col("total_open"),
             "ml": col(f"{me}_ml"), "opp_ml": col(f"{you}_ml"), "spread_odds": col(f"{me}_spread_odds")}
        # totals are settled on the sportsbook total (a hockey shootout goal is not counted)
        d["points"] = col("points") if "points" in g else col("home_pts") + col("away_pts")
        for c in GAME_EXTRAS:
            if c in g:
                d[c] = g[c]
        return pd.DataFrame(d)

    t = pd.concat([side(True), side(False)], ignore_index=True).sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    t["margin"] = t["pts_for"] - t["pts_against"]
    t["win"] = (t["margin"] > 0).astype(float).where(t["margin"].notna())
    t["cover"] = np.sign(t["margin"] + t["spread"])            # 1 cover, -1 no cover, 0 push
    t["over"] = np.sign(t["points"] - t["total"])              # 1 over, -1 under, 0 push
    t["win_prob"] = _implied(t["ml"]) / (_implied(t["ml"]) + _implied(t["opp_ml"]))     # no-vig moneyline price
    t["fav"] = (t["win_prob"] > 0.5).astype(float).where(t["win_prob"].notna())
    for name in ("div_game", "neutral"):
        if name in t:
            t[name] = pd.to_numeric(t[name], errors="coerce")
    if "roof" in t:
        t["dome"] = t["roof"].isin(["dome", "closed"]).astype(int)
    if "weekday" in t:
        for day in ("Thursday", "Monday", "Sunday", "Saturday"):
            t[day[:3].lower()] = (t["weekday"] == day).astype(int)

    # game number within the regular season, record so far, rest
    reg = t["kind"].isin(["reg"])
    t["game_num"] = reg.groupby([t["team"], t["season"]]).cumsum().where(reg)
    t["is_opener"] = (t["game_num"] == 1).astype(int)
    grp = [t["team"], t["season"]]
    played = reg.astype(int)
    won = (t["win"].fillna(0) * reg).astype(int)
    games_so_far = played.groupby(grp).cumsum() - played
    t["season_win_pct"] = (won.groupby(grp).cumsum() - won) / games_so_far.where(games_so_far >= 4)
    t["rest_days"] = t.groupby("team")["date"].diff().dt.days
    t["b2b"] = (t["rest_days"] == 1).astype(int)
    t["prev_win"] = t.groupby("team")["win"].shift(1)
    t["prev_cover"] = t.groupby("team")["cover"].shift(1)
    t["prev_over"] = t.groupby("team")["over"].shift(1)
    t["prev_margin"] = t.groupby("team")["margin"].shift(1)

    # the opponent's rest and record
    theirs = t[["game_id", "team", "rest_days", "b2b", "season_win_pct"]].rename(columns={
        "team": "opp", "rest_days": "opp_rest_days", "b2b": "opp_b2b", "season_win_pct": "opp_season_win_pct"})
    t = t.merge(theirs, on=["game_id", "opp"], how="left")
    t["rest_edge"] = t["rest_days"] - t["opp_rest_days"]

    # last meeting of this pair, any season
    t["pair"] = np.where(t["team"] < t["opp"], t["team"] + "|" + t["opp"], t["opp"] + "|" + t["team"])
    t = t.sort_values(["pair", "date", "game_id", "at_home"]).reset_index(drop=True)
    one = t[t["at_home"] == 1][["pair", "game_id", "date", "kind", "team", "margin", "over", "total", "points"]]  # one row per game
    one = one.sort_values(["pair", "date", "game_id"])
    one["prev_kind"] = one.groupby("pair")["kind"].shift(1)
    one["prev_date"] = one.groupby("pair")["date"].shift(1)
    one["prev_home_team"] = one.groupby("pair")["team"].shift(1)
    one["prev_home_margin"] = one.groupby("pair")["margin"].shift(1)
    one["prev_game_over"] = one.groupby("pair")["over"].shift(1)
    one["prev_total"] = one.groupby("pair")["total"].shift(1)
    one["prev_points"] = one.groupby("pair")["points"].shift(1)
    t = t.merge(one[["game_id", "prev_kind", "prev_date", "prev_home_team", "prev_home_margin", "prev_game_over",
                     "prev_total", "prev_points"]], on="game_id", how="left")
    t["total_vs_prev"] = t["total"] - t["prev_total"]
    t["prev_vs_opp_post"] = (t["prev_kind"] == "post").astype(int)
    t["prev_vs_opp_days"] = (t["date"] - t["prev_date"]).dt.days
    t["prev_vs_opp_margin"] = np.where(t["team"] == t["prev_home_team"], t["prev_home_margin"], -t["prev_home_margin"])

    # last season's playoff run
    post = t[t["kind"] == "post"]
    wins = post.groupby(["team", "season"])["win"].agg(po_wins="sum", po_games="count").reset_index()
    wins["season"] += 1
    t = t.merge(wins.rename(columns={"po_wins": "prev_po_wins", "po_games": "prev_po_games"}),
                on=["team", "season"], how="left")
    t[["prev_po_wins", "prev_po_games"]] = t[["prev_po_wins", "prev_po_games"]].fillna(0)
    regs = t[t["kind"] == "reg"].groupby(["team", "season"])["win"].agg(reg_wins="sum", reg_games="count").reset_index()
    regs["season"] += 1
    t = t.merge(regs.rename(columns={"reg_wins": "prev_reg_wins", "reg_games": "prev_reg_games"}),
                on=["team", "season"], how="left")
    t["prev_reg_win_pct"] = t["prev_reg_wins"] / t["prev_reg_games"]
    # how much the teams scored last regular season (average points in their games)
    scoring = t[t["kind"] == "reg"].groupby(["team", "season"])["points"].mean().reset_index()
    scoring["season"] += 1
    mine = scoring.rename(columns={"points": "prev_avg_points"})
    theirs = scoring.rename(columns={"team": "opp", "points": "opp_prev_avg_points"})
    t = t.merge(mine, on=["team", "season"], how="left").merge(theirs, on=["opp", "season"], how="left")
    t["total_vs_avg"] = t["total"] - (t["prev_avg_points"] + t["opp_prev_avg_points"]) / 2
    return t.sort_values(["date", "game_id", "at_home"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def binom_p_at_least(wins: int, n: int, p: float) -> float:
    """P(X >= wins) for X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    if wins <= 0:
        return 1.0
    lg = math.lgamma
    total = 0.0
    for k in range(wins, n + 1):
        total += math.exp(lg(n + 1) - lg(k + 1) - lg(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p))
    return min(total, 1.0)


@dataclass
class Result:
    n: int
    wins: int
    losses: int
    pushes: int
    roi: float
    last10: tuple[int, int, int]
    seasons: int
    p_vs_coin: float            # one-sided P(this many wins or more | 50% of decided bets)
    p_vs_breakeven: float       # same, against the 52.4% needed at -110
    by_season: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def win_pct(self) -> float:
        d = self.wins + self.losses
        return self.wins / d if d else float("nan")

    def record(self) -> str:
        return f"{self.wins}-{self.losses}" + (f"-{self.pushes}" if self.pushes else "")


def outcome(rows: pd.DataFrame, bet: str) -> pd.Series:
    """+1 win, -1 loss, 0 push, for each row."""
    if bet == "under":
        return -rows["over"]
    if bet == "over":
        return rows["over"]
    if bet == "ats":
        return rows["cover"]
    if bet == "ml":
        return rows["win"] * 2 - 1
    raise ValueError(f"unknown bet {bet!r}")


ODDS_COLUMN = {"under": "under_odds", "over": "over_odds", "ats": "spread_odds", "ml": "ml"}


def select(t: pd.DataFrame, query: str, bet: str) -> pd.DataFrame:
    rows = t.query(query, engine="python") if query.strip() else t
    need = ["total"] if bet in ("under", "over") else ["spread"] if bet == "ats" else ["ml"]
    rows = rows.dropna(subset=need)
    if bet in ("under", "over"):
        rows = rows.drop_duplicates("game_id")              # both teams may match; bet the game once
    return rows


def _decimal_profit(rows: pd.DataFrame, bet: str) -> pd.Series:
    """Profit per unit risked on a win, from the posted price where we have it, else -110."""
    col = ODDS_COLUMN[bet]
    odds = pd.to_numeric(rows[col], errors="coerce") if col in rows else pd.Series(np.nan, index=rows.index)
    odds = odds.where(odds.abs() >= 100)
    dec = pd.Series(np.where(odds > 0, odds / 100, 100 / odds.abs()), index=rows.index)
    return dec.fillna(PAYOUT)


def evaluate(t: pd.DataFrame, query: str, bet: str) -> Result:
    rows = select(t, query, bet).sort_values("date")
    res = outcome(rows, bet)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = len(rows)
    win_profit = _decimal_profit(rows, bet)
    decided = res != 0
    profit = np.where(res > 0, win_profit, -1.0)[decided.to_numpy()]
    roi = float(profit.sum() / (w + l)) if (w + l) else float("nan")
    breakeven = float((1 / (1 + win_profit[decided])).mean()) if (w + l) else BREAKEVEN
    last = res.tail(10)
    by = {int(s): (int((r > 0).sum()), int((r < 0).sum())) for s, r in res.groupby(rows["season"])}
    return Result(n, w, l, p, roi, (int((last > 0).sum()), int((last < 0).sum()), int((last == 0).sum())),
                  rows["season"].nunique(), binom_p_at_least(w, w + l, 0.5), binom_p_at_least(w, w + l, breakeven), by)


def leave_one_out(t: pd.DataFrame, conditions: list[str], bet: str) -> list[tuple[str, Result]]:
    """Drop each condition in turn: which ones carry the result?"""
    out = []
    for i, c in enumerate(conditions):
        rest = " and ".join(x for j, x in enumerate(conditions) if j != i)
        out.append((c, evaluate(t, rest, bet)))
    return out


# ---------------------------------------------------------------------------
# how often does chance produce a record like this?
# ---------------------------------------------------------------------------

# the vocabulary random angles are built from: (feature, kind of threshold)
VOCAB = [
    ("at_home", "bin"), ("spread", "num"), ("total", "num"), ("game_num", "game"), ("rest_days", "rest"),
    ("b2b", "bin"), ("prev_win", "bin"), ("prev_cover", "sign"), ("prev_over", "sign"),
    ("prev_po_wins", "po"), ("prev_vs_opp_post", "bin"), ("prev_vs_opp_days", "days"),
    ("prev_reg_win_pct", "pct"), ("is_opener", "bin"), ("rest_edge", "rest"), ("opp_rest_days", "rest"),
    ("season_win_pct", "pct"), ("opp_season_win_pct", "pct"), ("win_prob", "pct"), ("div_game", "bin"),
]


def _random_condition(t: pd.DataFrame, rng: random.Random) -> str:
    feat, kind = rng.choice([v for v in VOCAB if v[0] in t.columns])      # a league may lack a feature (no spreads in hockey)
    col = t[feat].dropna()
    if col.empty:                                   # a feature with no data in this frame
        return "at_home == 1"
    if kind == "bin":
        return f"{feat} == {rng.choice([0, 1])}"
    if kind == "sign":
        return f"{feat} == {rng.choice([-1, 1])}"
    if kind == "game":
        return f"{feat} == {rng.choice([1, 2, 3, 5, 10, 41, 82])}" if rng.random() < 0.5 else f"{feat} {rng.choice(['<', '>'])} {rng.choice([5, 10, 20, 40, 60])}"
    q = float(np.quantile(col, rng.uniform(0.1, 0.9)))
    return f"{feat} {rng.choice(['<', '>'])} {round(q * 2) / 2 if kind in ('num',) else round(q, 2)}"


def null_search(t: pd.DataFrame, bet: str, conditions: int, trials: int, n_min: int, n_max: int,
                seed: int = 0, base: str = "season > 2005") -> dict:
    """Generate random angles with `conditions` conditions and report how many land with
    n in [n_min, n_max] and how extreme the records are."""
    rng = random.Random(seed)
    records, kept = [], 0
    for _ in range(trials):
        conds = [_random_condition(t, rng) for _ in range(conditions)]
        q = " and ".join([base, *conds])
        try:
            rows = select(t, q, bet)
        except Exception:
            continue
        n = len(rows)
        if not n_min <= n <= n_max:
            continue
        res = outcome(rows, bet)
        w, l = int((res > 0).sum()), int((res < 0).sum())
        if w + l < n_min:
            continue
        kept += 1
        records.append((w, l, q))
    return {"trials": trials, "kept": kept, "records": records}


def summarize_null(records: list[tuple[int, int, str]], target_w: int, target_n: int) -> dict:
    """How many random angles with sample size ~target_n did at least as well as the target?"""
    close = [(w, l, q) for w, l, q in records if abs((w + l) - target_n) <= 3]
    at_least = [r for r in close if r[0] / (r[0] + r[1]) >= target_w / target_n]
    perfect = [r for r in close if r[1] == 0]
    return {"comparable": len(close), "as_good": len(at_least), "perfect": len(perfect),
            "best": sorted(close, key=lambda r: -(r[0] / (r[0] + r[1])))[:5]}


# ---------------------------------------------------------------------------
# an upcoming game
# ---------------------------------------------------------------------------

def season_of(when: pd.Timestamp) -> int:
    """Season label (start year) of a date: all three leagues start in August or later."""
    return when.year if when.month >= 8 else when.year - 1


def upcoming(games: pd.DataFrame, away: str, home: str, when, spread: float | None, total: float | None,
             kind: str = "reg", **extra) -> pd.DataFrame:
    """Both teams' feature rows for a game that has not been played, from the same code that
    builds the history (so every angle can be tested on it). `extra` sets any other column of the
    games frame, e.g. home_ml, away_ml, roof, wind, weekday."""
    when = pd.Timestamp(when)
    row = {"date": when, "away": away, "home": home, "kind": kind, "season": season_of(when),
           "home_spread_close": spread, "total_close": total, **extra}
    frame = pd.concat([games[games["date"] < when], pd.DataFrame([row])], ignore_index=True)
    t = team_games(frame, keep_unplayed=True)
    return t[t["game_id"] == t["game_id"].max()].reset_index(drop=True)


def active_teams(rows: pd.DataFrame, query: str) -> list[str]:
    """Which of the upcoming game's team rows satisfy the query."""
    hit = rows.query(query, engine="python") if query.strip() else rows
    return list(hit["team"])


def is_active(row: pd.Series, query: str) -> bool:
    return bool(len(pd.DataFrame([row]).query(query, engine="python")))

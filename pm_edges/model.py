"""Walk-forward models that ask one question: can anything we know before the game predict where the
closing line is wrong?

The market line is the baseline. The targets are *residuals against it*:

* side   = home margin + home spread        (0 when the line is exactly right; NBA and NFL)
* total  = points - closing total           (all three leagues)
* win    = home win minus the no-vig moneyline probability (NHL, and anywhere a moneyline exists)
* move_total / move_side = how far the closing line moved from the opening line. A model that predicts the
  move is a model of where the market is going, which is what matters when another market (Polymarket)
  lags the sportsbooks. Features for these use the opening line only.

A model that adds nothing predicts 0 for the residuals (or the market probability for `win`). Every
model is scored out of sample: for each test season it trains on all earlier seasons only, then predicts
that season. Skill is the reduction in squared error (log loss for `win`) versus the market, with a
standard error from the per-game differences, and a betting simulation at the posted prices.

Nothing here is tuned on the test seasons: model settings are fixed in advance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import angles as A

PAYOUT = A.PAYOUT
FORM = A.FORM_COLS

# features each task may use, all known before the game. Missing ones (no spread in hockey) are dropped per league.
COMMON = ["total_rel", "spread", "win_prob", "rest_days", "opp_rest_days", "rest_edge", "b2b", "opp_b2b", "game_num",
          "season_win_pct", "opp_season_win_pct", "prev_reg_win_pct", "prev_po_games", "is_post", "total_move",
          "spread_move", "total_vs_avg", "total_vs_prev", "prev_vs_opp_days"]
NFL_ONLY = ["dome", "wind", "temp", "div_game", "week", "thu", "mon", "neutral"]
FEATURES = COMMON + FORM + [f"opp_{c}" for c in FORM] + NFL_ONLY

# the same, but built from the opening line so they do not contain the move being predicted
OPEN_FEATURES = [c for c in COMMON if c not in ("total_rel", "spread", "win_prob", "total_move", "spread_move", "total_vs_avg",
                                                "total_vs_prev")] + ["total_open_rel", "spread_open", "total_open_vs_avg"] \
    + FORM + [f"opp_{c}" for c in FORM] + NFL_ONLY

TASKS = {
    "side": {"y": "ats_resid", "need": ["spread"], "features": FEATURES},
    "total": {"y": "tot_resid", "need": ["total"], "features": FEATURES},
    "win": {"y": "win_resid", "need": ["win_prob"], "features": FEATURES},
    "move_total": {"y": "total_move", "need": ["total_open", "total"], "features": OPEN_FEATURES},
    "move_side": {"y": "spread_move", "need": ["spread_open", "spread"], "features": OPEN_FEATURES},
}
MOVE_TASKS = ("move_total", "move_side")
# a larger opening-to-closing move than this is a parsing error in the source (the SBR tables have -1763 and 219.5)
MOVE_CAP = {"move_total": {"nba": 12.0, "nhl": 1.5, "nfl": 8.0}, "move_side": {"nba": 10.0, "nfl": 8.0}}
PRICE_COLS = ["over_odds", "under_odds", "ml", "opp_ml", "win_prob", "spread", "total"]
FIRST_TEST = {"nba": 2012, "nhl": 2012, "nfl": 2008}


def game_frame(t: pd.DataFrame) -> pd.DataFrame:
    """One row per game from the home team's side, with features, targets and prices."""
    h = t[t["at_home"] == 1].copy()
    h["home"], h["away"] = h["team"], h["opp"]
    h["is_post"] = (h["kind"] == "post").astype(int)
    h["spread_move"] = h["spread"] - h["spread_open"]
    h["total_move"] = h["total"] - h["total_open"]
    # totals drift upward over the years; give the models a total relative to the league's recent closing totals
    # (past games only), so a 2026 total of 230 is not read as "high" because 2012 totals were 195
    league_total = h["total"].shift(1).rolling(400, min_periods=100).mean()
    h["total_rel"] = h["total"] - league_total
    h["total_open_rel"] = h["total_open"] - league_total
    h["total_open_vs_avg"] = h["total_open"] - (h["prev_avg_points"] + h["opp_prev_avg_points"]) / 2
    for c in ("div_game", "neutral"):
        if c not in h:
            h[c] = np.nan
    return h.sort_values(["date", "game_id"]).reset_index(drop=True)


def usable_features(frame: pd.DataFrame, task: str = "total", min_fill: float = 0.3) -> list[str]:
    out = []
    for c in TASKS[task]["features"]:
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().mean() >= min_fill \
                and frame[c].nunique(dropna=True) > 1:
            out.append(c)
    return out


def _regressor(name: str):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=(10, 100, 1000, 10000, 100000)))
    if name == "gbm":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=200,
                                             l2_regularization=10.0, random_state=0)
    raise ValueError(name)


def task_data(frame: pd.DataFrame, task: str, lg: str = "nba") -> tuple[pd.DataFrame, list[str]]:
    spec = TASKS[task]
    d = frame.dropna(subset=[*spec["need"], spec["y"]])
    if task in MOVE_TASKS:
        d = d[d[spec["y"]].abs() <= MOVE_CAP[task].get(lg, 10.0)]
    d = d.reset_index(drop=True)
    return d, usable_features(d, task)


def walk_forward(frame: pd.DataFrame, task: str, lg: str, models=("ridge", "gbm"), first_test: int | None = None) -> pd.DataFrame:
    """Out-of-sample predictions: season s is predicted by models fit on games before s only."""
    d, feats = task_data(frame, task, lg)
    x = d[feats].apply(pd.to_numeric, errors="coerce")
    y = d[TASKS[task]["y"]]
    out = []
    for season in sorted(d["season"].unique()):
        if season < (first_test or FIRST_TEST[lg]):
            continue
        tr, te = (d["season"] < season).to_numpy(), (d["season"] == season).to_numpy()
        if tr.sum() < 1500 or te.sum() < 100:
            continue
        res = d.loc[te, ["season", "date", "away", "home", "kind", *[c for c in PRICE_COLS if c in d]]].copy()
        res["y"] = y[te].to_numpy()
        for name in models:
            res[f"p_{name}"] = _regressor(name).fit(x[tr], y[tr]).predict(x[te])
        if len(models) > 1:
            res["p_avg"] = res[[f"p_{n}" for n in models]].mean(axis=1)
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    model: str
    n: int
    gain: float             # reduction in mean squared error (or log loss) versus the market
    se: float
    corr: float | None = None
    by_season: dict[int, float] = field(default_factory=dict)

    @property
    def t(self) -> float:
        return self.gain / self.se if self.se else 0.0


def skill_regression(oos: pd.DataFrame, model: str) -> Skill:
    e0 = oos["y"] ** 2
    e1 = (oos["y"] - oos[f"p_{model}"]) ** 2
    d = e0 - e1
    by = {int(s): float(g.mean()) for s, g in d.groupby(oos["season"])}
    c = float(np.corrcoef(oos[f"p_{model}"], oos["y"])[0, 1]) if oos[f"p_{model}"].std() > 0 else 0.0
    return Skill(model, len(oos), float(d.mean()), float(d.std() / math.sqrt(len(d))), c, by)


def skill_win(oos: pd.DataFrame, model: str) -> Skill:
    """Log-loss gain of (market probability + predicted residual) over the market probability."""
    win = oos["win_prob"] + oos["y"]                        # y is win - win_prob
    p1 = np.clip(oos["win_prob"] + oos[f"p_{model}"], 0.02, 0.98)
    p0 = np.clip(oos["win_prob"], 0.02, 0.98)
    ll = lambda p: -(win * np.log(p) + (1 - win) * np.log(1 - p))
    d = ll(p0) - ll(p1)
    by = {int(s): float(g.mean()) for s, g in d.groupby(oos["season"])}
    return Skill(model, len(oos), float(d.mean()), float(d.std() / math.sqrt(len(d))), float(np.corrcoef(oos[f"p_{model}"], oos["y"])[0, 1]), by)


def market_calibration(frame: pd.DataFrame) -> dict:
    """How well the closing line predicts the result: slopes should be 1 and intercepts 0 if it is unbiased."""
    out = {}
    f = frame.dropna(subset=["spread", "margin"])
    if len(f) > 500:
        slope, icpt = np.polyfit(-f["spread"], f["margin"], 1)
        out["margin_on_spread"] = {"slope": float(slope), "intercept": float(icpt), "n": int(len(f)),
                                   "sd": float((f["margin"] + f["spread"]).std())}
    f = frame.dropna(subset=["total", "points"])
    slope, icpt = np.polyfit(f["total"], f["points"], 1)
    out["points_on_total"] = {"slope": float(slope), "intercept": float(icpt), "n": int(len(f)),
                              "sd": float((f["points"] - f["total"]).std()), "mean_resid": float((f["points"] - f["total"]).mean())}
    f = frame.dropna(subset=["win_prob", "win"])
    if len(f) > 500:
        f = f.assign(bin=pd.cut(f["win_prob"], [0, .3, .4, .45, .5, .55, .6, .7, 1.0]))
        g = f.groupby("bin", observed=True).agg(n=("win", "size"), predicted=("win_prob", "mean"), actual=("win", "mean"))
        out["win_calibration"] = [{"bin": str(i), **{k: float(v) for k, v in r.items()}} for i, r in g.iterrows()]
    return out


# ---------------------------------------------------------------------------
# betting simulation
# ---------------------------------------------------------------------------

def _profit(win: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.where(win > 0, dec, np.where(win < 0, -1.0, 0.0))


def _dec(odds: pd.Series) -> np.ndarray:
    o = pd.to_numeric(odds, errors="coerce")
    o = o.where(o.abs() >= 100)
    return np.where(o > 0, o / 100, 100 / o.abs()).astype(float)


def simulate(oos: pd.DataFrame, task: str, model: str, top_fracs=(0.5, 0.2, 0.1, 0.05)) -> list[dict]:
    """Act only on the games where the model disagrees with the market most (top fractions of |prediction|;
    the fractions are fixed, not optimised). Bets are settled at the posted price where we have it, else -110.
    For the move tasks there is no result to settle: the score is how far the line then moved in our favour
    (closing-line value, in points)."""
    p = oos[f"p_{model}"].to_numpy()
    y_all = oos["y"].to_numpy()
    rows = []
    for frac in top_fracs:
        cut = np.quantile(np.abs(p), 1 - frac)
        pick = np.abs(p) >= cut
        side = np.sign(p[pick])
        y = y_all[pick]
        if task in MOVE_TASKS:
            clv = side * y                               # the line moved toward the side we took
            moved = clv != 0
            rows.append({"top": frac, "bets": int(pick.sum()), "clv_pts": float(clv.mean()),
                         "clv_se": float(clv.std() / math.sqrt(len(clv))), "toward": float((clv[moved] > 0).mean()) if moved.any() else float("nan"),
                         "moved": float(moved.mean())})
            continue
        if task == "win":
            win_y = (oos["win_prob"].to_numpy() + y_all)[pick]
            win = np.where(side > 0, np.where(win_y > 0.5, 1, -1), np.where(win_y < 0.5, 1, -1))
            dec = np.where(side > 0, _dec(oos["ml"])[pick], _dec(oos["opp_ml"])[pick])
        elif task == "side":
            win = np.sign(y) * side                      # home cover and we bet home, or away cover and we bet away
            dec = np.full(pick.sum(), PAYOUT)
        else:
            win = np.sign(y) * side                      # over when the model expects points above the total
            over = _dec(oos["over_odds"])[pick] if "over_odds" in oos else np.full(pick.sum(), np.nan)
            under = _dec(oos["under_odds"])[pick] if "under_odds" in oos else np.full(pick.sum(), np.nan)
            dec = np.where(side > 0, over, under)
        dec = np.where(np.isnan(dec), PAYOUT, dec)
        profit = _profit(win, dec)
        decided = win != 0
        w, l = int((win > 0).sum()), int((win < 0).sum())
        n = w + l
        rows.append({"top": frac, "bets": n, "wins": w, "win_pct": w / n if n else float("nan"),
                     "roi": float(profit[decided].sum() / n) if n else float("nan"),
                     "roi_se": float(profit[decided].std() / math.sqrt(n)) if n > 1 else float("nan")})
    return rows


def importances(frame: pd.DataFrame, task: str, lg: str, last_seasons: int = 3, top: int = 12) -> list[dict]:
    """Permutation importance of the gradient-boosting model on the most recent test seasons."""
    from sklearn.inspection import permutation_importance
    d, feats = task_data(frame, task, lg)
    last = sorted(d["season"].unique())[-last_seasons:]
    tr, te = d[d["season"] < last[0]], d[d["season"].isin(last)]
    y = TASKS[task]["y"]
    m = _regressor("gbm").fit(tr[feats].apply(pd.to_numeric, errors="coerce"), tr[y])
    x_te = te[feats].apply(pd.to_numeric, errors="coerce")
    pi = permutation_importance(m, x_te, te[y], n_repeats=3, random_state=0, scoring="neg_mean_squared_error")
    order = np.argsort(-pi.importances_mean)[:top]
    return [{"feature": feats[i], "importance": float(pi.importances_mean[i]), "sd": float(pi.importances_std[i])} for i in order]


def evaluate_task(frame: pd.DataFrame, task: str, lg: str, models=("ridge", "gbm")) -> dict:
    oos = walk_forward(frame, task, lg, models)
    if oos.empty:
        return {"task": task, "available": False}
    scorer = skill_win if task == "win" else skill_regression
    res = {"task": task, "available": True, "n": int(len(oos)), "seasons": [int(oos["season"].min()), int(oos["season"].max())],
           "baseline_mse": float((oos["y"] ** 2).mean()) if task != "win" else None, "models": {}}
    if task == "win":
        w, p0 = oos["win_prob"] + oos["y"], np.clip(oos["win_prob"], 0.02, 0.98)
        res["baseline_loss"] = float((-(w * np.log(p0) + (1 - w) * np.log(1 - p0))).mean())
    for m in [*models, *(["avg"] if len(models) > 1 else [])]:
        s = scorer(oos, m)
        recent = [v for k, v in sorted(s.by_season.items())][-5:]
        res["models"][m] = {"gain": s.gain, "se": s.se, "t": s.t, "corr": s.corr,
                            "by_season": s.by_season, "recent5_gain": float(np.mean(recent)) if recent else None,
                            "seasons_positive": int(sum(v > 0 for v in s.by_season.values())), "seasons": len(s.by_season),
                            "bets": simulate(oos, task, m)}
    return res


# ---------------------------------------------------------------------------
# what to believe, and predictions for games not yet played
# ---------------------------------------------------------------------------

def placebo(frame: pd.DataFrame, task: str, lg: str, seeds=(0, 1, 2)) -> list[float]:
    """The same walk-forward run with each season's outcomes shuffled among its games. Any 'skill' found here is a
    bug or a leak, so t should sit near 0 (one value per shuffle)."""
    d, _ = task_data(frame, task, lg)
    y = TASKS[task]["y"]
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = d.copy()
        shuffled[y] = shuffled.groupby("season")[y].transform(lambda s, rng=rng: rng.permutation(s.to_numpy()))
        ev = evaluate_task(shuffled, task, lg, models=("ridge",))
        if ev.get("available"):
            out.append(ev["models"]["ridge"]["t"])
    return out


def status(task: str, ev: dict) -> dict:
    """A plain label for how far to trust a task's model, from its out-of-sample record."""
    if not ev.get("available"):
        return {"label": "not enough data", "tone": "muted"}
    m = ev["models"].get("avg") or next(iter(ev["models"].values()))
    frac = m["seasons_positive"] / max(m["seasons"], 1)
    steady = m["t"] >= 4 and frac >= 0.75 and (m["recent5_gain"] or 0) > 0
    if task in MOVE_TASKS:
        if steady:
            return {"label": "predicts the line move", "tone": "good",
                    "note": "usable only if the market you trade lags the sportsbooks' move"}
        return {"label": "no reliable skill", "tone": "muted"}
    top = next((b for b in m["bets"] if b["top"] == 0.1), None)
    if not steady:
        return {"label": "no skill beyond the line", "tone": "muted"}
    if top and top["roi"] > 0:
        return {"label": "beats the line and pays", "tone": "good"}
    return {"label": "beats the line, but not enough to pay the vig", "tone": "warn"}


def fit_final(frame: pd.DataFrame, task: str, lg: str, models=("ridge", "gbm")) -> dict:
    d, feats = task_data(frame, task, lg)
    x = d[feats].apply(pd.to_numeric, errors="coerce")
    y = d[TASKS[task]["y"]]
    return {"feats": feats, "models": {n: _regressor(n).fit(x, y) for n in models}}


def predict_final(fitted: dict, rows: pd.DataFrame) -> np.ndarray:
    x = rows.reindex(columns=fitted["feats"]).apply(pd.to_numeric, errors="coerce")
    return np.mean([m.predict(x) for m in fitted["models"].values()], axis=0)


def game_rows(upcoming_teams: pd.DataFrame) -> pd.DataFrame:
    """Home-side rows with the derived columns the models expect, from A.upcoming_many()."""
    return game_frame(upcoming_teams)

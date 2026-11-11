"""NBA player-prop features from ESPN box scores: trailing form, new teammate absences, opponent allowed rates, pace, game script.
Everything for a player-game uses only games strictly before it (plus the game's own pre-tip lines when supplied)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

STATS = ["min", "pts", "reb", "ast", "tpm", "stl", "blk", "fga", "fta", "tpa", "tov"]
FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}


def load_player_rows(path: str = "data/history/box/player_box.pkl") -> pd.DataFrame:
    p = pd.read_pickle(path)
    p = p[(p.stype.isin([2, 3, 4, 5]))].copy()
    p["team"] = p.team.replace(FIX)
    p["date"] = pd.to_datetime(p.date)
    p = p[p.player_id.notna()]
    p["reb"] = p.reb.fillna(p.oreb.fillna(0) + p.dreb.fillna(0))
    p["tpm"] = p.tpm if "tpm" in p else np.nan
    p["played"] = (p["min"].fillna(0) > 0) & ~p.dnp
    return p


def team_absences(p: pd.DataFrame, window: int = 10, min_games: int = 4, min_baseline: float = 12.0) -> pd.DataFrame:
    """Per team-game: baseline output of rotation players (>= min_games of the last `window` team games, baseline minutes >= min_baseline)
    who did not play. Columns: team, game_id, miss_min, miss_pts, miss_ast, miss_reb, miss_starters, miss_n."""
    g = p[p.played].groupby(["team", "game_id"]).date.first().reset_index().sort_values(["team", "date", "game_id"])
    by_game = {k: d for k, d in p[p.played].groupby(["team", "game_id"])}
    rows = []
    for team, tg in g.groupby("team"):
        hist: deque = deque(maxlen=window)                       # each entry: {player_id: (min, pts, ast, reb, starter)}
        for gid in tg.game_id:
            d = by_game[(team, gid)]
            played_now = set(d.player_id)
            cnt, tot = defaultdict(int), defaultdict(lambda: np.zeros(5))
            for entry in hist:
                for pid, v in entry.items():
                    cnt[pid] += 1; tot[pid] += v
            mm = mp = ma = mr = ms = mn = 0.0
            for pid, c in cnt.items():
                if c >= min_games and pid not in played_now:
                    base = tot[pid] / c
                    if base[0] >= min_baseline:
                        mm += base[0]; mp += base[1]; ma += base[2]; mr += base[3]; ms += base[4] >= 0.5; mn += 1
            rows.append({"team": team, "game_id": gid, "miss_min": mm, "miss_pts": mp, "miss_ast": ma, "miss_reb": mr, "miss_starters": ms, "miss_n": mn})
            hist.append({r.player_id: np.array([r["min"], r.pts, r.ast, r.reb, float(r.starter)]) for _, r in d.iterrows()})
    return pd.DataFrame(rows)


def opp_and_pace(path: str = "data/history/box/team_box.pkl") -> pd.DataFrame:
    """Per team-game: trailing (EWM, half-life 15 games) points/reb/ast/3PM the team ALLOWS, and the team's own pace; shifted so the game itself is excluded."""
    t = pd.read_pickle(path); t["team"] = t.team.replace(FIX); t["date"] = pd.to_datetime(t.date)
    t["pts"] = 2 * t.fgm + t.tpm + t.ftm; t["reb"] = t.oreb + t.dreb; t["poss"] = t.fga + 0.44 * t.fta + t.tov - t.oreb
    o = t.merge(t[["game_id", "team", "pts", "reb", "ast", "tpm", "poss"]].rename(columns={"team": "opp", "pts": "a_pts", "reb": "a_reb", "ast": "a_ast", "tpm": "a_tpm", "poss": "o_poss"}), on="game_id")
    o = o[o.team != o.opp].sort_values(["team", "date", "game_id"])
    o["pace"] = (o.poss + o.o_poss) / 2
    out = o[["game_id", "team", "opp", "date"]].copy()
    gp = o.groupby("team")
    for c in ("a_pts", "a_reb", "a_ast", "a_tpm", "pace"):
        out["allow_" + c.replace("a_", "") if c.startswith("a_") else c] = gp[c].transform(lambda s: s.shift(1).ewm(halflife=15, min_periods=5).mean())
    return out


def player_frame(p: pd.DataFrame, lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game, player who played). Features are trailing; targets are the game's actual stats. `lines`: game_id, home, away, home_spread_close, total_close."""
    d = p[p.played].sort_values(["player_id", "date", "game_id"]).copy()
    gp = d.groupby("player_id")
    for s in STATS:
        for hl in (6, 20):
            d[f"f_{s}_{hl}"] = gp[s].transform(lambda x: x.shift(1).ewm(halflife=hl, min_periods=3).mean())
    d["f_start_5"] = gp.starter.transform(lambda x: x.shift(1).astype(float).rolling(5, min_periods=2).mean())
    d["f_games"] = gp.cumcount()
    d["f_rest"] = (d.date - gp.date.shift(1)).dt.days.clip(upper=10)
    d["f_min_sd"] = gp["min"].transform(lambda x: x.shift(1).rolling(15, min_periods=5).std())
    ab = team_absences(p)
    d = d.merge(ab, on=["team", "game_id"], how="left")
    op = opp_and_pace()
    d = d.merge(op[["game_id", "team", "opp", "allow_pts", "allow_reb", "allow_ast", "allow_tpm", "pace"]], on=["game_id", "team"], how="left")
    opp_pace = op[["game_id", "team", "pace"]].rename(columns={"team": "opp", "pace": "opp_pace"})
    d = d.merge(opp_pace, on=["game_id", "opp"], how="left"); d["exp_pace"] = (d.pace + d.opp_pace) / 2
    if lines is not None:
        l = lines.set_index("game_id")
        d["total_close"] = d.game_id.map(l.total_close)
        hs = d.game_id.map(l.home_spread_close); ishome = d.team == d.game_id.map(l.home)
        d["team_spread"] = np.where(ishome, hs, -hs)           # negative = this player's team is favoured
        d["abs_spread"] = d.team_spread.abs()
        d["team_total"] = (d.total_close - d.team_spread) / 2
        d["home"] = ishome.astype(float)
    return d


# ---------------------------------------------------------------------------------------------- production model
import re
import unicodedata

from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "tpm": ["tpm"], "stl": ["stl"], "blk": ["blk"],
           "pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
MARKET_STAT = {"POINTS": "pts", "REBOUNDS": "reb", "ASSISTS": "ast", "THREE_POINTERS_MADE": "tpm", "STEALS": "stl", "BLOCKS": "blk",
               "POINTS_REBOUNDS_ASSISTS": "pra", "POINTS_REBOUNDS": "pr", "POINTS_ASSISTS": "pa", "REBOUNDS_ASSISTS": "ra"}
ABS = ["miss_min", "miss_pts", "miss_ast", "miss_reb", "miss_starters", "miss_n"]


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    return " ".join(t for t in s.split() if t not in ("jr", "sr", "ii", "iii", "iv", "v"))


def feature_cols(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c.startswith("f_")] + ABS


def target_series(d: pd.DataFrame, stat: str) -> pd.Series:
    return d[TARGETS[stat]].sum(axis=1)


def base_of(d: pd.DataFrame, stat: str, hl: int = 20) -> pd.Series:
    """The player's own trailing level for this stat (sum of component EWMs), floored so ratios stay finite."""
    return d[[f"f_{c}_{hl}" for c in TARGETS[stat]]].sum(axis=1).clip(lower=0.3)


def ratio_cols(d: pd.DataFrame) -> pd.DataFrame:
    """Scale-free versions of the form features, so the trees learn adjustments rather than levels (trees cannot extrapolate levels)."""
    r = pd.DataFrame(index=d.index)
    for s in STATS:
        r[f"r_{s}_6v20"] = d[f"f_{s}_6"] / d[f"f_{s}_20"].clip(lower=0.3)
    for s in ("fga", "fta", "tpa", "ast", "reb", "pts", "tov"):                      # per-minute usage rates (scale-free)
        r[f"u_{s}"] = d[f"f_{s}_20"] / d.f_min_20.clip(lower=5)
    r["f_min_20"] = d.f_min_20; r["f_start_5"] = d.f_start_5; r["f_rest"] = d.f_rest; r["f_min_sd"] = d.f_min_sd; r["f_games"] = d.f_games.clip(upper=200)
    for c in ABS:
        r[c] = d[c]
    return r


def fit_stat(d: pd.DataFrame, stat: str, cols: list[str] | None = None):
    """Predict y / base with sample weight base (a quasi-Poisson fit on the ratio); mean = base * prediction."""
    b = base_of(d, stat); y = target_series(d, stat)
    m = HistGradientBoostingRegressor(loss="squared_error", max_iter=200, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=120,
                                      l2_regularization=2.0, random_state=0)
    return m.fit(ratio_cols(d), y / b, sample_weight=b)


def predict_mean(model, d: pd.DataFrame, stat: str) -> np.ndarray:
    return (base_of(d, stat) * np.clip(model.predict(ratio_cols(d)), 0.05, 3.0)).values


def fit_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative-binomial overdispersion by moments: Var = mu + alpha mu^2 (alpha >= 0.005)."""
    mu = np.clip(mu, 0.05, None)
    return float(max(0.005, np.sum(((y - mu) ** 2 - mu) * mu ** 2) / np.sum(mu ** 4)))


def p_over(mu, alpha: float, line: float):
    """P(stat > line) for a half-point line under NB(mean mu, overdispersion alpha)."""
    mu = np.clip(np.asarray(mu, float), 0.05, None); r = 1.0 / alpha
    k = int(np.floor(line))                                             # over wins when stat >= k+1
    return 1.0 - nbinom.cdf(k, r, r / (r + mu))


def lines_probs(mu, y, alpha: float, offs=(-0.7, -0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)):
    """Raw P(over) and outcomes for half-point lines placed around each prediction (as a market would post them)."""
    P, O = [], []
    for off in offs:
        line = np.floor(mu * (1 + off)) + 0.5
        P.append(np.array([p_over(mu[i], alpha, line[i]) for i in range(len(mu))])); O.append((np.asarray(y) > line).astype(float))
    return np.concatenate(P), np.concatenate(O)


class Calibrator:
    """Isotonic map raw NB P(over) -> realised frequency, valid only inside the raw-probability range seen in the fit; outside it the raw value is kept
    (scikit-learn's isotonic regression returns junk beyond its support). Output is clipped to [0.01, 0.99]."""

    def __init__(self, iso, lo: float, hi: float):
        self.iso, self.lo, self.hi = iso, lo, hi

    def predict(self, p):
        p = np.asarray(p, float)
        out = np.where((p >= self.lo) & (p <= self.hi), self.iso.predict(np.clip(p, self.lo, self.hi)), p)
        return np.clip(out, 0.01, 0.99)


def fit_iso(mu, y, alpha: float) -> Calibrator:
    from sklearn.isotonic import IsotonicRegression
    p, o = lines_probs(mu, y, alpha)
    iso = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(p, o)
    return Calibrator(iso, float(np.quantile(p, 0.005)), float(np.quantile(p, 0.995)))


def train_all(d: pd.DataFrame, holdout_days: int = 300):
    """Fit one model per stat on everything, plus alpha and an isotonic calibrator from a model trained before the last `holdout_days` and scored on them."""
    cols = feature_cols(d); cut = d.date.max() - pd.Timedelta(days=holdout_days)
    old, new = d[d.date <= cut], d[d.date > cut]
    out = {"cols": cols, "models": {}, "alpha": {}, "iso": {}}
    for stat in TARGETS:
        m_old = fit_stat(old, stat, cols)
        mu_new, y_new = predict_mean(m_old, new, stat), target_series(new, stat).values
        out["alpha"][stat] = fit_alpha(y_new, mu_new)
        out["iso"][stat] = fit_iso(mu_new, y_new, out["alpha"][stat])
        out["models"][stat] = fit_stat(d, stat, cols)
    return out

"""Do player absences explain the closing-line error, and how much of the open-to-close move? (needs 02 and 03 first)"""
import numpy as np, pandas as pd
S = "data/model/nba/"
m = pd.read_pickle(S + "merged1.pkl"); A = pd.read_pickle(S + "absence.pkl").copy()
A["date"] = A.game_id.map(m.set_index("game_id").date)
A = A.dropna(subset=["S_missing"]).sort_values(["team", "date"]); A["d_missing"] = A.groupby("team").S_missing.diff()
h = A.rename(columns={"team": "home", "S_missing": "h_miss", "d_missing": "h_dmiss"})[["game_id", "home", "h_miss", "h_dmiss"]]
a = A.rename(columns={"team": "away", "S_missing": "a_miss", "d_missing": "a_dmiss"})[["game_id", "away", "a_miss", "a_dmiss"]]
d = m.merge(h, on=["game_id", "home"]).merge(a, on=["game_id", "away"]).dropna(subset=["h_miss", "a_miss", "h_dmiss", "a_dmiss"])
d["adj"] = d.a_miss - d.h_miss; d["dadj"] = d.a_dmiss - d.h_dmiss
d["err"] = d.margin - d.line; d["move"] = d.line - (-d.home_spread_open)
d = d[d.date >= "2022-10-01"]
def ols(y, X):
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    return b, np.sqrt(np.diag(np.linalg.inv(X.T @ X) * r.var(ddof=X.shape[1]))), 1 - r.var() / y.var()
b, se, _ = ols(d.err.values, np.c_[d.adj, d.dadj])
print(f"closing error ~ absences (n={len(d)}): level {b[1]:+.3f} (t={b[1]/se[1]:+.1f})  change {b[2]:+.3f} (t={b[2]/se[2]:+.1f})")
e = d.dropna(subset=["move"])
b, se, r2 = ols(e.move.values, np.c_[e.adj, e.dadj])
print(f"open->close move ~ absences (n={len(e)}): level {b[1]:+.3f} (t={b[1]/se[1]:+.1f})  change {b[2]:+.3f} (t={b[2]/se[2]:+.1f})  R^2={r2:.3f}")
e = e.assign(pred=b[0] + b[1] * e.adj + b[2] * e.dadj)
for th in (0.3, 0.5, 0.75, 1.0):
    x = e[e.pred.abs() >= th]; g = np.sign(x.pred) * x.move
    print(f"perfect-foresight availability, |predicted move| >= {th}: n={len(x)} mean gain from the open {g.mean():+.2f} pts")

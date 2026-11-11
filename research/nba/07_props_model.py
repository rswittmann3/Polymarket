"""Player-prop model, walk-forward by season. Compares trailing average, form model, +teammate absences, +game script/opponent/pace.
Metric: Poisson deviance / MAE of each stat (lower is better). There are no historical prop lines, so this shows how much predictive
information each block adds, NOT edge vs the market (that is what the forward prop log is for)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from pm_edges import nba_props as NP

m = pd.read_pickle("data/model/nba/merged1.pkl")[["game_id", "home", "away", "home_spread_close", "total_close"]]
p = NP.load_player_rows()
d = NP.player_frame(p, m)
d["sea"] = np.where(d.date.dt.month >= 9, d.date.dt.year, d.date.dt.year - 1)
d = d[(d.f_games >= 5) & d.f_min_20.notna() & (d.f_min_20 >= 12)].copy()
d["pra"] = d.pts + d.reb + d.ast
d.to_pickle("data/model/nba/props_frame.pkl")
FORM = [c for c in d.columns if c.startswith("f_")]
ABS = ["miss_min", "miss_pts", "miss_ast", "miss_reb", "miss_starters", "miss_n"]
CTX = ["allow_pts", "allow_reb", "allow_ast", "allow_tpm", "exp_pace", "total_close", "team_spread", "abs_spread", "team_total", "home"]
sets = {"form": FORM, "form+absences": FORM + ABS, "form+absences+context": FORM + ABS + CTX}
def pdev(y, mu):
    mu = np.clip(mu, 1e-3, None); y = np.asarray(y, float)
    return 2 * np.mean(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
res = []
for tgt, base in (("pts", "f_pts_20"), ("reb", "f_reb_20"), ("ast", "f_ast_20"), ("tpm", "f_tpm_20"), ("pra", None), ("min", "f_min_20")):
    for test in (2023, 2024, 2025):
        tr, te = d[d.sea < test], d[d.sea == test]
        te = te[te.total_close.notna()]                                   # same games for every model
        trl = tr[tr.total_close.notna() | (tr.sea < 2022)]                 # context columns are NaN where no lines; HGB handles that
        row = {"target": tgt, "test": test, "n": len(te)}
        b = te[base] if base else te.f_pts_20 + te.f_reb_20 + te.f_ast_20
        row["naive_dev"] = pdev(te[tgt], b); row["naive_mae"] = (te[tgt] - b).abs().mean()
        for name, cols in sets.items():
            mdl = HGB(loss="poisson", max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0).fit(tr[cols], tr[tgt])
            mu = mdl.predict(te[cols]); row[name + "_dev"] = pdev(te[tgt], mu); row[name + "_mae"] = (te[tgt] - mu).abs().mean()
        res.append(row)
R = pd.DataFrame(res); pd.set_option("display.width", 220)
g = R.groupby("target")[[c for c in R.columns if c.endswith("_dev") or c.endswith("_mae")]].mean()
for k in ("form", "form+absences", "form+absences+context"):
    g[k + "_gain%"] = 100 * (g.naive_dev - g[k + "_dev"]) / g.naive_dev
print(g.round(4).to_string())

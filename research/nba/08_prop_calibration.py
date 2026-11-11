"""Reliability of the prop model's P(over): train < 2025 (alpha fitted on the 2024 season with a model trained < 2024), test on 2025.
For each test player-game and several lines around the model mean, compare predicted P(over) with the actual frequency, by predicted-probability bucket."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pm_edges import nba_props as NP

d = pd.read_pickle("data/model/nba/props_frame.pkl")
rows = []
for stat in ("pts", "reb", "ast", "tpm", "pra"):
    tr24, te24, te = d[d.sea < 2024], d[d.sea == 2024], d[d.sea == 2025]
    a = NP.fit_alpha(NP.target_series(te24, stat).values, NP.predict_mean(NP.fit_stat(tr24, stat), te24, stat))
    m = NP.fit_stat(d[d.sea < 2025], stat); mu = NP.predict_mean(m, te, stat); y = NP.target_series(te, stat).values
    for off in (-0.15, 0.0, 0.15):                                         # lines below / at / above the mean, as a market would post
        line = np.floor(mu * (1 + off)) + 0.5
        p = np.array([NP.p_over(mu[i], a, line[i]) for i in range(len(mu))])
        rows.append(pd.DataFrame({"stat": stat, "alpha": a, "p": p, "o": (y > line).astype(float), "off": off}))
R = pd.concat(rows)
print("alpha by stat:", R.groupby("stat").alpha.first().round(3).to_dict())
R["bucket"] = pd.cut(R.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
t = R.groupby("bucket", observed=True).agg(n=("o", "size"), predicted=("p", "mean"), actual=("o", "mean")); t["gap"] = t.actual - t.predicted
print(t.round(3).to_string())
print("\nby stat (all lines): Brier model %s" % {s: round(float(((g.p - g.o) ** 2).mean()), 4) for s, g in R.groupby("stat")})
print("mean predicted vs actual, by stat:", {s: (round(float(g.p.mean()), 3), round(float(g.o.mean()), 3)) for s, g in R.groupby("stat")})

# ---- Platt scaling fitted on the 2024 season (model trained < 2024), evaluated on 2025 ----
from sklearn.linear_model import LogisticRegression
def lines_probs(mu, y, a, offs=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)):
    P, O = [], []
    for off in offs:
        line = np.floor(mu * (1 + off)) + 0.5
        P.append(np.array([NP.p_over(mu[i], a, line[i]) for i in range(len(mu))])); O.append((y > line).astype(float))
    return np.concatenate(P), np.concatenate(O)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
out = []
for stat in ("pts", "reb", "ast", "tpm", "pra"):
    tr24, te24, te = d[d.sea < 2024], d[d.sea == 2024], d[d.sea == 2025]
    m24 = NP.fit_stat(tr24, stat); mu24 = NP.predict_mean(m24, te24, stat); y24 = NP.target_series(te24, stat).values
    a = NP.fit_alpha(y24, mu24); p24, o24 = lines_probs(mu24, y24, a)
    cal = LogisticRegression(C=1e4).fit(lg(p24)[:, None], o24)
    m = NP.fit_stat(d[d.sea < 2025], stat); mu = NP.predict_mean(m, te, stat); y = NP.target_series(te, stat).values
    p, o = lines_probs(mu, y, a); pc = cal.predict_proba(lg(p)[:, None])[:, 1]
    ll = lambda q: -np.mean(o * np.log(np.clip(q, 1e-6, 1)) + (1 - o) * np.log(np.clip(1 - q, 1e-6, 1)))
    ext = (p < 0.2) | (p > 0.8)
    out.append(dict(stat=stat, slope=float(cal.coef_[0][0]), icpt=float(cal.intercept_[0]), logloss_raw=ll(p), logloss_cal=ll(pc), brier_raw=np.mean((p - o) ** 2), brier_cal=np.mean((pc - o) ** 2),
                    mean_pred_raw=p.mean(), mean_pred_cal=pc.mean(), mean_act=o.mean(), tail_n=int(ext.sum()), tail_pred_raw=p[ext].mean(), tail_pred_cal=pc[ext].mean(), tail_act=o[ext].mean()))
print("\nPlatt calibration (fit 2024, test 2025):"); print(pd.DataFrame(out).round(4).to_string(index=False))

from sklearn.isotonic import IsotonicRegression
out = []
for stat in ("pts", "reb", "ast", "tpm", "pra"):
    tr24, te24, te = d[d.sea < 2024], d[d.sea == 2024], d[d.sea == 2025]
    m24 = NP.fit_stat(tr24, stat); mu24 = NP.predict_mean(m24, te24, stat); y24 = NP.target_series(te24, stat).values
    a = NP.fit_alpha(y24, mu24); p24, o24 = lines_probs(mu24, y24, a)
    iso = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(p24, o24)
    m = NP.fit_stat(d[d.sea < 2025], stat); mu = NP.predict_mean(m, te, stat); y = NP.target_series(te, stat).values
    p, o = lines_probs(mu, y, a); pc = iso.predict(p)
    ll = lambda q: -np.mean(o * np.log(np.clip(q, 1e-6, 1)) + (1 - o) * np.log(np.clip(1 - q, 1e-6, 1)))
    for lo, hi in ((0, .15), (.15, .3), (.7, .85), (.85, 1)):
        s = (p >= lo) & (p < hi)
        if s.sum() > 200: out.append(dict(stat=stat, bucket=f"{lo}-{hi}", n=int(s.sum()), pred_raw=p[s].mean(), pred_iso=pc[s].mean(), actual=o[s].mean()))
    out.append(dict(stat=stat, bucket="ALL logloss", n=len(p), pred_raw=ll(p), pred_iso=ll(pc), actual=np.nan))
print("\nIsotonic calibration (fit 2024, test 2025) -- tail buckets and overall log-loss:"); print(pd.DataFrame(out).round(4).to_string(index=False))

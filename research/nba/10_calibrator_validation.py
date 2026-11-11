"""Out-of-sample check of the prop model's support-aware isotonic calibrator: model < 2024, alpha + calibrator fitted on 2024, scored on 2025 at lines 30-170% of the mean."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pm_edges import nba_props as NP
d = pd.read_pickle("data/model/nba/props_frame.pkl")
res = []
for stat in ("pts", "reb", "ast", "tpm", "pra"):
    tr, cal_s, te = d[d.sea < 2024], d[d.sea == 2024], d[d.sea == 2025]
    m = NP.fit_stat(tr, stat); mu24 = NP.predict_mean(m, cal_s, stat); y24 = NP.target_series(cal_s, stat).values; a = NP.fit_alpha(y24, mu24); cal = NP.fit_iso(mu24, y24, a)
    m2 = NP.fit_stat(d[d.sea < 2025], stat); mu = NP.predict_mean(m2, te, stat); y = NP.target_series(te, stat).values
    p, o = NP.lines_probs(mu, y, a); pc = cal.predict(p)
    ll = lambda q: -np.mean(o * np.log(np.clip(q, 1e-6, 1)) + (1 - o) * np.log(np.clip(1 - q, 1e-6, 1)))
    for name, mask in (("p<.1", p < .1), ("p>.9", p > .9), ("mid .3-.7", (p >= .3) & (p <= .7))):
        if mask.sum() > 300:
            res.append(dict(stat=stat, bucket=name, n=int(mask.sum()), raw=p[mask].mean(), cal=pc[mask].mean(), actual=o[mask].mean()))
    res.append(dict(stat=stat, bucket="ALL logloss", n=len(p), raw=ll(p), cal=ll(pc), actual=np.nan))
print(pd.DataFrame(res).round(4).to_string(index=False))

"""Do games with a big closing spread go over the closing total more often than the price implies? Cover RATES (pushes excluded) and mean points, by season, source, threshold."""
import numpy as np, pandas as pd
D = pd.read_pickle("data/model/nba/sched_feats.pkl")
D = D[D.kind.isin(["reg", "post"])].copy()
D["aspread"] = D.home_spread_close.abs()
D["over"] = np.where(D.err_t > 0, 1.0, np.where(D.err_t < 0, 0.0, np.nan))
def rate(x): x = x.dropna(); return len(x), x.mean(), x.std() / np.sqrt(len(x))
print("overall over rate: n=%d  %.4f (se %.4f); mean err_t %+.3f" % (*rate(D.over), D.err_t.mean()))
for th in (4, 6, 8, 10, 12):
    a = D[D.aspread >= th]; n, m, se = rate(a.over)
    print(f"|spread| >= {th:2d}: n={n:5d} over {m:.4f} (se {se:.4f}, z vs .5 = {(m-.5)/se:+.2f}) mean err_t {a.err_t.mean():+.2f}")
a = D[D.aspread >= 6]
print("\n|spread|>=6 by season (over rate / mean pts error / n):")
t = a.groupby("yr").agg(n=("over", "size"), over=("over", "mean"), err=("err_t", "mean")); print(t.round(3).T.to_string())
print("seasons with over rate > .5: %d of %d" % ((t.over > .5).sum(), len(t)))
print("\nby source:"); print(a.groupby("source").agg(n=("over", "size"), over=("over", "mean"), err=("err_t", "mean")).round(3))
print("\ninteraction with total level (|spread|>=6):"); a2 = a.assign(tl=pd.qcut(a.total_close, 4)); print(a2.groupby("tl", observed=True).agg(n=("over", "size"), over=("over", "mean"), err=("err_t", "mean")).round(3))
print("\nsmall-spread control (|spread|<3):"); b = D[D.aspread < 3]; print("n=%d over %.4f err %+.2f" % (len(b), b.over.mean(), b.err_t.mean()))
# opening-total version (only where an opening total exists and differs from NaN)
o = D[D.total_open.notna() & (D.aspread >= 6)].copy(); o["e_open"] = o.points - o.total_open
print("\nvs OPENING total, |spread|>=6: n=%d mean points above open %+.2f; vs close %+.2f" % (len(o), o.e_open.mean(), o.err_t.mean()))
# last 6 seasons only (the regime that matters)
r = a[a.yr >= 2021]; n, m, se = rate(r.over); print("2021+: n=%d over %.4f (se %.4f)" % (n, m, se))
# breakeven at -110: need 52.38%
print("\nbreakeven at -110 is 52.38%; observed over rate for |spread|>=6 across all years: %.4f" % a.over.mean())

"""(a) referee crew effect on total error, walk-forward  (b) shooting-luck regression (recent eFG/3P% vs longer baseline) vs spread/total error
(c) does the ratings model's total carry information beyond the closing total?   Needs 01/02 (merged1.pkl) and the box-score pulls."""
import numpy as np, pandas as pd
from math import erf, sqrt
S = "data/model/nba/"
m = pd.read_pickle(S + "merged1.pkl"); m["date"] = pd.to_datetime(m.date)
m["err_s"] = m.margin - m.line; m["err_t"] = m.points - m.total_close
gi = pd.read_pickle("data/history/box/game_info.pkl"); tb = pd.read_pickle("data/history/box/team_box.pkl")
def t_of(y, x):
    d = pd.DataFrame({"y": y, "x": x}).dropna(); X = np.c_[np.ones(len(d)), d.x]; b = np.linalg.lstsq(X, d.y, rcond=None)[0]; r = d.y.values - X @ b
    Xi = np.linalg.inv(X.T @ X); V = Xi @ (X.T * r**2) @ X @ Xi; return b[1], b[1] / np.sqrt(V[1, 1]), len(d)
def line(name, y, x): b, t, n = t_of(y, x); print(f"  {name:48s} coef {b:+.3f}  t {t:+.2f}  n={n}")

print("(c) ratings model vs closing total (gap = model_total - total_close):")
m["tgap"] = m.model_total - m.total_close
line("total error ~ total gap (all)", m.err_t, m.tgap)
for yr, d in m.groupby("sea"): line(f"   season {yr}", d.err_t, d.tgap)
line("total error ~ pace gap (model_pace - league mean pace)", m.err_t, m.model_pace - m.model_pace.mean())

print("\n(a) referee crew, walk-forward (crew score = mean of each ref's shrunk prior average total error, shrink k=40 games):")
gi = gi.assign(refs=gi.officials.str.split("|"))
x = m.merge(gi[["game_id", "refs"]], on="game_id").sort_values("date").reset_index(drop=True)
sums, cnt, score = {}, {}, []
for _, r in x.iterrows():
    refs = [f for f in r.refs if f]
    score.append(np.mean([sums.get(f, 0) / (cnt.get(f, 0) + 40) for f in refs]) if refs else np.nan)
    for f in refs: sums[f] = sums.get(f, 0) + r.err_t; cnt[f] = cnt.get(f, 0) + 1
x["crew"] = score; x["crew_n"] = [np.mean([cnt.get(f, 0) for f in r if f]) if r else 0 for r in x.refs]
y = x[x.crew_n >= 60]; line("total error ~ crew score (refs with >= 60 prior games)", y.err_t, y.crew)
for yr, d in y.groupby("sea"): line(f"   season {yr}", d.err_t, d.crew)
print("  distinct refs:", len(cnt), "| score SD %.3f pts" % y.crew.std())

print("\n(b) shooting luck: trailing-8 minus trailing-40 offensive eFG% / opponent-allowed eFG% (ordered by date, current game excluded):")
tb = tb.merge(gi[["game_id"]], on="game_id"); tb["date"] = pd.to_datetime(tb.date)
tb["efg"] = (tb.fgm + 0.5 * tb.tpm) / tb.fga
o = tb.merge(tb[["game_id", "team", "efg", "tpa", "tpm"]].rename(columns={"team": "opp", "efg": "opp_efg", "tpa": "opp_tpa", "tpm": "opp_tpm"}), on="game_id"); o = o[o.team != o.opp].sort_values(["team", "date"])
gp = o.groupby("team")
for k in (8, 40):
    o[f"off{k}"] = gp.efg.transform(lambda s: s.shift(1).rolling(k, min_periods=k // 2).mean()); o[f"def{k}"] = gp.opp_efg.transform(lambda s: s.shift(1).rolling(k, min_periods=k // 2).mean())
o["off_luck"] = o.off8 - o.off40; o["def_luck"] = o.def8 - o.def40       # + off_luck: hot shooting lately; + def_luck: opponents shot well against them lately
H = o[o.home == True].set_index("game_id"); A = o[o.home == False].set_index("game_id"); z = m.set_index("game_id")
z["h_off"], z["a_off"], z["h_def"], z["a_def"] = H.off_luck, A.off_luck, H.def_luck, A.def_luck
z["luck_diff"] = (z.h_off - z.h_def) - (z.a_off - z.a_def)               # home net hot-ness minus away
z["luck_sum"] = z.h_off + z.a_off - z.h_def - z.a_def                     # both teams hot -> more points?
line("spread error ~ home-minus-away efg luck", z.err_s, z.luck_diff * 100)
line("total error  ~ combined efg luck", z.err_t, z.luck_sum * 100)
for yr, d in z.groupby("sea"): line(f"   spread season {yr}", d.err_s, d.luck_diff * 100)

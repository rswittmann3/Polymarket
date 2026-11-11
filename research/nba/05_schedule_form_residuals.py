"""Do schedule / travel / form features predict the closing-line error (spread and total)? 2008-2026 closing lines, kind=reg+post.
Every feature uses only information available before tip. Reports t-stat, per-season sign consistency, and BH-adjusted q."""
import numpy as np, pandas as pd
from math import radians, sin, cos, asin, sqrt

g = pd.read_pickle("data/history/games_nba.pkl"); g["date"] = pd.to_datetime(g.date)
g = g[g.kind.isin(["reg", "post"]) & g.home_spread_close.notna() & g.total_close.notna() & g.margin.notna()].copy()
g["yr"] = np.where(g.date.dt.month >= 9, g.date.dt.year + 1, g.date.dt.year)
g = g.sort_values(["date", "home"]).reset_index(drop=True); g["gid"] = g.index
g["err_s"] = g.margin + g.home_spread_close            # home margin minus (-home_spread)  (+ => home beat the line)
g["err_t"] = g.points - g.total_close
g["cover_s"] = g.err_s; g["cover_t"] = g.err_t

LOC = {"ATL":(33.76,-84.40),"BOS":(42.37,-71.06),"BKN":(40.68,-73.97),"NJN":(40.68,-74.14),"CHA":(35.23,-80.84),"CHH":(35.23,-80.84),"CHI":(41.88,-87.67),"CLE":(41.50,-81.69),
 "DAL":(32.79,-96.81),"DEN":(39.75,-105.00),"DET":(42.34,-83.06),"GSW":(37.77,-122.39),"GS":(37.77,-122.39),"HOU":(29.75,-95.36),"IND":(39.76,-86.16),"LAC":(34.04,-118.27),"LAL":(34.04,-118.27),
 "MEM":(35.14,-90.05),"MIA":(25.78,-80.19),"MIL":(43.04,-87.92),"MIN":(44.98,-93.28),"NOP":(29.95,-90.08),"NOH":(29.95,-90.08),"NOK":(35.46,-97.52),"NYK":(40.75,-73.99),"NY":(40.75,-73.99),
 "OKC":(35.46,-97.52),"SEA":(47.62,-122.35),"ORL":(28.54,-81.38),"PHI":(39.90,-75.17),"PHX":(33.45,-112.07),"POR":(45.53,-122.67),"SAC":(38.58,-121.50),"SAS":(29.43,-98.44),"SA":(29.43,-98.44),
 "TOR":(43.64,-79.38),"UTA":(40.77,-111.90),"UTAH":(40.77,-111.90),"WAS":(38.90,-77.02),"WSH":(38.90,-77.02),"NO":(29.95,-90.08)}
def dist(a, b):
    if a not in LOC or b not in LOC: return np.nan
    (la, lo), (lb, lq) = LOC[a], LOC[b]
    la, lo, lb, lq = map(radians, (la, lo, lb, lq)); h = sin((lb-la)/2)**2 + cos(la)*cos(lb)*sin((lq-lo)/2)**2
    return 6371*2*asin(sqrt(h)) * 0.621371

# one row per team-game, in time order
tg = pd.concat([g.assign(team=g.home, opp=g.away, is_home=1, cover=g.err_s, tot=g.err_t, venue=g.home),
                g.assign(team=g.away, opp=g.home, is_home=0, cover=-g.err_s, tot=g.err_t, venue=g.home)]).sort_values(["team", "date", "gid"])
grp = tg.groupby("team")
tg["rest"] = (tg.date - grp.date.shift(1)).dt.days.clip(upper=7)
tg["prev_venue"] = grp.venue.shift(1)
tg["travel"] = [dist(a, b) for a, b in zip(tg.prev_venue, tg.venue)]
tg["g_last4"] = grp.date.transform(lambda s: [ (s.iloc[max(0,i-3):i+1] > s.iloc[i] - pd.Timedelta(days=4)).sum() for i in range(len(s))])
tg["g_last7"] = grp.date.transform(lambda s: [ (s.iloc[max(0,i-5):i+1] > s.iloc[i] - pd.Timedelta(days=7)).sum() for i in range(len(s))])
tg["road_trip"] = grp.is_home.transform(lambda s: s.eq(0).astype(int).groupby(s.ne(s.shift()).cumsum()).cumsum()) * (1 - tg.is_home)
for k in (10, 30):                                      # trailing cover margin and total error, shifted so the current game is excluded
    tg[f"ats{k}"] = grp.cover.transform(lambda s: s.shift(1).rolling(k, min_periods=k//2).mean())
    tg[f"tot{k}"] = grp.tot.transform(lambda s: s.shift(1).rolling(k, min_periods=k//2).mean())
tg["gnum"] = grp.cumcount(); tg["alt"] = tg.venue.map(lambda v: 1 if v == "DEN" else (0.6 if v == "UTA" else 0)) 
H = tg[tg.is_home == 1].set_index("gid"); A = tg[tg.is_home == 0].set_index("gid")
D = g.set_index("gid").copy()
for c in ("rest", "travel", "g_last4", "g_last7", "road_trip", "ats10", "ats30", "tot10", "tot30"):
    D["h_" + c] = H[c]; D["a_" + c] = A[c]
D["rest_diff"] = D.h_rest - D.a_rest
D["a_b2b"] = (D.a_rest == 1).astype(int); D["h_b2b"] = (D.h_rest == 1).astype(int)
D["b2b_diff"] = D.a_b2b - D.h_b2b                       # + => away tired
D["a_3in4"] = (D.a_g_last4 >= 3).astype(int); D["h_3in4"] = (D.h_g_last4 >= 3).astype(int); D["tired_diff"] = D.a_3in4 - D.h_3in4
D["a_travel_k"] = D.a_travel / 1000; D["h_travel_k"] = D.h_travel / 1000
D["a_long_rest"] = (D.a_rest >= 4).astype(int); D["h_long_rest"] = (D.h_rest >= 4).astype(int)
D["ats_diff10"] = D.h_ats10 - D.a_ats10; D["ats_diff30"] = D.h_ats30 - D.a_ats30
D["tot_form10"] = D.h_tot10 + D.a_tot10; D["tot_form30"] = D.h_tot30 + D.a_tot30
D["playoff"] = (D.kind == "post").astype(int)
D["season_frac"] = D.groupby("yr").date.transform(lambda s: (s - s.min()).dt.days / 200)
D["early"] = (D.season_frac < 0.15).astype(int)
D["denver_visitor"] = (D.home == "DEN").astype(int)
D["total_level"] = (D.total_close - D.total_close.mean()) / 10
D["fav_big"] = (D.home_spread_close.abs() >= 9).astype(int)
D["home_fav"] = (D.home_spread_close < 0).astype(int)
FEATS_S = ["rest_diff", "b2b_diff", "a_b2b", "h_b2b", "tired_diff", "a_travel_k", "h_travel_k", "a_long_rest", "h_long_rest", "ats_diff10", "ats_diff30", "a_road_trip", "playoff", "early", "fav_big", "home_fav"]
FEATS_T = ["a_b2b", "h_b2b", "tired_diff", "a_3in4", "h_3in4", "a_travel_k", "tot_form10", "tot_form30", "playoff", "early", "denver_visitor", "total_level"]
def ols_t(y, x):
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(d) < 30 or d.x.nunique() < 2: return np.nan, np.nan, len(d)
    X = np.c_[np.ones(len(d)), d.x.values]; b = np.linalg.lstsq(X, d.y.values, rcond=None)[0]
    r = d.y.values - X @ b; V = np.linalg.inv(X.T @ X) @ (X.T * r**2) @ X @ np.linalg.inv(X.T @ X); return b[1], np.sqrt(V[1, 1]), len(d)
rows = []
for tgt, fs in (("err_s", FEATS_S), ("err_t", FEATS_T)):
    for f in fs:
        b, se, n = ols_t(D[tgt], D[f]); sg = []
        for yr, d in D.groupby("yr"):
            sg.append(np.sign(ols_t(d[tgt], d[f])[0]))
        rows.append({"target": tgt, "feature": f, "n": n, "coef": b, "se": se, "t": b / se, "yrs_same_sign": f"{int((np.array(sg) == np.sign(b)).sum())}/{len(sg)}"})
R = pd.DataFrame(rows)
from math import erf
R["p"] = [1 - erf(abs(t) / sqrt(2)) for t in R.t]
R = R.sort_values("p"); m = len(R); R["q"] = np.minimum.accumulate((R.p * m / np.arange(1, m + 1))[::-1])[::-1]
pd.set_option("display.width", 200); print(R.round(4).to_string(index=False))
D.to_pickle("data/model/nba/sched_feats.pkl")

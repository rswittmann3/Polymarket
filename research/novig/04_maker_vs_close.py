"""Maker result measured against the CLOSING price (VWAP of the last 10 min of pregame trades, else the last pregame trade), cents per contract,
plus the dollar size of the pool per day. Pregame trades only (no fee for anyone pregame). Positive = the resting side beat the close."""
import sys, glob; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from novig_mm import flow

M = pd.read_pickle("data/model/nba/maker_markouts.pkl")
M = M[M.phase != "live"].copy()
M["ts_start"] = M.ts + M.mins_to_start * 60                           # scheduled start (s)
def close_price(g):
    pre = g[g.mins_to_start > 0]
    last = pre[pre.mins_to_start <= 10]
    if len(last) and last.qty.sum() > 0: return float((last.pc * last.qty).sum() / last.qty.sum())
    return float(pre.sort_values("ts").pc.iloc[-1]) if len(pre) else np.nan
cp = M.groupby("marketId").apply(close_price, include_groups=False).rename("close"); M = M.join(cp, on="marketId")
M["maker_close_c"] = -100 * M.dir * (M.close - M.pc)
M = M[M.close.notna() & (M.mins_to_start > 10)]                       # exclude the closing window itself
M["grp"] = np.where(M.mt.isin(["SPREAD", "TOTAL", "MONEY"]), M.mt, "PROP")
M["usd"] = M.qty * M.maker_close_c / 100
days = M.day.nunique()
def agg(g):
    w = g.qty; mean = (g.maker_close_c * w).sum() / w.sum()
    cl = g.assign(wm=g.maker_close_c * g.qty).groupby("marketId").agg(wm=("wm", "sum"), w=("qty", "sum"))
    se = np.sqrt(((cl.wm - mean * cl.w) ** 2).sum()) / cl.w.sum()
    return pd.Series({"trades": len(g), "contracts_k": round(w.sum() / 1e3), "maker_vs_close_c": round(mean, 3), "se_c": round(se, 3), "markets": g.marketId.nunique(), "pool_$/day": round(g.usd.sum() / days, 1), "notional_$k/day": round(w.sum() * g.pc.mean() / 1e3 / days, 1)})
pd.set_option("display.width", 200)
print("days:", days)
print("\nBy market group, main vs alt line, phase (maker result vs close):")
print(M.groupby(["grp", "line", "phase"]).apply(agg, include_groups=False).to_string())
print("\nAll pregame, by group:"); print(M.groupby(["grp", "line"]).apply(agg, include_groups=False).to_string())
print("\nTOTAL pool across everything: $%.0f/day (all makers combined), contracts/day %.0fk" % (M.usd.sum() / days, M.qty.sum() / 1e3 / days))
M.to_pickle("data/model/nba/maker_vs_close.pkl")

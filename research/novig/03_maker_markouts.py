"""Realised maker result on Novig NBA, from the public trade files joined to recorded market metadata (start time, type, main/alt line).
maker_c = cents per contract the resting side earned vs the VWAP 60s / 300s later (positive = makers made money net of adverse selection;
the maker credit, live only, is not included). Standard errors clustered by market.  Needs tools/fetch_novig_trades.py first."""
import sys, glob
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from novig_mm import flow

metas = []
for d in ("data/novig", "data/props"):
    paths = sorted(Path(d).glob("touch_*.jsonl.gz"))
    if paths: metas.append(flow.load_meta(paths))
meta = pd.concat(metas); meta = meta[~meta.index.duplicated(keep="last")]
meta = meta[meta.lg == "NBA"]
print("recorded NBA markets:", len(meta), meta.groupby(["mt", "line"]).size().to_dict())
parts = []
for f in sorted(glob.glob("data/novig_trades/NBA_*.csv")):
    raw = pd.read_csv(f, usecols=["timestamp", "outcomeId", "marketId", "tradeType", "cost", "qty", "side"])
    m = flow.trade_markouts(raw, meta)
    if not m.empty: parts.append(m.assign(day=f[-14:-4]))
M = pd.concat(parts, ignore_index=True)
M.to_pickle("data/model/nba/maker_markouts.pkl")
print("taker trades joined:", len(M), "| contracts:", int(M.qty.sum()), "| days:", M.day.nunique())
pd.set_option("display.width", 220)
for h in ("maker_60s", "maker_300s"):
    s = flow.summarize(M, h); print("\n", h); print(s.sort_values(["type", "line", "phase"]).to_string(index=False))

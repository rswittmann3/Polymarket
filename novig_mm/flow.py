"""Who wins when you make markets, and when?  Join recorded touches (which carry each market's scheduled start) with
Novig's daily public trade files and measure, per phase of the game, the maker's result against the later price.

For a trade at price P (taker side) the maker is on the other side of the same print, so
    maker_result_c = -100 * dir * (VWAP(trades in the window h..h+w after the trade) - P)       [cents per contract]
Positive = makers made money net of the spread they captured and the information they gave away; the maker credit
is NOT included (live only: 25% of the taker fee, 0.015*P*(1-P) per contract). Standard errors are clustered by market.
Requires pandas/numpy (`pip install -e ".[novig,history]"`).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import analyze

PHASES = [(-1e12, 0, "live"), (0, 60, "pre <1h"), (60, 360, "pre 1-6h"), (360, 1440, "pre 6-24h"), (1440, 1e12, "pre >24h")]
BIG = 10 ** 9


def load_meta(touch_paths: Iterable[Path]) -> pd.DataFrame:
    """marketId -> league, type, line (main/alt), scheduled start (ms), fee schedule, from recorder files."""
    rows = list(analyze.read_rows(touch_paths))
    mains = analyze.main_lines(rows)
    last: dict[str, dict] = {}
    for r in rows:
        last[r["m"]] = r
    df = pd.DataFrame([{"marketId": m, "lg": r["lg"], "mt": r["mt"], "st": r["st"], "ch": r.get("ch"),
                        "line": "main" if m in mains else "alt"} for m, r in last.items()])
    return df.set_index("marketId")


def phase_of(minutes_to_start: pd.Series) -> pd.Series:
    """Phase by minutes until the scheduled start: intervals are (lo, hi], so a print exactly at the start is live."""
    out = pd.Series("", index=minutes_to_start.index, dtype="object")
    for lo, hi, name in PHASES:
        out[(minutes_to_start > lo) & (minutes_to_start <= hi)] = name
    return out


def trade_markouts(trades: pd.DataFrame, meta: pd.DataFrame, horizons=((60, 60), (300, 300))) -> pd.DataFrame:
    """`trades`: raw rows of a trades.csv (both sides). Returns one row per STRAIGHT taker trade in `meta` markets."""
    df = trades[(trades.tradeType == "STRAIGHT") & (trades.qty > 0) & trades.marketId.isin(meta.index)]
    if df.empty:
        return pd.DataFrame()
    canon = df.groupby("marketId").outcomeId.min()
    t = df[df.side == "TAKER"].copy()
    is_c = t.outcomeId == t.marketId.map(canon)
    price = t.cost / t.qty
    t["pc"] = np.where(is_c, price, 1 - price)
    t["dir"] = np.where(is_c, 1, -1)
    ts = pd.to_datetime(t.timestamp, utc=True, format="ISO8601")
    t["ts"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("float64")
    t["mid"] = t.marketId.astype("category").cat.codes.astype("int64")
    t = t.sort_values(["mid", "ts"]).reset_index(drop=True)
    key = t.mid.values * BIG + (t.ts.values - t.ts.values.min())
    cq = np.concatenate([[0.0], np.cumsum(t.qty.values)])
    cqp = np.concatenate([[0.0], np.cumsum((t.qty * t.pc).values)])
    out = t[["marketId", "qty", "pc", "dir", "ts"]].copy()
    for h, w in horizons:
        lo = np.searchsorted(key, key + h, side="right")
        hi = np.searchsorted(key, key + h + w, side="right")
        same = (hi > lo) & (t.mid.values[np.minimum(hi - 1, len(t) - 1)] == t.mid.values)
        wq, wqp = cq[hi] - cq[lo], cqp[hi] - cqp[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            fut = np.where(same & (wq > 0), wqp / wq, np.nan)
        out[f"maker_{h}s"] = -100 * t.dir.values * (fut - t.pc.values)
    m = meta.loc[out.marketId]
    out["lg"], out["mt"], out["line"], out["ch"] = m.lg.values, m.mt.values, m.line.values, m.ch.values
    out["mins_to_start"] = (m.st.values / 1000 - out.ts.values) / 60
    out["phase"] = phase_of(out.mins_to_start)
    out["pq"] = out.pc * (1 - out.pc)
    return out


def summarize(m: pd.DataFrame, horizon: str = "maker_300s") -> pd.DataFrame:
    rows = []
    for (lg, mt, line, ph), g in m.groupby(["lg", "mt", "line", "phase"]):
        v = g[g[horizon].notna()]
        if len(v) < 50:
            continue
        cl = v.assign(wm=v[horizon] * v.qty).groupby("marketId").agg(wm=("wm", "sum"), w=("qty", "sum"))
        mean = cl.wm.sum() / cl.w.sum()
        se = float(np.sqrt(((cl.wm - mean * cl.w) ** 2).sum()) / cl.w.sum())
        credit = float((0.015 * g.pq * g.qty).sum()) if ph == "live" and (g.ch == "WHEN_LIVE").any() else 0.0
        rows.append({"league": lg, "type": mt, "line": line, "phase": ph, "trades": len(g), "contracts": int(g.qty.sum()),
                     "maker_c": round(mean, 3), "se_c": round(se, 3), "markets": cl.shape[0],
                     "live_credit_$": round(credit, 2)})
    return pd.DataFrame(rows)


def run(touch_dir: Path, trades_root: Path, days: int | None = None) -> pd.DataFrame:
    meta = load_meta(sorted(Path(touch_dir).glob("touch_*.jsonl.gz")))
    dirs = sorted(d for d in glob.glob(os.path.join(trades_root, "20*")) if os.path.exists(os.path.join(d, "trades.csv")))
    parts = []
    for d in dirs[-days:] if days else dirs:
        raw = pd.read_csv(os.path.join(d, "trades.csv"), usecols=["timestamp", "outcomeId", "marketId", "tradeType", "cost", "qty", "side"])
        parts.append(trade_markouts(raw, meta))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

"""Summarise recorded touches: how wide are Novig's spreads and how deep is the top of book, by market type
and by time to start?  Run after `novig_mm record` has collected a few hours of data."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

ONE = Decimal(1)
BUCKETS = [(0, 30), (30, 120), (120, 360), (360, 1440), (1440, 10 ** 9)]    # minutes to start


def read_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def spread_of(row: dict[str, Any]) -> Decimal | None:
    b0, b1 = row["b"]
    if b0 is None or b1 is None:
        return None
    return ONE - Decimal(b0) - Decimal(b1)


def _bucket(mins: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= mins < hi:
            return f"{lo}-{hi}m" if hi < 10 ** 9 else f">{lo}m"
    return "started"


def main_lines(rows: list[dict[str, Any]]) -> set[str]:
    """Market ids that are the main line of their (event, market type): the two-sided market whose mid is closest to
    50%. Everything else with the same event and type is an alternate line (far strikes, usually wide and thin)."""
    best: dict[tuple, tuple[float, str]] = {}
    for r in rows:
        b0, b1 = r["b"]
        if b0 is None or b1 is None:
            continue
        dist = abs(float(Decimal(b0) - Decimal(b1))) / 2          # |mid - 0.5| of outcome 0
        key = (r["e"], r["mt"])
        if key not in best or dist < best[key][0]:
            best[key] = (dist, r["m"])
    return {m for _, m in best.values()}


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (league, marketType, main/alt line, minutes-to-start bucket). Spreads in cents; qty is the smaller
    side of the touch; `vol` is the median all-time contracts traded in the market."""
    rows = list(rows)
    mains = main_lines(rows)
    g: dict[tuple, dict[str, list]] = defaultdict(lambda: {"sp": [], "q": [], "two": 0, "n": 0, "mk": set(), "vol": {}})
    for r in rows:
        mins = (r["st"] - r["ts"]) / 60000
        if mins < 0:
            continue
        line = "main" if r["m"] in mains else "alt"
        s = g[(r["lg"], r["mt"], line, _bucket(mins))]
        s["n"] += 1
        s["mk"].add(r["m"])
        s["vol"][r["m"]] = r.get("vol", 0)                          # latest value wins
        sp = spread_of(r)
        if sp is not None:
            s["two"] += 1
            s["sp"].append(float(sp) * 100)
            s["q"].append(min(r["q"]))
    out = []
    for (lg, mt, line, b), s in sorted(g.items()):
        sp = sorted(s["sp"])
        out.append({
            "league": lg, "type": mt, "line": line, "to_start": b, "markets": len(s["mk"]),
            "vol_med": int(median(s["vol"].values())) if s["vol"] else 0, "obs": s["n"],
            "two_sided_%": round(100 * s["two"] / s["n"], 1),
            "spread_c_p25": round(sp[len(sp) // 4], 2) if sp else None,
            "spread_c_med": round(median(sp), 2) if sp else None,
            "spread_c_p75": round(sp[3 * len(sp) // 4], 2) if sp else None,
            "touch_qty_med": int(median(s["q"])) if s["q"] else None,
        })
    return out

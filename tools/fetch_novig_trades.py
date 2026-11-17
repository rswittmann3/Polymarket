"""Stream Novig's public daily trade files (data.novig.com/reporting/trade-data/<date>/trades.csv, ~260 MB/day, mostly parlays) and keep only
single-leg trades for the given leagues: data/novig_trades/<league>_<date>.csv. Read-only public data, anonymized.

    python tools/fetch_novig_trades.py --league NBA --days 7          # newest 7 published days
    python tools/fetch_novig_trades.py --league NBA --from 2026-10-21 --to 2026-11-15
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import httpx

BASE = "https://data.novig.com/reporting/trade-data"
OUT = Path("data/novig_trades")


def fetch_day(client: httpx.Client, date: str, leagues: set[str]) -> dict[str, int]:
    rows = {lg: [] for lg in leagues}
    with client.stream("GET", f"{BASE}/{date}/trades.csv", timeout=600) as r:
        r.raise_for_status()
        lines = r.iter_lines()
        header = next(lines)
        cols = next(csv.reader([header]))
        i_lg, i_tt = cols.index("league"), cols.index("tradeType")
        for line in lines:
            if ",STRAIGHT," not in line:
                continue
            f = next(csv.reader([line]))
            if len(f) > i_lg and f[i_tt] == "STRAIGHT" and f[i_lg] in leagues:
                rows[f[i_lg]].append(f)
    OUT.mkdir(parents=True, exist_ok=True)
    for lg, rs in rows.items():
        with open(OUT / f"{lg}_{date}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rs)
    return {lg: len(rs) for lg, rs in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    a = ap.parse_args()
    leagues = set(a.league.split(","))
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (research)"}) as c:
        dates = c.get(f"{BASE}/index.json", timeout=60).json()["dates"]
        if a.start:
            dates = [d for d in dates if a.start <= d <= (a.end or "9999")]
        elif a.days:
            dates = dates[-a.days:]
        for d in dates:
            if all((OUT / f"{lg}_{d}.csv").exists() for lg in leagues):
                continue
            print(d, fetch_day(c, d, leagues), flush=True)


if __name__ == "__main__":
    main()

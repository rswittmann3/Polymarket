"""Rebuild pm_edges/margin_table.json from the public 1-second index price history.

    python tools/build_margin_table.py --days 7

For every five-minute window of BTC/ETH/SOL/XRP it measures the lead at several times-left and records whether
the leader went on to lose (see pm_edges/margin.py for the definitions), then stores a Wilson 95% upper bound on
the reversal rate per (time-left bucket, lead size). Run it again when markets get calmer or wilder.
"""
from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import httpx

BASE = "https://gateway.polymarket.us/v1/asset-prices/history"
ASSETS = ("btc", "eth", "sol", "xrp")
LS = [20, 30, 45, 60, 75, 90, 105, 120]
ZE = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]          # last bin is z >= 4
OUT = Path(__file__).resolve().parent.parent / "pm_edges" / "margin_table.json"


def pull(days: int) -> dict[str, dict[int, float]]:
    end = int(time.time()) // 300 * 300 - 300
    t, data = end - days * 86400, {a: {} for a in ASSETS}
    c = httpx.Client(timeout=30)
    while t < end:
        to = min(end, t + 3599)
        for _ in range(8):
            r = c.get(BASE, params=[("assets", f"crypto:{a}") for a in ASSETS]
                      + [("from", t), ("to", to), ("bucket", "ASSET_PRICE_BUCKET_1S")])
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", "10")) + 1)
                continue
            r.raise_for_status()
            break
        for s in r.json()["series"]:
            for cd in s["candles"]:
                data[s["asset"]["symbol"]][cd["timestamp"]] = float(cd["close"])
        t = to + 1
        time.sleep(1.1)                            # the public gateway rate-limits bursts
    return data


def wilson_ub(k: int, n: int, zc: float = 1.96) -> float:
    if n == 0:
        return 1.0
    ph = k / n
    d = 1 + zc * zc / n
    return (ph + zc * zc / (2 * n) + zc * sqrt(ph * (1 - ph) / n + zc * zc / (4 * n * n))) / d


def samples(data) -> list[tuple[int, float, bool]]:
    out = []
    for dd in data.values():
        t0 = min(dd)
        n = max(dd) - t0 + 1
        p = [dd[t0 + i] for i in range(n)]
        pre = [0.0]
        for x in p:
            pre.append(pre[-1] + x)
        sq = [0.0] * (n + 1)
        for i in range(n):
            sq[i + 1] = sq[i] + ((p[i] - p[i - 60]) ** 2 if i >= 60 else 0.0)
        S = ((t0 + 700) // 300 + 1) * 300
        while S + 300 <= t0 + n:
            E = S + 300
            ix = lambda t: t - t0  # noqa: E731
            p0 = (pre[ix(S)] - pre[ix(S - 60)]) / 60
            up = (pre[ix(E)] - pre[ix(E - 60)]) / 60 >= p0
            for L in LS:
                t = E - L
                known = (pre[ix(t)] - pre[ix(E - 60)]) if L < 60 else 0.0
                mean_est = (known + min(L, 60) * p[ix(t - 1)]) / 60
                sig = sqrt((sq[ix(t)] - sq[ix(t - 540)]) / 540)
                if sig > 0:
                    z = (mean_est - p0) / sig
                    out.append((L, abs(z), (z >= 0) != up))
            S += 300
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    smp = samples(pull(args.days))
    table = {}
    for L in LS:
        row = []
        for i, lo in enumerate(ZE):
            hi = ZE[i + 1] if i + 1 < len(ZE) else 1e9
            s = [x for x in smp if x[0] == L and lo <= x[1] < hi]
            k = sum(x[2] for x in s)
            row.append({"z_lo": lo, "n": len(s), "rev": k, "ub": round(wilson_ub(k, len(s)), 5)})
        for j in range(1, len(row)):               # a bigger lead can't be riskier
            row[j]["ub"] = min(row[j]["ub"], row[j - 1]["ub"])
        table[str(L)] = row
    meta = {"source": f"{args.days} days of 1-second index prices, BTC/ETH/SOL/XRP",
            "z": "(expected final average - price to beat)/RMS of 60 s changes over the last 540 s",
            "ub": "Wilson 95% upper bound on the reversal rate, made monotone in z"}
    OUT.write_text(json.dumps({"meta": meta, "table": table}, indent=1))
    print(f"wrote {OUT} from {len(smp)} samples")


if __name__ == "__main__":
    main()

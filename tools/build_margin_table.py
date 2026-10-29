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
from math import erf, sqrt
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


def samples(data) -> list[tuple[int, float, bool, float]]:
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
                    fin = (pre[ix(E)] - pre[ix(E - 60)]) / 60
                    out.append((L, abs(z), (z >= 0) != up, (fin - p0) / sig * (1 if z >= 0 else -1)))
            S += 300
    return out


NOISE = {"exact": 0.50, "approx": 0.71}    # sd, in sigma units, of exchange-vs-reconstruction error (36 windows)


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def build(smp, s: float) -> dict:
    t = {}
    for L in LS:
        row = []
        for i, lo in enumerate(ZE):
            hi = ZE[i + 1] if i + 1 < len(ZE) else 1e9
            cell = [x for x in smp if x[0] == L and lo <= x[1] < hi]
            n = len(cell)
            k = sum(phi(-x[3] / s) for x in cell)    # expected reversals once the measurement noise is added
            row.append({"z_lo": lo, "n": n, "rev": round(k, 2), "ub": round(wilson_ub(k, n) if n else 1.0, 5)})
        for j in range(1, len(row)):                 # a bigger lead can't be riskier
            row[j]["ub"] = min(row[j]["ub"], row[j - 1]["ub"])
        t[str(L)] = row
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    smp = samples(pull(args.days))
    meta = {"source": f"{args.days} days of 1-second index prices, BTC/ETH/SOL/XRP, with measurement noise added",
            "z": "(expected final average - price to beat)/RMS of 60 s changes over the last 540 s",
            "noise": "exchange published values vs my candle reconstruction: sd 0.50 sigma on the price to beat and on the final average",
            "exact": "price to beat taken from the exchange (noise on the final average only)",
            "approx": "price to beat reconstructed from candles (noise on both ends)",
            "ub": "Wilson 95% upper bound on the reversal rate, monotone in z"}
    OUT.write_text(json.dumps({"meta": meta, "exact": build(smp, NOISE["exact"]), "approx": build(smp, NOISE["approx"])}, indent=1))
    print(f"wrote {OUT} from {len(smp)} samples")


if __name__ == "__main__":
    main()

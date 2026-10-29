"""Sweep price bands and edge thresholds for the margin-gated late-favorite strategy.

Inputs (all already on disk, no API calls):
  * 1-second index prices (BTC): /tmp/claude-0/mm/hist7.json   -> window open reference, lead z, realized outcome
  * proxy order-book prices: /tmp/claude-0/s/bt.json           -> polymarket.com BTC 5m Up price at ~1-minute
                                                                  resolution (the US book is not archived)

For every window and each time-left, the favorite's price comes from the latest proxy point at or before that time
(at most MAX_STALE seconds old); the ask is that price plus ASK_PAD (a half-spread). A trade is taken at the FIRST time
(earliest) the rule passes, once per window. P&L is per $1 stake at the ask: a win pays 1/ask - 1 (minus the taker fee),
a loss costs the whole stake. Results are split by time (first 60% train, last 40% test) to see whether a setting that
looks best in-sample holds up.

    python tools/backtest_margin_strategy.py
"""
from __future__ import annotations

import itertools
import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pm_edges.margin import edge_after_risk, lead_z, reversal_ub, sigma_ref  # noqa: E402

IDX = "/tmp/claude-0/mm/hist7.json"
PROXY = "/tmp/claude-0/s/bt.json"
LS = (100, 90, 75, 60, 45, 30, 20)
MAX_STALE = int(sys.argv[1]) if len(sys.argv) > 1 else 75
ASK_PAD = 0.01
FEE_RATE = 0.0695


def fee(p: float) -> float:
    return FEE_RATE * p * (1 - p)


def load():
    idx = {int(k): v for k, v in json.load(open(IDX))["btc"].items()}
    proxy = json.load(open(PROXY))
    rows = []
    for w in proxy:
        S = w["S"]
        E = S + 300
        try:
            p0 = sum(idx[S - 60 + i] for i in range(60)) / 60
            fin = sum(idx[E - 60 + i] for i in range(60)) / 60
        except KeyError:
            continue
        up = fin >= p0
        cands = []
        for L in LS:
            t = E - L
            pts = [(o, p) for o, p in w["hist"] if o <= 300 - L]
            if not pts or (300 - L) - pts[-1][0] > MAX_STALE:
                continue
            p_up = pts[-1][1]
            try:
                known = sum(idx[E - 60 + i] for i in range(60 - L)) if L < 60 else 0.0
                prices = [idx[x] for x in range(t - 600, t)]
            except KeyError:
                continue
            sg = sigma_ref(prices)
            if sg <= 0:
                continue
            z = lead_z(p0, known, prices[-1], L, sg)
            book_side = "Up" if p_up >= 0.5 else "Down"
            fav_price = p_up if book_side == "Up" else 1 - p_up
            cands.append({"L": L, "z": z, "book_side": book_side, "ask": fav_price + ASK_PAD,
                          "ub": reversal_ub(L, abs(z)), "model_side": "Up" if z >= 0 else "Down"})
        rows.append({"S": S, "up": up, "cands": cands})
    rows.sort(key=lambda r: r["S"])
    return rows


def run(rows, lo, hi, min_edge, need_agree=True):
    pnl, n, wins = 0.0, 0, 0
    for w in rows:
        for c in w["cands"]:                       # LS is descending: the earliest time-left first
            a = c["ask"]
            if not lo <= a <= hi:
                continue
            if need_agree and c["model_side"] != c["book_side"]:
                continue
            if edge_after_risk(a, fee(a), c["ub"]) < min_edge:
                continue
            won = (c["book_side"] == "Up") == w["up"]
            pnl += (1 / a - 1 - fee(a) / a) if won else -1.0   # per $1 staked
            n += 1
            wins += won
            break
    return n, wins, pnl


def main() -> None:
    rows = load()
    cut = int(len(rows) * 0.6)
    train, test = rows[:cut], rows[cut:]
    days = (rows[-1]["S"] - rows[0]["S"]) / 86400
    print(f"{len(rows)} windows over {days:.1f} days (train {len(train)}, test {len(test)}); stake $1, ask = proxy + {ASK_PAD}")
    # baseline: the OLD strategy, no margin check at all, band 0.95-0.99
    for name, rs in (("train", train), ("test", test), ("all", rows)):
        n, w, p = run(rs, 0.95, 0.99, -9, need_agree=False)
        print(f"BASELINE old strategy ({name}): trades {n}, losses {n - w}, P&L ${p:+.2f}, per trade ${p / max(n, 1):+.3f}")
    results = []
    for lo, hi, e in itertools.product((0.80, 0.85, 0.90, 0.93, 0.95), (0.97, 0.98, 0.99), (-0.005, 0.0, 0.005, 0.01, 0.02, 0.03)):
        tr = run(train, lo, hi, e)
        te = run(test, lo, hi, e)
        al = run(rows, lo, hi, e)
        results.append((lo, hi, e, tr, te, al))
    print("\nlo    hi    edge  | TRAIN n  L  P&L   | TEST n  L  P&L    | ALL n  L  P&L    $/trade")
    results.sort(key=lambda r: -(r[3][2]))   # rank by TRAIN P&L only, then look at TEST
    for lo, hi, e, tr, te, al in results[:14]:
        print(f"{lo:.2f}  {hi:.2f}  {e:+.3f} | {tr[0]:4d} {tr[0] - tr[1]:2d} {tr[2]:+7.2f} | {te[0]:4d} {te[0] - te[1]:2d} {te[2]:+7.2f} | {al[0]:4d} {al[0] - al[1]:2d} {al[2]:+7.2f}  {al[2] / max(al[0], 1):+.3f}")
    print("\ncurrent live setting (0.95-0.99, edge 0.005):", run(rows, 0.95, 0.99, 0.005))
    print("test-set ranking (best by TEST P&L, for robustness):")
    for lo, hi, e, tr, te, al in sorted(results, key=lambda r: -r[4][2])[:6]:
        print(f"  {lo:.2f}-{hi:.2f} edge {e:+.3f}: train {tr[0]} trades {tr[2]:+.2f} | test {te[0]} trades {te[2]:+.2f} ({te[0] - te[1]} losses)")


if __name__ == "__main__":
    main()

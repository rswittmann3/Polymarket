"""How safe is a lead? An empirical reversal table for Polymarket US 5-minute crypto Up/Down windows.

A window opening at S and closing at E = S + 300 settles Up when mean(close[E-60 .. E-1]) >= mean(close[S-60 .. S-1]),
using the 1-second index prices from the public Asset Prices API. With `left` seconds to go, the expected final
average is `(known + n_future * price_now) / 60`, and the lead is measured in units of how far the index usually
moves in a minute:

    z = (expected final average - price to beat) / RMS(60-second changes over the last 540 s)

Why not a random-walk formula: the index is smooth, so one-second steps are tiny and persistent, and the real
60-second spread is about 6.5x what a random walk built from one-second steps predicts. A normal model built that
way called 2.7% reversals on windows it rated 99.99% safe. So `margin_table.json` holds what actually happened
over 7 days (BTC/ETH/SOL/XRP, 8,064 windows): for each time-left bucket and lead size, how often the leader lost.
The table stores a Wilson 95% upper bound on that rate, made monotone in z. Rebuild it with the history pull in
tools/build_margin_table.py when volatility regimes change.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

WINDOW = 60
SIGMA_SPAN = 540        # seconds of 60-second changes used for the volatility reference
MAX_LEFT = 120          # beyond this the table has nothing: treat as unsafe
_TABLE = json.loads((Path(__file__).parent / "margin_table.json").read_text())["table"]
_BUCKETS = sorted(int(k) for k in _TABLE)


def sigma_ref(prices: list[float]) -> float:
    """RMS of 60-second changes over the last SIGMA_SPAN seconds. `prices` is dense 1-second data, oldest
    first, ending at the latest published second; needs SIGMA_SPAN + 60 values."""
    if len(prices) < SIGMA_SPAN + WINDOW:
        return 0.0
    tail = prices[-(SIGMA_SPAN + WINDOW):]
    d = [tail[i] - tail[i - WINDOW] for i in range(WINDOW, len(tail))]
    return sqrt(sum(x * x for x in d) / len(d))


def lead_z(p0: float, known_sum: float, s_now: float, left: float, sigma: float) -> float:
    """Signed lead: positive means Up is ahead. p0 is the price to beat; known_sum the sum of the index prices
    already published inside [E-60, now) (zero while left >= 60); s_now the latest index price."""
    if sigma <= 0:
        return 0.0
    n_future = int(round(min(max(left, 0.0), WINDOW)))
    expected = (known_sum + n_future * s_now) / WINDOW
    return (expected - p0) / sigma


def reversal_ub(left: float, z_abs: float) -> float:
    """Upper bound (95%) on the chance the current leader loses, given |z| and seconds left. Rounds `left` UP to
    the next table bucket (less time is safer, so this errs cautious); no data past MAX_LEFT, so 1.0."""
    if left > MAX_LEFT:
        return 1.0
    bucket = next(b for b in _BUCKETS if left <= b)
    ub = _TABLE[str(bucket)][0]["ub"]
    for row in _TABLE[str(bucket)]:
        if z_abs >= row["z_lo"]:
            ub = row["ub"]
    return ub


def edge_after_risk(ask: float, fee_per_share: float, ub: float) -> float:
    """Worst-case expected profit per share: win probability (1 - ub) less the price and the taker fee."""
    return (1.0 - ub) - ask - fee_per_share

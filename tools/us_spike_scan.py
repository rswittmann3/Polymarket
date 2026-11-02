"""Find 'YES price jumped' spikes in long-dated Polymarket US markets (from the data/us_all recorder) and log paper signals.

Rule under test (found on ~50k resolved global-Polymarket markets, 2024-2026, every half-year positive):
  after the YES mid rises by >= --rise within --window hours, in a market whose scheduled end is >= --min-days away
  and whose YES price is 0.10-0.90, BUYING NO and holding to resolution beat its price (NO won ~67% vs ~55% priced).
Whether that holds on the US exchange is exactly what this script measures, with executable prices.

    python tools/us_spike_scan.py scan  --window 24 --rise 0.12     # print/log current signals
    python tools/us_spike_scan.py track                              # mark old signals to market (+1d,+3d,+7d,+14d,+30d)
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REC = "data/us_all/rec-*.jsonl"
EVENTS = Path("data/us_all/events.json")                  # all active events with their markets; refreshed by `refresh`
SIGNALS = Path("data/us_all/signals.jsonl")
HORIZONS = [1, 3, 7, 14, 30]


def load_series(since: float) -> dict[str, list[tuple[float, float, float]]]:
    """slug -> [(t, bid, ask)] from quote rows and trade rows (which also carry bid/ask)."""
    out: dict[str, list] = defaultdict(list)
    for fn in sorted(glob.glob(REC)):
        with open(fn) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["t"] < since or r.get("b") is None or r.get("a") is None:
                    continue
                out[r["s"]].append((r["t"], r["b"], r["a"]))
    return out


def sched_end() -> dict[str, tuple[float, str]]:
    d = {}
    if not EVENTS.exists():
        return d
    for e in json.loads(EVENTS.read_text()):
        for m in e.get("markets", []):
            try:
                end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            d[m["slug"]] = (end, e.get("slug") or m["slug"])
    return d


def mid_at(s: list, t: float) -> float | None:
    """mid at the last row at or before t (None if the series does not reach back that far)."""
    if not s or s[0][0] > t + 3600:
        return None
    lo, hi = 0, len(s) - 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if s[m][0] <= t:
            lo = m
        else:
            hi = m - 1
    return (s[lo][1] + s[lo][2]) / 2


def scan(a) -> None:
    now = time.time()
    series = load_series(now - (a.window + 30) * 3600)
    ends = sched_end()
    last_sig: dict[str, float] = {}                       # slug -> time of its latest signal
    if SIGNALS.exists():
        for line in SIGNALS.open():
            r = json.loads(line)
            last_sig[r["slug"]] = max(last_sig.get(r["slug"], 0), r["t"])
    n = 0
    for slug, s in series.items():
        if slug not in ends or len(s) < 3:
            continue
        end, ev = ends[slug]
        if end - now < a.min_days * 86400:
            continue
        t, bid, ask = s[-1]
        if now - t > 1800 or ask - bid > a.max_spread:
            continue
        mid, past = (bid + ask) / 2, mid_at(s, now - a.window * 3600)
        if past is None or not (0.10 <= mid <= 0.90) or mid - past < a.rise:
            continue
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d")
        if now - last_sig.get(slug, 0) < 7 * 86400:          # same spacing as the 12h-history backtest (one episode per market per week)
            continue
        sig = {"slug": slug, "ev": ev, "t": round(now), "day": day, "yes_bid": bid, "yes_ask": ask, "past_mid": round(past, 4), "rise": round(mid - past, 4),
               "no_ask": round(1 - bid, 4), "days_to_end": round((end - now) / 86400), "window_h": a.window}
        n += 1
        print(f"SIGNAL {slug}: YES {past:.3f} -> {mid:.3f} (+{mid - past:.3f} in {a.window}h), buy NO at {1 - bid:.3f}, ends in {sig['days_to_end']}d")
        if not a.dry:
            SIGNALS.parent.mkdir(parents=True, exist_ok=True)
            with SIGNALS.open("a") as f:
                f.write(json.dumps(sig) + "\n")
    print(f"{n} new signals ({len(series)} markets with data)")


def refresh(a) -> None:
    import httpx
    from pm_edges.pmus import GATEWAY
    http, evs, off = httpx.Client(timeout=40), [], 0
    while True:
        for _ in range(6):
            r = http.get(f"{GATEWAY}/v1/events", params={"limit": 200, "offset": off, "active": "true", "closed": "false"})
            if r.status_code == 429:
                time.sleep(11)
                continue
            break
        batch = r.json().get("events", []) if r.status_code == 200 else []
        if not batch:
            break
        evs += batch
        off += 200
        time.sleep(2.2)
    if evs:
        EVENTS.write_text(json.dumps(evs))
    print(f"refreshed {len(evs)} events")


def track(a) -> None:
    if not SIGNALS.exists():
        print("no signals yet")
        return
    sigs = [json.loads(l) for l in SIGNALS.open()]
    series = load_series(min(s["t"] for s in sigs) - 3600)
    now = time.time()
    rows = defaultdict(list)
    clus = defaultdict(lambda: defaultdict(list))         # horizon -> (day, sport/category) -> moves
    for sg in sigs:
        s = series.get(sg["slug"], [])
        for h in HORIZONS:
            t = sg["t"] + h * 86400
            if t > now:
                continue
            mid = mid_at(s, t)
            if mid is None:
                continue
            mv = (1 - mid) - sg["no_ask"]                   # NO value at the horizon (mid) minus the executable entry price
            rows[h].append(mv)
            clus[h][(sg["day"], sg["ev"].split("-")[0])].append(mv)
    print(f"{len(sigs)} signals")
    for h in HORIZONS:
        v = rows.get(h, [])
        if v:
            cm = [sum(x) / len(x) for x in clus[h].values()]
            print(f"  +{h:2d}d: n={len(v):3d} mean NO move (mid vs entry ask, i.e. before exit costs) {sum(v) / len(v) * 100:+.2f}c"
                  f"   | {len(cm)} independent clusters (day x category): mean of cluster means {sum(cm) / len(cm) * 100:+.2f}c")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "track", "refresh"])
    ap.add_argument("--window", type=float, default=24.0)
    ap.add_argument("--rise", type=float, default=0.12)
    ap.add_argument("--min-days", type=float, default=60.0)
    ap.add_argument("--max-spread", type=float, default=0.05)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"scan": scan, "track": track, "refresh": refresh}[a.cmd](a)

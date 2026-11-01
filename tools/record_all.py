"""Record top-of-book, trades and open interest for every liquid Polymarket US market, for strategy research.

Listens only (no orders). Uses the authenticated LITE market-data stream, which carries best bid/ask, last trade,
cumulative shares traded and open interest. To keep the file small it writes
  * a trade row whenever sharesTraded rises (price, size), and
  * a quote row when the bid/ask changed and at least --quote-secs passed since the last row for that market.

    python tools/record_all.py --hours 1.9            # appends to data/us_all/rec-YYYYMMDD.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
from dotenv import dotenv_values

from pm_edges.pmus import GATEWAY, Signer

WS_PATH = "/v1/ws/markets"
OUT = Path("data/us_all")


def list_slugs(min_mid: float, max_mid: float) -> list[str]:
    http = httpx.Client(timeout=40)
    slugs, off = [], 0
    while True:
        for _ in range(6):
            r = http.get(f"{GATEWAY}/v1/events", params={"limit": 200, "offset": off, "active": "true", "closed": "false"})
            if r.status_code == 429:
                time.sleep(11)
                continue
            break
        if r.status_code != 200:
            break
        evs = r.json().get("events", [])
        if not evs:
            break
        for e in evs:
            for m in e.get("markets", []):
                b = (m.get("bestBidQuote") or {}).get("value")
                a = (m.get("bestAskQuote") or {}).get("value")
                if b and a and min_mid <= (float(a) + float(b)) / 2 <= max_mid:
                    slugs.append(m["slug"])
        off += 200
        time.sleep(2.2)
    return slugs


async def run(hours: float, quote_secs: float, lo: float, hi: float) -> None:
    env = dotenv_values(".env")
    signer = Signer(env["POLYMARKET_KEY_ID"], env["POLYMARKET_SECRET_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    stop = time.time() + hours * 3600
    while time.time() < stop:
        slugs = await asyncio.get_running_loop().run_in_executor(None, list_slugs, lo, hi)
        print(f"[{datetime.now(timezone.utc):%H:%M:%S}] subscribing to {len(slugs)} markets", flush=True)
        last: dict[str, tuple] = {}
        last_row_t: dict[str, float] = {}
        cycle_end = min(stop, time.time() + 3600)       # re-list the markets every hour
        n_rows = 0
        try:
            async with websockets.connect("wss://api.polymarket.us" + WS_PATH, additional_headers=signer.headers("GET", WS_PATH),
                                          max_size=None, ping_interval=20) as ws:
                for i in range(0, len(slugs), 300):
                    await ws.send(json.dumps({"subscribe": {"requestId": f"r{i}", "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA_LITE",
                                                            "marketSlugs": slugs[i:i + 300]}}))
                    await asyncio.sleep(0.3)
                f = None
                day = ""
                while time.time() < cycle_end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 5)
                    except asyncio.TimeoutError:
                        continue
                    d = json.loads(raw).get("marketDataLite")
                    if not d:
                        continue
                    now = time.time()
                    today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d")
                    if today != day:
                        if f:
                            f.close()
                        f, day = (OUT / f"rec-{today}.jsonl").open("a"), today
                    s = d["marketSlug"]
                    g = lambda k: float(d[k]["value"]) if d.get(k) else None  # noqa: E731
                    bid, ask, ltp = g("bestBid"), g("bestAsk"), g("lastTradePx")
                    shares, oi = float(d.get("sharesTraded") or 0), float(d.get("openInterest") or 0)
                    prev = last.get(s)
                    row = None
                    if prev is not None and shares > prev[3]:
                        row = {"k": "t", "s": s, "t": round(now, 2), "px": ltp, "q": round(shares - prev[3], 2), "b": bid, "a": ask}
                    elif prev is None or ((bid, ask) != (prev[0], prev[1]) and now - last_row_t.get(s, 0) >= quote_secs):
                        row = {"k": "q", "s": s, "t": round(now, 2), "b": bid, "a": ask, "bd": d.get("bidDepth"), "ad": d.get("askDepth"),
                               "lt": ltp, "oi": oi}
                    last[s] = (bid, ask, ltp, shares)
                    if row:
                        f.write(json.dumps(row) + "\n")
                        last_row_t[s] = now
                        n_rows += 1
                        if n_rows % 500 == 0:
                            f.flush()
                if f:
                    f.close()
        except Exception as e:
            print(f"[{datetime.now(timezone.utc):%H:%M:%S}] stream error {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(5)
        print(f"[{datetime.now(timezone.utc):%H:%M:%S}] cycle done, rows {n_rows}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.9)
    ap.add_argument("--quote-secs", type=float, default=15.0)
    ap.add_argument("--min-mid", type=float, default=0.02)
    ap.add_argument("--max-mid", type=float, default=0.98)
    a = ap.parse_args()
    asyncio.run(run(a.hours, a.quote_secs, a.min_mid, a.max_mid))

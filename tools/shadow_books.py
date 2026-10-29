"""Record the real Polymarket US order book for every 5-minute window, so a strategy can be backtested on prices
that were actually available (not proxies). Listens only; places nothing; no REST calls (the live bot needs the
rate limit). Compute the index-price lead afterwards from one bulk history pull.

    python tools/shadow_books.py --hours 12          # writes data/pmus/shadow_books.jsonl

One line per (asset, second) while 125 s to 10 s remain: top 3 bids/offers (price, size) on the long (Up) side and the
last trade.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import websockets
from dotenv import load_dotenv

from pm_edges.pmus import Signer, updown_slug

OUT = Path("data/pmus/shadow_books.jsonl")
WS_PATH = "/v1/ws/markets"
ASSETS = ["btc", "eth", "sol", "xrp"]


async def main(hours: float) -> None:
    load_dotenv()
    signer = Signer(os.environ["POLYMARKET_KEY_ID"], os.environ["POLYMARKET_SECRET_KEY"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stop = time.time() + hours * 3600
    last_write: dict[str, int] = {}
    with OUT.open("a") as f:
        while time.time() < stop:
            try:
                async with websockets.connect("wss://api.polymarket.us" + WS_PATH,
                                              additional_headers=signer.headers("GET", WS_PATH)) as ws:
                    subscribed: set[int] = set()
                    while time.time() < stop:
                        now = time.time()
                        start = int(now // 300) * 300
                        if start not in subscribed:
                            subscribed.add(start)
                            await ws.send(json.dumps({"subscribe": {
                                "requestId": f"s{start}", "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                                "marketSlugs": [updown_slug(start, a) for a in ASSETS]}}))
                        try:
                            md = json.loads(await asyncio.wait_for(ws.recv(), 1.0)).get("marketData")
                        except asyncio.TimeoutError:
                            continue
                        if not md or md.get("state") != "MARKET_STATE_OPEN":
                            continue
                        slug = md["marketSlug"]
                        left = start + 300 - now
                        sec = int(now)
                        if not 10 <= left <= 125 or last_write.get(slug) == sec:
                            continue
                        last_write[slug] = sec
                        lv = lambda rows: [[float(x["px"]["value"]), float(x["qty"])] for x in rows[:3]]  # noqa: E731
                        lt = ((md.get("stats") or {}).get("lastTradePx") or {}).get("value")
                        f.write(json.dumps({"t": round(now, 2), "slug": slug, "left": round(left, 1), "bids": lv(md.get("bids", [])),
                                            "offers": lv(md.get("offers", [])), "last": float(lt) if lt else None}) + "\n")
                        f.flush()
            except Exception as e:   # reconnect on any stream error
                print("stream error:", e, flush=True)
                await asyncio.sleep(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0)
    asyncio.run(main(ap.parse_args().hours))

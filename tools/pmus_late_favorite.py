"""Late-window favorite buyer for Polymarket US 5-minute crypto Up/Down markets (BTC, ETH, SOL, XRP).

Recreates what wallet 0x50f7 does on polymarket.com: in the last ~2 minutes of a window, buy
whichever side is already priced 0.95-0.99, hold to settlement, repeat. Per $1 that wins you a cent
or two; one reversal loses the stake, so it only pays if the favorite loses less often than roughly
(1 - ask). A 7-day backtest on polymarket.com could not tell that apart from break-even.

Each asset has its own $1 stake and a kill switch: after a LOSS on an asset it stops trading that
asset for the rest of the session. A trade that cannot be confirmed (order unreadable, never
settles) also halts its asset. Maximum loss is therefore about --usd per asset.

Prices come from the authenticated markets WebSocket (the public REST gateway allows only ~5
requests per 10 s). DRY RUN BY DEFAULT; `--live` sends real fill-or-kill orders.

    python tools/pmus_late_favorite.py                              # watch + log decisions
    python tools/pmus_late_favorite.py --live --usd 1               # BTC, ETH, SOL, XRP at $1 each

Price convention (docs): `price.value` is always the Up/long price. Buying Down at X is a
BUY_SHORT with price.value = 1 - X, i.e. hitting the best long bid.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets

from pm_edges.margin import edge_after_risk, lead_z, max_price, reversal_ub, sigma_ref
from pm_edges.pmus import (GATEWAY, Book, Signer, Trader, event_slug, limit_order, taker_fee, updown_slug)

LOG = Path("data/pmus/late_favorite.jsonl")
HEARTBEAT = Path("data/pmus/heartbeat")  # the dashboard reads this to show "Bot running" on any OS
STATE = Path("data/pmus/state.json")  # survives restarts: open trades, per-asset stats, stop flag
FOK = "TIME_IN_FORCE_FILL_OR_KILL"
WS_PATH = "/v1/ws/markets"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def pick(book: Book, lo: float, hi: float):
    """(side, ask, depth, price_value, intent) for the favorite if it is priced in [lo, hi]."""
    if not book.bids or not book.offers:
        return None
    up_ask, up_depth = book.offers[0]
    bid, bid_depth = book.bids[0]
    down_ask = round(1 - bid, 4)
    if up_ask >= down_ask and lo <= up_ask <= hi:
        return "Up", up_ask, up_depth, up_ask, "ORDER_INTENT_BUY_LONG"
    if down_ask > up_ask and lo <= down_ask <= hi:
        return "Down", down_ask, bid_depth, bid, "ORDER_INTENT_BUY_SHORT"
    return None


def safety_check(side: str, book: Book, last_long: float | None, hist, now: float, args) -> str | None:
    """Return a reason to NOT buy, or None if the favorite is backed by a real, steady market.

    Why: the one loss so far (ETH, bought Up 0.95) was a 0.15/0.95 book: a high ask in an empty market,
    with the last trade at ~0.01 for Up. A favorite must be (1) in a tight book, (2) confirmed by actual
    trades, and (3) not falling. `hist` is [(t, long_mid)] oldest first."""
    if not book.bids or not book.offers:
        return "one-sided book"
    bid, offer = book.bids[0][0], book.offers[0][0]
    if offer - bid > args.max_spread:
        return f"wide spread {bid:.2f}/{offer:.2f}"
    sgn = (lambda x: x) if side == "Up" else (lambda x: 1 - x)
    if last_long is None:
        return "no last trade to confirm"
    if sgn(last_long) < args.min_ask - args.last_tol:
        return f"last trade {sgn(last_long):.2f} disagrees with quote"
    mids = [(t, sgn(m)) for t, m in hist]
    cur = sgn((bid + offer) / 2)
    ref = [m for t, m in mids if now - t >= args.drop_secs]
    if ref and ref[-1] - cur >= args.max_drop:
        return f"falling: {ref[-1]:.2f} -> {cur:.2f} in {args.drop_secs:g}s"
    recent = [m for t, m in mids if now - t <= 60]
    if recent and max(recent) - cur >= args.max_off_high:
        return f"{max(recent) - cur:.2f} below its 60s high"
    return None


def book_from_ws(md: dict) -> Book:
    lv = lambda rows: [(float(x["px"]["value"]), float(x["qty"])) for x in rows]  # noqa: E731
    return Book(md["marketSlug"], md.get("state", ""), sorted(lv(md.get("bids", [])), reverse=True),
                sorted(lv(md.get("offers", []))))


def settle_result(http: httpx.Client, ev_slug: str):
    """'Up'/'Down' once resolved; None if not yet (or rate limited)."""
    r = http.get(f"{GATEWAY}/v1/events/slug/{ev_slug}")
    if r.status_code != 200:
        return None
    m = r.json().get("event", r.json())["markets"][0]
    if m.get("status") != "MARKET_STATUS_RESOLVED":
        return None
    try:
        yes, no = (float(x) for x in json.loads(m["outcomePrices"]))
    except (KeyError, ValueError):
        return None
    return "Up" if yes > no else "Down" if no > yes else None


class Bot:
    def __init__(self, args, trader: Trader | None, http: httpx.Client, log) -> None:
        self.a, self.trader, self.http, self.log = args, trader, http, log
        self.books: dict[str, Book] = {}
        self.last: dict[str, float] = {}                    # slug -> last trade price (long scale)
        self.hist: dict[str, list] = {}                     # slug -> [(t, long_mid)]
        self.skipped: set = set()
        self.px: dict[str, dict[int, float]] = {a: {} for a in args.assets}   # 1-second index prices
        self.px_ok_at = 0.0
        self.px_try_at = 0.0
        self.px_busy = False
        self.attempts: dict[str, int] = {}
        self.ptb: dict[tuple[str, int], float] = {}        # exchange-published price to beat per (asset, window start)
        self.ptb_try: dict[tuple[str, int], float] = {}
        self.ptb_busy = False
        self.halted: dict[str, str] = {}                   # asset -> reason
        self.done: set[str] = set()                        # market slugs already decided
        self.pending: dict[str, dict] = {}                 # asset -> unsettled trade
        self.stats = {a: [0, 0, 0.0] for a in args.assets}  # wins, losses, pnl
        self.spent, self.n_live = 0.0, 0
        self._hb = 0.0
        self.stop_all = False                              # set by the first loss when --stop-on-any-loss
        self.load_state()

    def load_state(self) -> None:
        if self.a.fresh or not STATE.exists():
            return
        d = json.loads(STATE.read_text())
        self.pending, self.halted, self.stop_all = d["pending"], d["halted"], d["stop_all"]
        self.stats.update({k: v for k, v in d["stats"].items() if k in self.stats})
        self.done = set(d.get("done", []))
        self.spent, self.n_live = d.get("spent", 0.0), d.get("n_live", 0)
        if self.pending or self.halted or self.stop_all:
            self.say(f"resumed state: pending {list(self.pending)} halted {self.halted} stop_all={self.stop_all}")

    def save_state(self) -> None:
        STATE.write_text(json.dumps({"pending": self.pending, "halted": self.halted, "stop_all": self.stop_all,
                                     "stats": self.stats, "done": sorted(self.done)[-40:],
                                     "spent": self.spent, "n_live": self.n_live}))

    def emit(self, rec: dict) -> None:
        self.log.write(json.dumps(rec, default=str) + "\n")
        self.log.flush()

    def say(self, msg: str) -> None:
        print(f"[{ts()}] {msg}", flush=True)

    def summary(self) -> str:
        return " | ".join(f"{a} {w}W/{l}L {p:+.3f}" + (" HALTED" if a in self.halted else "")
                          for a, (w, l, p) in self.stats.items())

    # -- settlement ----------------------------------------------------------
    def settle(self, now: float) -> None:
        for asset, t in list(self.pending.items()):
            if now < t["next_check"]:
                continue
            t["next_check"] = now + 15
            res = settle_result(self.http, t["ev"])
            if res is None:
                if now > t["end"] + 900:
                    self.halted[asset] = "never settled; check the app"
                    self.say(f"{asset} {t['slug'][-6:]} unresolved after 15 min -> HALTED; check the app")
                    del self.pending[asset]
                continue
            won = res == t["side"]
            p = t["qty"] * (1 - t["ask"]) - t["fee"] if won else -(t["qty"] * t["ask"]) - t["fee"]
            st = self.stats[asset]
            st[0] += won
            st[1] += not won
            st[2] += p
            self.say(f"SETTLED {asset} {t['slug'][-6:]} bought {t['side']} -> {res}: "
                     f"{'WIN' if won else 'LOSS'} {p:+.3f}")
            if not won:
                self.halted[asset] = "lost; stopped betting this asset"
                self.say(f"{asset} HALTED after loss")
                if self.a.stop_on_any_loss:
                    self.stop_all = True
                    self.say("LOSS -> stopping all trading (--stop-on-any-loss)")
            self.emit({"settled": t["slug"], "asset": asset, "result": res, "won": won, "pnl": round(p, 4)})
            self.say("  " + self.summary())
            del self.pending[asset]
            self.save_state()

    # -- index price feed ------------------------------------------------------
    def _fetch_prices(self) -> None:
        now = int(time.time())
        r = self.http.get(f"{GATEWAY}/v1/asset-prices/history",
                          params=[("assets", f"crypto:{a}") for a in self.a.assets]
                          + [("from", now - 720), ("to", 0), ("bucket", "ASSET_PRICE_BUCKET_1S")])
        if r.status_code == 429:
            raise RuntimeError("429")
        r.raise_for_status()
        for srs in r.json()["series"]:
            d = self.px.setdefault(srs["asset"]["symbol"], {})
            for c in srs["candles"]:
                d[c["timestamp"]] = float(c["close"])
            for t in [t for t in d if t < now - 900]:
                del d[t]
        self.px_ok_at = time.time()

    async def _poll_prices(self) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._fetch_prices)
        except Exception as e:   # rate limited or a blip: back off and let the margin check fail closed meanwhile
            self.px_try_at = time.time() + (7.0 if "429" in str(e) else 2.0)
        finally:
            self.px_busy = False

    def _fetch_ptb(self, asset: str, start: int) -> None:
        r = self.http.get(f"{GATEWAY}/v1/events/slug/{event_slug(start, asset)}")
        if r.status_code == 429:
            raise RuntimeError("429")
        if r.status_code != 200:
            return
        m = r.json().get("event", r.json())["markets"][0]
        v = ((m.get("assetPriceTerms") or {}).get("priceToBeat") or {}).get("value")
        if v is not None:
            self.ptb[(asset, start)] = float(v)

    async def _poll_ptb(self, asset: str, start: int) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._fetch_ptb, asset, start)
        except Exception:
            self.ptb_try[(asset, start)] = time.time() + 6      # rate limited: wait a bit longer
        finally:
            self.ptb_busy = False

    def margin_view(self, asset: str, start: int, now: float):
        """How far ahead the index price is for this window, or a reason string if it cannot be judged."""
        d = self.px.get(asset) or {}
        if not d or now - self.px_ok_at > self.a.max_feed_age or now - max(d) > self.a.max_feed_age:
            return "no fresh index price"
        t_eff = max(d) + 1                      # first second not yet published
        end = start + 300
        left = end - t_eff
        if left < 1:
            return "window about to close"
        try:
            p0 = sum(d[start - 60 + i] for i in range(60)) / 60
            known = sum(d[end - 60 + i] for i in range(60 - left)) if left < 60 else 0.0
            prices = [d[t] for t in range(t_eff - 600, t_eff)]
        except KeyError:
            return "index price history incomplete"
        sg = sigma_ref(prices)
        if sg <= 0:
            return "no volatility reference"
        exact = (asset, start) in self.ptb          # the exchange's own price to beat, not my reconstruction of it
        if exact:
            p0 = self.ptb[(asset, start)]
        z = lead_z(p0, known, prices[-1], left, sg)
        return {"z": z, "left": left, "side": "Up" if z >= 0 else "Down", "exact": exact}

    # -- decisions -----------------------------------------------------------
    async def consider(self, asset: str, start: int, now: float) -> None:
        slug = updown_slug(start, asset)
        left = start + 300 - now
        if (self.stop_all or slug in self.done or asset in self.halted or asset in self.pending
                or not self.a.t_min <= left <= self.a.t_max):
            return
        book = self.books.get(slug)
        c = pick(book, self.a.min_ask, self.a.max_ask) if book and book.state == "MARKET_STATE_OPEN" else None
        if not c:
            return
        side, ask, depth, pv, intent = c
        why = safety_check(side, book, self.last.get(slug), self.hist.get(slug, []), now, self.a)
        mv = None
        if not why:   # the book looks sane: now check the index price agrees and the lead is big enough for the time left
            mv = self.margin_view(asset, start, now)
            if isinstance(mv, str):
                why = "margin: " + mv
            elif mv["side"] != side:
                why = f"margin: index price has {mv['side']} ahead (z={mv['z']:+.1f}), book favors {side}"
            else:
                mv["ub"] = reversal_ub(mv["left"], abs(mv["z"]), mv["exact"])
                mv["edge"] = edge_after_risk(ask, taker_fee(ask, 1.0), mv["ub"])
                if mv["edge"] < self.a.min_edge:
                    why = (f"margin: lead z={abs(mv['z']):.1f} with {mv['left']:.0f}s left can still reverse up to "
                           f"{mv['ub']:.1%}, edge {mv['edge'] * 100:+.1f}c at {ask:.2f}")
        if why:
            if (slug, why.split()[0]) not in self.skipped:
                self.skipped.add((slug, why.split()[0]))
                self.say(f"{asset} {slug[-6:]} T-{left:.0f}s {side} @ {ask:.2f} SKIPPED: {why}")
                self.emit({"asset": asset, "slug": slug, "t": round(now), "left": round(left), "skipped": why,
                           "side": side, "ask": ask, "bid": book.bids[0], "offer": book.offers[0]})
            return
        qty = math.floor(self.a.usd / ask * 100) / 100
        self.done.add(slug)
        self.save_state()
        if qty < 0.01 or qty > depth:
            self.say(f"{asset} {slug[-6:]} {side} @ {ask:.2f}: size {qty} vs depth {depth}; skip")
            return
        fee = taker_fee(ask, qty)
        self.say(f"{asset} {slug[-6:]} T-{left:.0f}s favorite {side} @ {ask:.2f} -> BUY {qty} "
                 f"(${qty * ask:.2f}) z={abs(mv['z']):.1f} reversal<={mv['ub']:.1%} edge {mv['edge'] * 100:+.1f}c "
                 f"{'LIVE' if self.a.live else 'dry'}")
        rec = {"asset": asset, "slug": slug, "t": round(now), "left": round(left), "side": side, "ask": ask,
               "qty": qty, "live": self.a.live, "bid": book.bids[0], "offer": book.offers[0],
               "margin": {k: round(v, 5) if isinstance(v, float) else v for k, v in mv.items()}}
        trade = {"slug": slug, "ev": event_slug(start, asset), "end": start + 300, "side": side, "ask": ask,
                 "qty": qty, "fee": fee, "next_check": start + 300 + 20}
        if self.a.live:
            if self.n_live >= self.a.max_trades or self.spent + self.a.usd > self.a.max_spend:
                self.say("  live caps reached; not trading")
                return
            # Offers flicker (one vanished within 50 ms of our order), so a limit one cent past the quote misses.
            # An IOC order pays only what is actually offered, so allow up to the highest price that still clears
            # the edge rule: price improvement is kept, and we never pay more than the model says is worth it.
            top = min(0.99, max(ask, max_price(mv["ub"], self.a.min_edge) or ask))
            if intent.endswith("LONG"):
                lim = math.floor(top * 100 + 1e-9) / 100
            else:   # Down is bought as a short: its price is 1 - the long price, so a higher Down price is a lower long price
                lim = max(0.01, math.ceil((1 - top) * 100 - 1e-9) / 100)
            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(None, self.trader.create,
                                                 limit_order(slug, intent, round(lim, 2), qty, FOK))
                o = await loop.run_in_executor(None, self.trader.order_outcome, res["id"])
            except Exception as e:  # the order may or may not exist: stop this asset, log loudly
                rec["error"] = str(e)
                self.halted[asset] = "unconfirmed order; check the app"
                self.say(f"  {asset} ORDER ERROR (asset HALTED): {e}")
                self.emit(rec)
                return
            rec["order"] = o
            got = float(o.get("cumQuantity") or 0)
            px = float((o.get("avgPx") or {}).get("value") or 0)
            # avgPx is on the long (Up) scale; a Down fill at 0.99 reports 0.01
            px = ask if px <= 0 else px if intent.endswith("LONG") else 1 - px
            comm = float((o.get("commissionNotionalTotalCollected") or {}).get("value") or 0)
            self.say(f"  order {res.get('id')} state={o.get('state')} filled {got} of {qty} @ {px:.4f} fee {comm:.4f}")
            if got <= 0:      # expired empty: look again on the next book update, up to --max-attempts per window
                n = self.attempts[slug] = self.attempts.get(slug, 0) + 1
                if n < self.a.max_attempts:
                    self.done.discard(slug)
                    self.say(f"  not filled (attempt {n}/{self.a.max_attempts}); will look again")
            if got > 0:
                self.spent += got * px
                self.n_live += 1
                trade.update(ask=px, qty=got, fee=comm)
                self.pending[asset] = trade
        else:
            self.pending[asset] = trade
        self.save_state()
        self.emit(rec)

    # -- stream --------------------------------------------------------------
    async def run(self) -> None:
        stop_at = time.time() + self.a.hours * 3600
        subscribed: set[int] = set()
        while time.time() < stop_at:
            try:
                async with websockets.connect("wss://api.polymarket.us" + WS_PATH,
                                              additional_headers=self.trader_signer.headers("GET", WS_PATH)) as ws:
                    subscribed.clear()
                    self.say("websocket connected")
                    while time.time() < stop_at:
                        now = time.time()
                        start = int(now // 300) * 300
                        if start not in subscribed:  # new window: subscribe to its four markets
                            subscribed.add(start)
                            await ws.send(json.dumps({"subscribe": {
                                "requestId": f"w{start}", "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                                "marketSlugs": [updown_slug(start, x) for x in self.a.assets]}}))
                        try:
                            raw = await asyncio.wait_for(ws.recv(), 1.0)
                            md = json.loads(raw).get("marketData")
                            if md:
                                b = book_from_ws(md)
                                self.books[b.slug] = b
                                lt = ((md.get("stats") or {}).get("lastTradePx") or {}).get("value")
                                if lt is not None:
                                    self.last[b.slug] = float(lt)
                                if b.bids and b.offers:
                                    h = self.hist.setdefault(b.slug, [])
                                    h.append((time.time(), (b.bids[0][0] + b.offers[0][0]) / 2))
                                    del h[:-400]
                        except asyncio.TimeoutError:
                            pass
                        now = time.time()
                        if not self.px_busy and now >= self.px_try_at:     # keep the 1-second index prices fresh
                            self.px_busy = True
                            self.px_try_at = now + self.a.feed_poll
                            asyncio.create_task(self._poll_prices())
                        if not self.ptb_busy and now - start >= 65:     # the exchange publishes it about a minute in
                            for a_ in self.a.assets:
                                key = (a_, start)
                                if key not in self.ptb and now - self.ptb_try.get(key, 0) > 8:
                                    self.ptb_try[key] = now
                                    self.ptb_busy = True
                                    asyncio.create_task(self._poll_ptb(a_, start))
                                    break
                        for asset in self.a.assets:
                            await self.consider(asset, start, now)
                        self.settle(now)
                        if now - self._hb > 5:
                            HEARTBEAT.write_text(str(now))
                            self._hb = now
                        if self.a.live and not self.pending and (
                                self.stop_all
                                or len(self.halted) == len(self.a.assets)
                                or self.n_live >= self.a.max_trades):
                            self.say("all assets halted or caps reached; done. " + self.summary())
                            return
            except Exception as e:  # any stream failure: reconnect; open trades keep being tracked
                self.say(f"websocket error ({type(e).__name__}: {e}); reconnecting in 3s")
                await asyncio.sleep(3)
        self.say("finished. " + self.summary())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="send real orders (default: dry run)")
    ap.add_argument("--assets", default="btc,eth,sol,xrp")
    ap.add_argument("--usd", type=float, default=1.0, help="stake per trade (max 5)")
    ap.add_argument("--min-ask", type=float, default=0.95)
    ap.add_argument("--max-ask", type=float, default=0.99)
    ap.add_argument("--t-max", type=float, default=100.0, help="start looking this many seconds before the end")
    ap.add_argument("--max-spread", type=float, default=0.08, help="skip books wider than this (bid/offer)")
    ap.add_argument("--last-tol", type=float, default=0.03, help="last trade must be within this of --min-ask")
    ap.add_argument("--drop-secs", type=float, default=20.0)
    ap.add_argument("--max-drop", type=float, default=0.03, help="skip if the favorite fell this much in --drop-secs")
    ap.add_argument("--max-off-high", type=float, default=0.05, help="skip if this far below its 60s high")
    ap.add_argument("--min-edge", type=float, default=0.005, help="required worst-case profit per share (dollars)")
    ap.add_argument("--max-attempts", type=int, default=3, help="orders per window per asset if the first expire empty")
    ap.add_argument("--max-feed-age", type=float, default=8.0, help="skip if the index price feed is older than this")
    ap.add_argument("--feed-poll", type=float, default=3.0, help="seconds between index price fetches")
    ap.add_argument("--t-min", type=float, default=15.0, help="stop looking this many seconds before the end")
    ap.add_argument("--max-trades", type=int, default=60, help="live: stop after this many filled orders")
    ap.add_argument("--max-spend", type=float, default=100.0, help="live: stop after this much total stake")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--stop-on-any-loss", action="store_true", help="stop all assets after the first loss")
    ap.add_argument("--fresh", action="store_true", help="ignore saved state (data/pmus/state.json)")
    ap.add_argument("--slip", type=float, default=0.01, help="limit this far through the touch (fills at book price)")
    args = ap.parse_args()
    args.assets = [x.strip().lower() for x in args.assets.split(",") if x.strip()]
    if not 0 < args.usd <= 5:
        sys.exit("--usd must be in (0, 5]")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    kid, sec = os.environ.get("POLYMARKET_KEY_ID"), os.environ.get("POLYMARKET_SECRET_KEY")
    if not (kid and sec):
        sys.exit("needs POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY (the market stream is authenticated)")
    signer = Signer(kid, sec)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] late-favorite ({'LIVE' if args.live else 'dry-run'}) {','.join(args.assets)} "
          f"ask {args.min_ask}-{args.max_ask} T-{args.t_max:g}..{args.t_min:g}s ${args.usd}/trade/asset, "
          f"stop asset on loss", flush=True)
    with httpx.Client(timeout=15.0) as http, LOG.open("a") as log:
        bot = Bot(args, Trader(signer, http) if args.live else None, http, log)
        bot.trader_signer = signer
        asyncio.run(bot.run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

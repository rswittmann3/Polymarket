"""Late-window favorite buyer for Polymarket US BTC 5-minute Up/Down markets.

Recreates what wallet 0x50f7 does on polymarket.com: in the last ~2 minutes of a window, buy
whichever side is already priced 0.95-0.99, hold to settlement, repeat. Per $1 that wins you earn a
cent or two; one reversal loses the stake, so it only pays if the favorite loses less often than
roughly (1 - ask) / 1. A 7-day backtest on polymarket.com could not tell that apart from break-even.

DRY RUN BY DEFAULT (logs what it would buy). `--live` sends fill-or-kill orders with real money.

    python tools/pmus_late_favorite.py                       # watch + log decisions
    python tools/pmus_late_favorite.py --live --usd 1 --max-spend 5 --max-trades 5

Price convention (docs): `price.value` is always the Up/long price. Buying Down at X is a
BUY_SHORT with price.value = 1 - X, i.e. hitting the best long bid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from pm_edges.pmus import GATEWAY, Signer, Trader, get_book, limit_order, taker_fee, updown_slug

LOG = Path("data/pmus/late_favorite.jsonl")
FOK = "TIME_IN_FORCE_FILL_OR_KILL"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def pick(book, lo: float, hi: float):
    """Return (side, ask, depth, price_value, intent) for the favorite if it is priced in [lo, hi]."""
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


def settle_result(http: httpx.Client, slug: str):
    """'Up'/'Down' once the market has resolved, else None."""
    ev = slug.removeprefix("cpc-")
    r = http.get(f"{GATEWAY}/v1/events/slug/{ev}")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="send real orders (default: dry run)")
    ap.add_argument("--usd", type=float, default=1.0, help="stake per trade (max 5)")
    ap.add_argument("--min-ask", type=float, default=0.95)
    ap.add_argument("--max-ask", type=float, default=0.99)
    ap.add_argument("--t-max", type=float, default=120.0, help="start looking this many seconds before the end")
    ap.add_argument("--t-min", type=float, default=25.0, help="stop looking this many seconds before the end")
    ap.add_argument("--max-trades", type=int, default=5, help="live: stop after this many orders")
    ap.add_argument("--max-spend", type=float, default=5.0, help="live: stop after this much total stake")
    ap.add_argument("--hours", type=float, default=2.0, help="stop after this long")
    ap.add_argument("--poll", type=float, default=1.5)
    ap.add_argument("--slip", type=float, default=0.01, help="limit this far through the touch (fills still at the book price)")
    args = ap.parse_args()
    if not 0 < args.usd <= 5:
        sys.exit("--usd must be in (0, 5]")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    kid, sec = os.environ.get("POLYMARKET_KEY_ID"), os.environ.get("POLYMARKET_SECRET_KEY")
    trader = Trader(Signer(kid, sec)) if kid and sec else None
    if args.live and trader is None:
        sys.exit("--live needs POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] late-favorite ({'LIVE' if args.live else 'dry-run'}) ask {args.min_ask}-{args.max_ask} "
          f"T-{args.t_max:g}..{args.t_min:g}s ${args.usd}/trade cap ${args.max_spend} / {args.max_trades} trades", flush=True)
    done: set[str] = set()
    pending: list[dict] = []   # trades awaiting settlement
    spent, n_live, stop_at = 0.0, 0, time.time() + args.hours * 3600
    wins = losses = 0
    pnl = 0.0

    with httpx.Client(timeout=15.0) as http, LOG.open("a") as log:
        def emit(rec: dict) -> None:
            log.write(json.dumps(rec, default=str) + "\n")
            log.flush()

        while time.time() < stop_at:
            now = time.time()
            start = int(now // 300) * 300
            slug, left = updown_slug(start), start + 300 - now

            # settle finished trades
            for t in list(pending):
                if now > t["end"] + 20:
                    res = settle_result(http, t["slug"])
                    if res is None:
                        if now > t["end"] + 600:
                            print(f"[{ts()}] {t['slug'][-6:]} still unresolved after 10 min; check the app", flush=True)
                            pending.remove(t)
                        continue
                    won = res == t["side"]
                    p = t["qty"] * (1 - t["ask"]) - t["fee"] if won else -(t["qty"] * t["ask"]) - t["fee"]
                    pnl += p
                    wins += won
                    losses += not won
                    print(f"[{ts()}] SETTLED {t['slug'][-6:]} bought {t['side']} -> {res}: {'WIN' if won else 'LOSS'} "
                          f"{p:+.3f}  | session {wins}W/{losses}L {pnl:+.3f}", flush=True)
                    emit({"settled": t["slug"], "result": res, "won": won, "pnl": round(p, 4)})
                    pending.remove(t)

            if args.live and (n_live >= args.max_trades or spent + args.usd > args.max_spend) and not pending:
                print(f"[{ts()}] live caps reached, done. session {wins}W/{losses}L {pnl:+.3f}", flush=True)
                return

            if slug not in done and args.t_min <= left <= args.t_max:
                try:
                    book = get_book(http, slug)
                except httpx.HTTPError as e:
                    print(f"[{ts()}] book error {e}", flush=True)
                    time.sleep(5)
                    continue
                c = pick(book, args.min_ask, args.max_ask) if book and book.open else None
                if c:
                    side, ask, depth, pv, intent = c
                    qty = math.floor(args.usd / ask * 100) / 100
                    fee = taker_fee(ask, qty)
                    if qty < 0.01 or qty > depth:
                        print(f"[{ts()}] {slug[-6:]} {side} {ask:.2f}: size {qty} vs depth {depth}; skip", flush=True)
                        done.add(slug)
                    else:
                        print(f"[{ts()}] {slug[-6:]} T-{left:.0f}s favorite {side} @ {ask:.2f} "
                              f"-> BUY {qty} (${qty * ask:.2f}, fee~${fee:.3f}) {'LIVE' if args.live else 'dry'}", flush=True)
                        rec = {"slug": slug, "t": round(now), "left": round(left), "side": side, "ask": ask, "qty": qty,
                               "fee": round(fee, 4), "live": args.live, "book_bid": book.bids[0], "book_offer": book.offers[0]}
                        done.add(slug)
                        if args.live:
                            try:
                                lim = min(0.99, pv + args.slip) if intent.endswith("LONG") else max(0.01, pv - args.slip)
                                res = trader.create(limit_order(slug, intent, round(lim, 2), qty, FOK))
                                o = trader.order_outcome(res["id"])
                                rec["result"], rec["order"] = res, o
                                got = float(o.get("cumQuantity") or 0)
                                px = float((o.get("avgPx") or {}).get("value") or ask)
                                comm = float((o.get("commissionNotionalTotalCollected") or {}).get("value") or 0)
                                print(f"[{ts()}]   order {res.get('id')} state={o.get('state')} filled {got} of {qty} "
                                      f"@ {px:.4f} fee {comm:.4f}", flush=True)
                                if got > 0:
                                    spent += got * px
                                    n_live += 1
                                    pending.append({"slug": slug, "end": start + 300, "side": side, "ask": px, "qty": got, "fee": comm})
                            except Exception as e:  # keep running; the order may or may not exist, so log it loudly
                                rec["error"] = str(e)
                                print(f"[{ts()}]   ORDER ERROR: {e}", flush=True)
                        else:
                            pending.append({"slug": slug, "end": start + 300, "side": side, "ask": ask, "qty": qty, "fee": fee})
                        emit(rec)
            time.sleep(args.poll)
    print(f"[{ts()}] finished. session {wins}W/{losses}L {pnl:+.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

"""Place a tiny real order on a Polymarket US BTC 5-minute market, to prove the plumbing works.

DRY RUN BY DEFAULT: finds the open window, reads the book, prints the order it would send (and, if
keys are set, asks the exchange to preview it and shows your balance). Nothing is sent until --live.

    export POLYMARKET_KEY_ID=...        # or put both in .env (gitignored)
    export POLYMARKET_SECRET_KEY=...    # from polymarket.us/developer; shown once
    python tools/pmus_test_trade.py                 # dry run
    python tools/pmus_test_trade.py --live          # buy ~$1 of Up (FOK), then sell it back
    python tools/pmus_test_trade.py --live --hold   # keep it to settlement instead of selling back

This is a plumbing test, not the arbitrage: on Polymarket US Up-ask + Down-ask = 1 + spread, so a
buy-both gap cannot exist. A round trip costs the spread plus two taker fees (a few cents on $1).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import httpx

from pm_edges.pmus import Signer, Trader, get_book, limit_order, taker_fee, updown_slug

MAX_USD = 5.0


def find_open_window(http: httpx.Client, min_left: float):
    now = time.time()
    start = int(now // 300) * 300
    slug = updown_slug(start)
    try:
        book = get_book(http, slug)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 429:
            raise
        time.sleep(5)  # public gateway rate limit
        return None
    left = start + 300 - now
    if book and book.open and left >= min_left and book.offers and book.bids:
        return slug, book, left
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="actually send orders")
    ap.add_argument("--hold", action="store_true", help="don't sell back; hold to settlement")
    ap.add_argument("--usd", type=float, default=1.0, help=f"approx dollars to spend (max {MAX_USD})")
    ap.add_argument("--min-left", type=float, default=45.0, help="seconds that must remain in the window")
    args = ap.parse_args()
    if not 0 < args.usd <= MAX_USD:
        sys.exit(f"--usd must be in (0, {MAX_USD}]")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key, secret = os.environ.get("POLYMARKET_KEY_ID"), os.environ.get("POLYMARKET_SECRET_KEY")
    trader = Trader(Signer(key, secret)) if key and secret else None
    if args.live and trader is None:
        sys.exit("--live needs POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY")

    with httpx.Client(timeout=15.0) as http:
        found = None
        while found is None:
            found = find_open_window(http, args.min_left)
            if found is None:
                # too close to the end of this window: sleep until the next one has opened
                time.sleep(max(2.0, (int(time.time() // 300) + 1) * 300 + 3 - time.time()) if time.time() % 300 > 300 - args.min_left else 3)
        slug, book, left = found
        ask, ask_qty = book.offers[0]
        bid = book.bids[0][0]
        qty = math.floor(args.usd / ask * 100) / 100
        if qty < 0.01 or qty > ask_qty:
            sys.exit(f"cannot size {args.usd} USD at ask {ask} (top-of-book qty {ask_qty})")
        fee = taker_fee(ask, qty)
        print(f"{slug}  {left:.0f}s left  bid {bid:.2f} / ask {ask:.2f}  (spread {ask - bid:.2f})")
        print(f"BUY {qty} Up @ {ask:.2f} FOK = ${qty * ask:.2f} + est. fee ${fee:.3f}"
              f"; round trip est. cost ${qty * (ask - bid) + fee + taker_fee(bid, qty):.3f}")
        buy = limit_order(slug, "ORDER_INTENT_BUY_LONG", ask, qty, "TIME_IN_FORCE_FILL_OR_KILL")

        if trader is not None:
            print("balances:", trader.balances())
            print("preview:", trader.preview({k: buy[k] for k in ("marketSlug", "intent", "type", "price", "quantity")}))
        if not args.live:
            print("dry run: nothing sent. Add --live to place it.")
            return

        res = trader.create(buy)
        print("buy result:", res)
        if args.hold:
            print("holding to settlement.")
            return
        book2 = get_book(http, slug)
        if not (book2 and book2.open and book2.bids):
            print("book closed or empty; use close_position / the app to exit.")
            return
        sell = limit_order(slug, "ORDER_INTENT_SELL_LONG", book2.bids[0][0], qty, "TIME_IN_FORCE_FILL_OR_KILL")
        try:
            print("sell result:", trader.create(sell))
        except RuntimeError as e:
            print("limit sell failed:", e, "-> falling back to close_position")
            print("close result:", trader.close_position(slug))


if __name__ == "__main__":
    main()

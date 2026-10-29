"""Poll Polymarket's Bitcoin Up/Down 5-minute markets and buy both sides when it locks in a profit.

A pair (1 Up + 1 Down) always pays $1. If the executable asks plus the taker fees come to less
than $1, buying both is a risk-free gain on paper. This does NOT trigger on "Up + Down < 100%"
alone: fees are about 3.5c per pair near 50/50, so the check is `1 - asks - fees >= --min-edge`.

DRY RUN BY DEFAULT. Nothing is sent unless you pass --live and set the wallet env vars.

    python tools/btc5m_arb.py                       # watch + log, no orders
    python tools/btc5m_arb.py --once                # one scan of the current and next window
    python tools/btc5m_arb.py --live --max-shares 10 --max-spend 25

Live mode needs `pip install py-clob-client` and in the environment (or .env):
    PM_PRIVATE_KEY   private key of the signing wallet
    PM_FUNDER        the Polymarket proxy/funder address holding the USDC
    PM_SIGNATURE_TYPE  0 EOA, 1 email/Magic proxy (default), 2 Safe

Both legs go out as fill-or-kill limit orders at the worst price the book walk needs. If one leg
fills and the other does not you are left holding a one-sided bet; the tool then tries to sell
that leg back and says so loudly. That legging risk is real and not removable here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from pm_edges.btc5m import (Market, Opportunity, WINDOW, fetch_asks, fetch_market, find_opportunity,
                            window_start)

LOG = Path("data/btc5m/polls.jsonl")
CLOB_HOST = "https://clob.polymarket.com"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


class Trader:
    """Thin wrapper over py-clob-client so the import stays optional."""

    def __init__(self) -> None:
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError:
            sys.exit("--live needs: pip install py-clob-client")
        key, funder = os.environ.get("PM_PRIVATE_KEY"), os.environ.get("PM_FUNDER")
        if not key or not funder:
            sys.exit("--live needs PM_PRIVATE_KEY and PM_FUNDER in the environment")
        self._OrderArgs, self._OrderType, self.BUY, self.SELL = OrderArgs, OrderType, BUY, SELL
        self.c = ClobClient(CLOB_HOST, chain_id=137, key=key, funder=funder,
                            signature_type=int(os.environ.get("PM_SIGNATURE_TYPE", "1")))
        self.c.set_api_creds(self.c.create_or_derive_api_creds())

    def fok(self, token: str, side: str, price: float, size: float) -> dict:
        order = self.c.create_order(self._OrderArgs(token_id=token, price=round(price, 2),
                                                    size=round(size, 2), side=side))
        return self.c.post_order(order, self._OrderType.FOK)


def filled(resp: dict) -> bool:
    return bool(resp) and bool(resp.get("success")) and resp.get("status") in ("matched", "MATCHED")


def execute(trader: Trader, m: Market, opp: Opportunity) -> dict:
    """Send both legs at once. Unwind a lone fill. Returns a record for the log."""
    legs = {"up": (m.up_token, opp.up_limit), "down": (m.down_token, opp.down_limit)}
    with ThreadPoolExecutor(2) as pool:
        futs = {k: pool.submit(_safe, trader, tok, trader.BUY, px, opp.shares) for k, (tok, px) in legs.items()}
        res = {k: f.result() for k, f in futs.items()}
    ok = {k: filled(r) for k, r in res.items()}
    rec = {"results": res, "filled": ok, "unwind": None}
    if ok["up"] != ok["down"]:
        k = "up" if ok["up"] else "down"
        print(f"[{ts()}] !!! ONE LEG ONLY: {k} filled, other did not. Selling {k} back.", flush=True)
        # Limit of 0.01 on a FOK sell takes whatever the bids will pay for the full size.
        rec["unwind"] = _safe(trader, legs[k][0], trader.SELL, 0.01, opp.shares)
        print(f"[{ts()}] unwind result: {rec['unwind']}", flush=True)
    return rec


def _safe(trader, token, side, price, size):
    try:
        return trader.fok(token, side, price, size)
    except Exception as e:  # a rejected FOK raises in py-clob-client; keep it as data
        return {"success": False, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="actually place orders (default: dry run)")
    ap.add_argument("--once", action="store_true", help="scan once and exit")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    ap.add_argument("--min-edge", type=float, default=0.005,
                    help="required profit per pair in USDC AFTER fees (cushion for rounding/slippage)")
    ap.add_argument("--max-shares", type=float, default=10.0, help="pairs per window")
    ap.add_argument("--max-spend", type=float, default=25.0, help="stop after this much total USDC committed")
    ap.add_argument("--min-seconds-left", type=float, default=20.0, help="don't trade this close to the window end")
    ap.add_argument("--next", dest="next_window", action="store_true", default=True)
    ap.add_argument("--no-next", dest="next_window", action="store_false", help="only the current window")
    args = ap.parse_args()

    load_env()
    trader = Trader() if args.live else None
    LOG.parent.mkdir(parents=True, exist_ok=True)
    mode = "LIVE" if args.live else "dry-run"
    print(f"[{ts()}] btc5m arb ({mode}) min-edge={args.min_edge} max-shares={args.max_shares} "
          f"max-spend={args.max_spend}", flush=True)

    traded: set[str] = set()
    spent = 0.0
    markets: dict[int, Market | None] = {}
    with httpx.Client(timeout=10.0) as http, LOG.open("a") as log:
        while True:
            now = time.time()
            start = window_start(now)
            starts = [start] + ([start + WINDOW] if args.next_window else [])
            for s in starts:
                try:
                    if s not in markets or markets[s] is None:
                        markets[s] = fetch_market(http, s)
                    m = markets[s]
                    if m is None or not m.accepting:
                        continue
                    up, down = fetch_asks(http, m.up_token), fetch_asks(http, m.down_token)
                except httpx.HTTPError as e:
                    print(f"[{ts()}] network error: {e}", flush=True)
                    continue
                if not up or not down:
                    continue
                opp = find_opportunity(up, down, fee_rate=m.fee_rate, fee_exponent=m.fee_exponent,
                                       min_edge=args.min_edge, max_shares=args.max_shares,
                                       min_shares=m.min_size)
                left = m.end_ts - now
                row = {"t": round(now, 1), "slug": m.slug, "ask_up": up[0][0], "ask_down": down[0][0],
                       "sum": round(up[0][0] + down[0][0], 4), "opp": bool(opp), "left": round(left)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[{ts()}] {m.slug[-10:]} up {up[0][0]:.2f} + down {down[0][0]:.2f} = {row['sum']:.2f}"
                      f"  left {left:4.0f}s  {'ARB' if opp else '-'}", flush=True)
                if not opp or m.slug in traded:
                    continue
                if left < args.min_seconds_left:
                    print(f"[{ts()}]   skip: {left:.0f}s left", flush=True)
                    continue
                if spent + opp.cost > args.max_spend:
                    print(f"[{ts()}]   skip: would exceed --max-spend ({spent + opp.cost:.2f})", flush=True)
                    continue
                print(f"[{ts()}]   {opp.shares:g} pairs: cost ${opp.cost:.2f} (fees ${opp.fee:.3f}) "
                      f"-> profit ${opp.profit:.3f} ({opp.profit_per_share:.3f}/pair)", flush=True)
                traded.add(m.slug)
                if trader is None:
                    log.write(json.dumps({"slug": m.slug, "dry_run_trade": opp.__dict__}) + "\n")
                    continue
                spent += opp.cost
                rec = execute(trader, m, opp)
                log.write(json.dumps({"slug": m.slug, "trade": rec}, default=str) + "\n")
                log.flush()
                print(f"[{ts()}]   filled up={rec['filled']['up']} down={rec['filled']['down']}", flush=True)
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

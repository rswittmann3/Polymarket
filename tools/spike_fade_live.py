"""Tiny live trial of the long-dated spike fade on Polymarket US: BUY NO after a YES price spike, hold to resolution.

Hard limits (all enforced in code, not by convention):
  * at most MAX_POSITIONS open positions and MAX_TOTAL_USD total stake over the whole trial
  * one position per market and per event
  * stake per position <= --usd (default 1.0, hard cap 2.0)
  * fill-or-kill taker order at the best bid of the long side only; nothing rests on the book
  * re-reads the live book right before ordering and refuses thin books / wide spreads / a spike that has already faded
  * DRY RUN BY DEFAULT; `--live` is required to send an order. Never calls cancel_all.

    python tools/spike_fade_live.py enter --slug <market-slug> [--live]
    python tools/spike_fade_live.py status          # mark open positions to the live book
    python tools/spike_fade_live.py stop            # write the stop flag (refuses further entries)

Price convention: the book is for the YES (long) side. Buying NO at x is BUY_SHORT with long price 1 - x, i.e. hitting
the best YES bid.
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

from pm_edges.pmus import Signer, Trader, get_book, limit_order, taker_fee

STATE = Path("data/us_all/live_trial.json")
SIGNALS = Path("data/us_all/signals.jsonl")
FOK = "TIME_IN_FORCE_FILL_OR_KILL"
MAX_POSITIONS = 5
MAX_TOTAL_USD = 5.0
MAX_SPREAD = 0.02          # long-side bid-ask
MIN_TOUCH_MULT = 1.5       # shares at the best bid must be >= this x the shares we want
MIN_SPIKE_LEFT = 0.08      # YES mid must still be at least this far above its level 24h before the signal


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"stop": False, "positions": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def signal_for(slug: str) -> dict | None:
    if not SIGNALS.exists():
        return None
    hit = None
    for line in SIGNALS.open():
        r = json.loads(line)
        if r["slug"] == slug:
            hit = r
    return hit


def trader() -> Trader:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    kid, sec = os.environ.get("POLYMARKET_KEY_ID"), os.environ.get("POLYMARKET_SECRET_KEY")
    if not (kid and sec):
        sys.exit("needs POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY")
    return Trader(Signer(kid, sec))


def enter(a) -> None:
    st = load_state()
    if st["stop"]:
        sys.exit("stop flag is set; no entries")
    if a.usd <= 0 or a.usd > 2.0:
        sys.exit("--usd must be in (0, 2]")
    open_pos = [p for p in st["positions"] if p.get("qty", 0) > 0]
    if len(open_pos) >= MAX_POSITIONS:
        sys.exit(f"already {len(open_pos)} positions (max {MAX_POSITIONS})")
    spent = sum(p["cost"] for p in open_pos)
    if spent + a.usd > MAX_TOTAL_USD + 1e-9:
        sys.exit(f"total stake would exceed ${MAX_TOTAL_USD}")
    sig = signal_for(a.slug)
    if sig is None:
        sys.exit("no logged signal for that slug; the trial only trades logged signals")
    if any(p["slug"] == a.slug or p["ev"] == sig["ev"] for p in open_pos):
        sys.exit("already have a position in that market/event")

    with httpx.Client(timeout=15.0) as http:
        book = get_book(http, a.slug)
    if not book or not book.open or not book.bids or not book.offers:
        sys.exit("book missing or not open")
    bid, ask = book.bids[0][0], book.offers[0][0]
    spread, touch = ask - bid, book.bids[0][1]
    mid = (bid + ask) / 2
    no_px = 1 - bid
    qty = math.floor(a.usd / (no_px + taker_fee(no_px, 1)) * 100) / 100
    left = mid - sig["past_mid"]
    print(f"{a.slug}: YES {bid:.3f}/{ask:.3f} spread {spread:.3f}, NO buy px {no_px:.3f}, touch {touch:.2f} sh, "
          f"spike left {left:+.3f}, qty {qty}")
    for bad, why in ((spread > MAX_SPREAD + 1e-9, f"spread {spread:.3f} > {MAX_SPREAD}"),
                     (touch < qty * MIN_TOUCH_MULT, f"touch {touch:.2f} < {MIN_TOUCH_MULT}x qty {qty}"),
                     (left < MIN_SPIKE_LEFT, f"spike faded ({left:+.3f} < {MIN_SPIKE_LEFT})"),
                     (not 0.10 <= mid <= 0.90, f"YES mid {mid:.3f} outside 0.10-0.90"),
                     (qty < 0.5, "qty too small")):
        if bad:
            sys.exit(f"REFUSED: {why}")
    order = limit_order(a.slug, "ORDER_INTENT_BUY_SHORT", bid, qty, FOK)
    if not a.live:
        print("DRY RUN, order would be:", json.dumps(order))
        return
    t = trader()
    pv = t.preview(order)
    print("preview:", json.dumps(pv)[:400])
    rec = {"slug": a.slug, "ev": sig["ev"], "t": round(time.time()), "yes_bid": bid, "yes_ask": ask, "no_px_quote": no_px,
           "qty_req": qty, "signal": sig, "qty": 0, "cost": 0.0}
    try:
        res = t.create(order)
        o = t.order_outcome(res["id"])
    except Exception as e:                      # the order may or may not exist: stop everything until a human looks
        rec["error"] = str(e)
        st["stop"] = True
        st["positions"].append(rec)
        save_state(st)
        sys.exit(f"ORDER ERROR (trial halted, check the app): {e}")
    got = float(o.get("cumQuantity") or 0)
    px_long = float((o.get("avgPx") or {}).get("value") or 0)
    px = (1 - px_long) if px_long > 0 else no_px
    fee = float((o.get("commissionNotionalTotalCollected") or {}).get("value") or 0)
    rec.update(order=o, qty=got, avg_no_px=px, fee=fee, cost=got * px + fee, state=o.get("state"))
    st["positions"].append(rec)
    save_state(st)
    print(f"order {res.get('id')} state={o.get('state')} filled {got} of {qty} @ NO {px:.4f}, fee {fee:.4f}, cost ${rec['cost']:.3f}")


def status(_a) -> None:
    st = load_state()
    print(f"stop flag: {st['stop']}; {len(st['positions'])} position records")
    tot_cost = tot_val = 0.0
    with httpx.Client(timeout=15.0) as http:
        for p in st["positions"]:
            if p.get("qty", 0) <= 0:
                print(f"  {p['slug']}: no fill ({p.get('error') or p.get('state')})")
                continue
            time.sleep(2.1)
            b = get_book(http, p["slug"])
            if not b or not b.open:
                print(f"  {p['slug']}: book {'missing' if not b else b.state}: market may have resolved, CHECK THE APP; cost ${p['cost']:.3f}")
                continue
            no_bid = 1 - b.offers[0][0] if b.offers else None     # what we could sell NO for right now
            val = p["qty"] * no_bid if no_bid is not None else 0.0
            tot_cost += p["cost"]
            tot_val += val
            print(f"  {p['slug']}: {p['qty']} sh bought NO @ {p['avg_no_px']:.3f}, NO bid now {no_bid}, MTM {val - p['cost']:+.3f}")
    print(f"total cost ${tot_cost:.3f}, marked value ${tot_val:.3f}, MTM P&L {tot_val - tot_cost:+.3f} (kill rule: stop if <= -3.00)")


def stop(_a) -> None:
    st = load_state()
    st["stop"] = True
    save_state(st)
    print("stop flag written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enter", "status", "stop"])
    ap.add_argument("--slug")
    ap.add_argument("--usd", type=float, default=1.0)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.cmd == "enter" and not a.slug:
        sys.exit("--slug required")
    {"enter": enter, "status": status, "stop": stop}[a.cmd](a)

"""Live check: are the player props offered on Polymarket, and how much would it cost to enter?

    python tools/prop_liquidity.py --away WAS --home SF --date 2026-10-19 --kickoff 20:15 \
        [--stakes 5,25,50] [--out data/board/prop_liquidity.json]

Fetches fresh Gamma markets and CLOB order books (no Odds API credits), then for
every player-prop outcome walks the ask book for each stake and reports the
average fill, slippage against the best ask, and what could not be filled. Exit
slippage is measured by selling the same number of shares into the bids.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date, time
from pathlib import Path

from pm_edges.config import load_config
from pm_edges.pipeline import GameSpec, collect
from pm_edges.polymarket import effective_fee_rate, fee_per_share, walk_book


def raw_markets(raw_dir: str) -> dict[str, dict]:
    out = {}
    for path in glob.glob(f"{raw_dir}/gamma-api__events__*.json"):
        for ev in json.load(open(path))["body"]:
            for m in ev.get("markets") or []:
                out[str(m.get("id"))] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--date", required=True, type=date.fromisoformat)
    ap.add_argument("--kickoff", type=time.fromisoformat)
    ap.add_argument("--stakes", default="5,25,50", help="order sizes in dollars; none may exceed --max-stake")
    ap.add_argument("--max-stake", type=float, default=50.0)
    ap.add_argument("--min-liquidity", type=float, default=0.0)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/board/prop_liquidity.json"))
    args = ap.parse_args()
    stakes = [float(s) for s in args.stakes.split(",")]
    if max(stakes) > args.max_stake:
        ap.error(f"stakes above ${args.max_stake:g} are not modelled (raise --max-stake to override)")

    cfg = load_config(data_dir=args.data_dir)
    spec = GameSpec("nfl", args.away, args.home, args.date, args.kickoff)
    run = collect(spec, cfg, books=False, use_clob=True)
    raw = raw_markets(run.raw_dir)

    rows = []
    for m in run.markets:
        if m.market_type != "player_prop" and "Total Touchdowns" in m.question:
            continue
        r = raw.get(m.market_id, {})
        if (r.get("sportsMarketType") or "") in ("team_first_touchdowns",):
            continue
        if not r.get("sportsMarketType") or "_" not in r.get("sportsMarketType", "") and r.get("sportsMarketType") in (
                "moneyline", "spreads", "totals"):
            continue
        if m.market_type not in ("player_prop",) and ":" not in m.question:
            continue
        rate, _ = effective_fee_rate(m, cfg)
        for o in m.outcomes:
            sel = o.selection
            fills = {}
            for s in stakes:
                w = walk_book(o.ask_levels, s)
                exit_w = walk_book(o.bid_levels, w["shares"] * (o.bid_levels[0][0] if o.bid_levels else 0)) if w["shares"] else None
                fills[str(int(s))] = {
                    "filled": w["filled"], "spent": round(w["spent"], 2), "shares": round(w["shares"], 1),
                    "avg": None if w["avg"] is None else round(w["avg"], 4),
                    "slip_c": None if (w["avg"] is None or o.ask is None) else round((w["avg"] - o.ask) * 100, 2),
                    "last": w["last"],
                }
            spread_c = None if (o.ask is None or o.bid is None) else round((o.ask - o.bid) * 100, 1)
            depth_best = None if o.ask is None or not o.ask_levels else round(o.ask_levels[0][0] * o.ask_levels[0][1], 2)
            within = lambda c: round(sum(p * sz for p, sz in o.ask_levels if o.ask is not None and p <= o.ask + c / 100 + 1e-9), 2)
            bid_depth = round(sum(p * sz for p, sz in o.bid_levels[:3]), 2) if o.bid_levels else 0.0
            flags = []
            if not r.get("active", True) or r.get("closed"):
                flags.append("market closed")
            if r.get("acceptingOrders") is False:
                flags.append("not accepting orders")
            if o.ask is None:
                flags.append("no ask: cannot buy")
            if o.ask is not None and o.ask >= 0.97:
                flags.append("ask 97¢ or more: nothing left to win")
            if depth_best is not None and depth_best < stakes[-1]:
                flags.append(f"only ${depth_best:,.0f} at best ask")
            if spread_c is not None and spread_c > 3:
                flags.append(f"{spread_c:g}¢ spread")
            if o.bid is None or not o.bid_levels:
                flags.append("no bid: exit only by holding")
            elif bid_depth < 50:
                flags.append(f"thin bids (${bid_depth:,.0f} in top 3)")
            top = fills.get(str(int(stakes[-1])))
            if top and not top["filled"]:
                flags.append(f"cannot fill ${stakes[-1]:g}")
            elif top and top["slip_c"] is not None and top["slip_c"] > 1:
                flags.append(f"${stakes[-1]:g} slips {top['slip_c']:g}¢")
            if m.liquidity < 1000:
                flags.append(f"market liquidity ${m.liquidity:,.0f}")
            rows.append({
                "market_id": m.market_id, "slug": m.slug, "question": m.question,
                "outcome": o.name, "player": m.question.split(":")[0] if ":" in m.question else "",
                "smt": r.get("sportsMarketType"), "line": m.line,
                "selection": sel.label() if sel else None,
                "ask": o.ask, "bid": o.bid, "spread_c": spread_c,
                "depth_best_usd": depth_best, "depth_1c_usd": within(1), "depth_3c_usd": within(3),
                "bid_depth_top3_usd": bid_depth,
                "fee_per_share": round(fee_per_share(o.ask, rate, m.fee_exponent), 4) if o.ask else None,
                "liquidity": round(m.liquidity), "volume": round(m.volume),
                "accepting_orders": r.get("acceptingOrders"), "order_min_size": r.get("orderMinSize"),
                "tick": r.get("orderPriceMinTickSize"), "fills": fills, "flags": flags,
                "dnp_rule": "Under/No" if "inactive or does not play, this market will resolve to" in (r.get("description") or "") else None,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"as_of": run.fetched_at.isoformat(), "stakes": stakes, "rows": rows}, separators=(",", ":")))
    print(f"{len(rows)} outcomes written to {args.out} (books as of {run.fetched_at:%H:%M:%S} UTC)")


if __name__ == "__main__":
    main()

"""Join sportsbook fair values (data/board/edges.json) with live order-book depth
(data/board/prop_liquidity.json) into the rows the prop brief page embeds.

    python tools/prop_brief_data.py --out props.json

Run tools/export_board.py and tools/prop_liquidity.py first.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RATE = 0.03   # Polymarket NFL taker fee rate, per the markets' own feeSchedule
STATS = {"pass_attempts": "Passing attempts", "pass_yds": "Passing yards", "pass_tds": "Passing TDs", "pass_completions": "Completions",
         "rush_yds": "Rushing yards", "reception_yds": "Receiving yards", "receptions": "Receptions",
         "anytime_td": "Anytime TD"}


def fee(p: float) -> float:
    return RATE * p * (1 - p)


def taker_max(fair: float, bar: float = 0.03) -> float | None:
    """Highest ask where fair - ask - fee still leaves `bar`."""
    for i in range(990, 9, -1):
        p = i / 1000
        if p + fee(p) <= fair - bar:
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=Path("data/board"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--odds-asof", help="ISO time of the sportsbook snapshot")
    args = ap.parse_args()

    liq = json.loads((args.board / "prop_liquidity.json").read_text())
    stakes = [str(int(s)) for s in liq["stakes"]]
    by_mid: dict[str, list[dict]] = {}
    for r in liq["rows"]:
        by_mid.setdefault(r["market_id"], []).append(r)

    edges = [x for x in json.loads((args.board / "edges.json").read_text()) if x["kind"] == "player_prop"]
    out = []
    for x in edges:
        mid, idx = x["id"].rsplit("-", 1)
        rs = by_mid.get(mid)
        if not rs:
            continue
        r = rs[int(idx)]
        parts = x["selection"].split(" ")
        stat = next(p for p in parts if p in STATS)
        side = parts[-1] if x["selection"].endswith(("YES", "NO")) else parts[-2]
        line = None if stat == "anytime_td" else float(parts[-1])
        ask, f = r["ask"], x["fair"]
        if ask is None:
            continue
        pin_fair = None if x["pin_edge_pts"] is None else round(x["ask"] + x["fee"] + x["pin_edge_pts"] / 100, 4)
        out.append({
            "player": x["market"].split(":")[0], "stat": STATS[stat], "statkey": stat,
            "side": side.title(), "line": line,
            "ask": ask, "bid": r["bid"], "spread_c": r["spread_c"],
            "fair": f, "pin_fair": pin_fair, "n_books": x["n_books"],
            "edge_fresh": round((f - ask - fee(ask)) * 100, 2),
            "fills": {s: {"ok": r["fills"][s]["filled"], "avg": r["fills"][s]["avg"], "slip": r["fills"][s]["slip_c"]}
                      for s in stakes},
            "depth_best": r["depth_best_usd"], "depth_3c": r["depth_3c_usd"], "bid_depth": r["bid_depth_top3_usd"],
            "liquidity": r["liquidity"], "taker_max": taker_max(f),
            "maker_bid": round(math.floor((f - 0.02) * 100) / 100, 2),
            "dnp_rule": r["dnp_rule"],
        })
    out.sort(key=lambda r: (r["player"], r["statkey"], r["line"] or 0, r["side"]))
    args.out.write_text(json.dumps({"book_asof": liq["as_of"], "odds_asof": args.odds_asof,
                                    "stakes": stakes, "rows": out}, separators=(",", ":")))
    print(f"{len(out)} rows, stakes {stakes}")


if __name__ == "__main__":
    main()

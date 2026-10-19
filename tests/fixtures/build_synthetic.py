"""Builds tests/fixtures/synthetic/*.json.

SYNTHETIC DATA. Shapes follow the documented/observed Gamma, CLOB and Odds
API responses (including Gamma's JSON-encoded string fields); the numbers are
modelled on the 2026-10-19 WAS @ SF reference lines. Replace or supplement
with real captures from data/raw/ after the first live run.

    python tests/fixtures/build_synthetic.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "synthetic"
FEES = {"feesEnabled": True, "feeSchedule": {"rate": 0.05, "exponent": 1, "taker_only": True, "rebate_rate": 0.25}}
BASE = "nfl-was-sf-2026-10-20"


def market(mid, question, outcomes, tokens, bid, ask, last, liq, smt=None, line=None, **extra):
    m = {
        "id": str(mid),
        "conditionId": f"0xcond{mid}",
        "question": question,
        "slug": f"{BASE}-{mid}",
        "outcomes": json.dumps(outcomes),                       # JSON string inside JSON
        "outcomePrices": json.dumps([str(last), str(round(1 - last, 3))]),
        "clobTokenIds": json.dumps(tokens),
        "bestBid": bid,
        "bestAsk": ask,
        "lastTradePrice": last,
        "liquidity": str(liq),
        "liquidityNum": liq,
        "volume": str(liq * 4),
        "volumeNum": liq * 4,
        "gameStartTime": "2026-10-20 00:15:00+00",
        "endDate": "2026-10-27T00:15:00Z",
        "active": True,
        "closed": False,
        **FEES,
    }
    if smt:
        m["sportsMarketType"] = smt
    if line is not None:
        m["line"] = line
    m.update(extra)
    return m


def gamma_events():
    game = {
        "id": "9001", "slug": BASE, "title": "Commanders vs. 49ers",
        "startTime": "2026-10-20T00:15:00Z",
        "markets": [
            market(101, "Commanders vs. 49ers", ["Commanders", "49ers"], ["tok101a", "tok101b"],
                   0.20, 0.21, 0.205, 250000, "moneyline"),
            market(102, "Spread: 49ers (-7.5)", ["49ers", "Commanders"], ["tok102a", "tok102b"],
                   0.49, 0.50, 0.495, 80000, "spreads", -7.5),
            market(103, "Spread: 49ers (-6.5)", ["49ers", "Commanders"], ["tok103a", "tok103b"],
                   0.53, 0.54, 0.535, 20000, "spreads", -6.5),
            market(104, "Commanders vs. 49ers: O/U 52.5", ["Over", "Under"], ["tok104a", "tok104b"],
                   0.465, 0.475, 0.47, 60000, "totals", 52.5),
            market(105, "Commanders vs. 49ers: O/U 53.5", ["Over", "Under"], ["tok105a", "tok105b"],
                   0.43, 0.45, 0.44, 5000, "totals", 53.5),
            market(106, "Commanders vs. 49ers: O/U 49.5", ["Over", "Under"], ["tok106a", "tok106b"],
                   0.70, 0.72, 0.71, 9000, "totals", 49.5, closed=True),
            market(107, "Commanders vs. 49ers: O/U 56.5", ["Over", "Under"], ["tok107a", "tok107b"],
                   0.20, 0.30, 0.25, 40, "totals", 56.5),
            market(108, "1H Spread: 49ers (-3.5)", ["49ers", "Commanders"], ["tok108a", "tok108b"],
                   0.48, 0.50, 0.49, 3000, "first_half_spreads", -3.5),
        ],
    }
    props = {
        "id": "9002", "slug": f"{BASE}-player-props", "title": "Commanders vs. 49ers: Player Props",
        "markets": [
            market(201, "Brock Purdy: Passing Yards O/U 245.5", ["Over", "Under"], ["tok201a", "tok201b"],
                   0.47, 0.49, 0.48, 7000, "player_props"),
            market(202, "Christian McCaffrey: Rushing Yards O/U 75.5", ["Over", "Under"], ["tok202a", "tok202b"],
                   0.44, 0.46, 0.45, 6000, "player_props"),
            market(203, "Will Christian McCaffrey score a touchdown?", ["Yes", "No"], ["tok203a", "tok203b"],
                   0.62, 0.64, 0.63, 15000, "player_props"),
            market(204, "Will Jayden Daniels throw for 250+ passing yards?", ["Yes", "No"], ["tok204a", "tok204b"],
                   0.38, 0.40, 0.45, 4000, "player_props"),
        ],
    }
    quarter = {
        "id": "9003", "slug": f"{BASE}-highest-scoring-quarter", "title": "Highest scoring quarter",
        "markets": [
            market(301, "Will the 4th quarter be the highest scoring quarter?", ["Yes", "No"],
                   ["tok301a", "tok301b"], 0.30, 0.33, 0.31, 3000, feesEnabled=False, feeSchedule=None),
        ],
    }
    return game, props, quarter


def search():
    game, props, quarter = gamma_events()
    other = {"id": "9100", "slug": "nfl-was-dal-2026-11-26", "title": "Commanders vs. Cowboys"}
    return {"events": [{"id": e["id"], "slug": e["slug"], "title": e["title"]} for e in (game, props, quarter)]
            + [other], "tags": [], "profiles": []}


def books():
    # token -> CLOB /book. Prices and sizes are strings; arrays not best-first.
    def b(token, bids, asks):
        return {"market": "0xcond", "asset_id": token, "timestamp": "1792447200000", "hash": "0x0",
                "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
                "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
                "tick_size": "0.01", "min_order_size": "5", "neg_risk": False, "last_trade_price": "0.5"}
    return {
        "tok101a": b("tok101a", [(0.19, 5000), (0.20, 2000)], [(0.22, 3000), (0.21, 1500)]),
        "tok101b": b("tok101b", [(0.78, 4000), (0.79, 1500)], [(0.81, 5000), (0.80, 900)]),
        "tok104a": b("tok104a", [(0.465, 1200)], [(0.475, 800), (0.49, 2000)]),
        "tok104b": b("tok104b", [(0.525, 800)], [(0.535, 1200), (0.55, 3000)]),
        "tok204a": b("tok204a", [(0.38, 200)], [(0.47, 1000), (0.40, 300), (0.42, 500)]),
        "tok204b": b("tok204b", [(0.60, 300)], [(0.62, 200)]),
    }


def history():
    # Moneyline Commanders token: drifted from 0.25 to 0.21 over the last 7 hours.
    t_end = 1792447200  # 2026-10-19T22:00:00Z (synthetic)
    pts = []
    for i in range(85):
        t = t_end - (84 - i) * 300
        pts.append({"t": t, "p": round(0.25 - 0.04 * i / 84, 4)})
    return {"tok101a": {"history": pts}}


def odds_events():
    return [
        {"id": "evt_was_sf", "sport_key": "americanfootball_nfl", "sport_title": "NFL",
         "commence_time": "2026-10-20T00:15:00Z", "home_team": "San Francisco 49ers",
         "away_team": "Washington Commanders"},
        {"id": "evt_other", "sport_key": "americanfootball_nfl", "sport_title": "NFL",
         "commence_time": "2026-10-19T17:00:00Z", "home_team": "Dallas Cowboys",
         "away_team": "Washington Commanders"},
    ]


def _bm(key, markets):
    return {"key": key, "title": key.title(), "last_update": "2026-10-19T21:58:00Z",
            "markets": [{"key": k, "last_update": "2026-10-19T21:58:00Z", "outcomes": o} for k, o in markets]}


def _o(name, price, point=None, description=None):
    d = {"name": name, "price": price}
    if point is not None:
        d["point"] = point
    if description is not None:
        d["description"] = description
    return d


SF, WAS = "San Francisco 49ers", "Washington Commanders"


def odds_main():
    def core(ml, sp, tot, h1=None, props=()):
        mk = [("h2h", [_o(SF, ml[0]), _o(WAS, ml[1])]),
              ("spreads", [_o(SF, sp[0], -7.5), _o(WAS, sp[1], 7.5)]),
              ("totals", [_o("Over", tot[0], 52.5), _o("Under", tot[1], 52.5)])]
        if h1:
            mk.append(("spreads_h1", [_o(SF, h1[0], -3.5), _o(WAS, h1[1], 3.5)]))
        mk.extend(props)
        return mk
    purdy = lambda o, u: ("player_pass_yds", [_o("Over", o, 245.5, "Brock Purdy"), _o("Under", u, 245.5, "Brock Purdy"),
                                              _o("Over", -110, 249.5, "Jayden Daniels"), _o("Under", -110, 249.5, "Jayden Daniels")])
    cmc_rush = ("player_rush_yds", [_o("Over", -110, 72.5, "Christian McCaffrey"), _o("Under", -110, 72.5, "Christian McCaffrey")])
    return {
        "id": "evt_was_sf", "sport_key": "americanfootball_nfl", "commence_time": "2026-10-20T00:15:00Z",
        "home_team": SF, "away_team": WAS,
        "bookmakers": [
            _bm("pinnacle", core((-410, 360), (-104, -106), (-104, -112), (-105, -105))),
            _bm("draftkings", core((-425, 335), (-110, -110), (-105, -115), (-110, -110),
                                   [purdy(-112, -108), cmc_rush,
                                    ("player_anytime_td", [_o("Yes", -175, None, "Christian McCaffrey")])])),
            _bm("fanduel", core((-455, 350), (-110, -110), (-108, -112), None,
                                [purdy(-110, -110), cmc_rush,
                                 ("player_anytime_td", [_o("Yes", -180, None, "Christian McCaffrey")])])),
            _bm("betmgm", core((-400, 320), (-115, -105), (-110, -110))),
        ],
    }


def odds_alt():
    return {
        "id": "evt_was_sf", "sport_key": "americanfootball_nfl", "commence_time": "2026-10-20T00:15:00Z",
        "home_team": SF, "away_team": WAS,
        "bookmakers": [
            _bm("pinnacle", [("alternate_spreads", [_o(SF, -125, -6.5), _o(WAS, 105, 6.5),
                                                    _o(SF, 110, -8.5), _o(WAS, -130, 8.5)]),
                             ("alternate_totals", [_o("Over", -118, 51.5), _o("Under", 102, 51.5)])]),
            _bm("draftkings", [("alternate_spreads", [_o(SF, -130, -6.5), _o(WAS, 110, 6.5)]),
                               ("alternate_totals", [_o("Over", -135, 50.5), _o("Under", 110, 50.5),
                                                     _o("Over", 115, 54.5), _o("Under", -140, 54.5)]),
                               ("player_rush_yds_alternate", [_o("Over", -105, 74.5, "Christian McCaffrey")])]),
        ],
    }


def main():
    OUT.mkdir(exist_ok=True)
    game, props, quarter = gamma_events()
    files = {
        "gamma_event_base.json": [game],
        "gamma_event_props.json": [props],
        "gamma_event_quarter.json": [quarter],
        "gamma_search.json": search(),
        "clob_books.json": books(),
        "clob_history.json": history(),
        "odds_events.json": odds_events(),
        "odds_event_main.json": odds_main(),
        "odds_event_alt.json": odds_alt(),
    }
    for name, body in files.items():
        (OUT / name).write_text(json.dumps(body, indent=1) + "\n")
    print(f"wrote {len(files)} files to {OUT}")


if __name__ == "__main__":
    main()

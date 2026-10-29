import importlib.util
import pathlib

from pm_edges.pmus import Book

spec = importlib.util.spec_from_file_location(
    "late", pathlib.Path(__file__).resolve().parent.parent / "tools" / "pmus_late_favorite.py")
late = importlib.util.module_from_spec(spec)
spec.loader.exec_module(late)


def book(bid, offer):
    return Book("s", "MARKET_STATE_OPEN", [(bid, 50.0)], [(offer, 70.0)])


def test_up_favorite_buys_long_at_offer():
    assert late.pick(book(0.96, 0.97), 0.95, 0.99) == ("Up", 0.97, 70.0, 0.97, "ORDER_INTENT_BUY_LONG")


def test_down_favorite_is_short_priced_at_long_bid():
    # long bid 0.03 -> Down ask = 0.97; BUY_SHORT with price.value = the long bid (docs: price is always long-side)
    side, ask, depth, pv, intent = late.pick(book(0.03, 0.04), 0.95, 0.99)
    assert (side, ask, pv, intent) == ("Down", 0.97, 0.03, "ORDER_INTENT_BUY_SHORT") and depth == 50.0


def test_no_trade_outside_band():
    assert late.pick(book(0.55, 0.56), 0.95, 0.99) is None      # coin flip
    assert late.pick(book(0.99, 1.0), 0.95, 0.99) is None       # ask 1.00 never offered / above band
    assert late.pick(Book("s", "x", [], []), 0.95, 0.99) is None


def test_short_fill_price_is_converted_from_long_scale():
    # the exchange reports avgPx on the Up scale: a Down fill at 0.99 shows as 0.01
    src = pathlib.Path(late.__file__).read_text()
    assert "1 - px" in src and "intent.endswith(\"LONG\")" in src


class A:  # default thresholds
    max_spread, last_tol, drop_secs, max_drop, max_off_high, min_ask = 0.08, 0.03, 20.0, 0.03, 0.05, 0.95


def chk(side, bid, offer, last, hist=(), now=100.0):
    b = Book("s", "MARKET_STATE_OPEN", [(bid, 10.0)], [(offer, 10.0)])
    return late.safety_check(side, b, last, list(hist), now, A)


def test_replay_losing_eth_trade_is_blocked():
    # 18:09:25 ETH: long bid 0.15 / offer 0.95, last trade 0.01. Bought Up at 0.90, settled Down, lost $0.955.
    assert "spread" in chk("Up", 0.15, 0.95, 0.01)
    assert "disagrees" in chk("Up", 0.04, 0.10, 0.01)        # even in a tight book, last trade must agree


def test_replay_winning_trades_still_pass():
    assert chk("Down", 0.04, 0.09, 0.05) is None             # sol 18:05, bought Down, won
    assert chk("Down", 0.05, 0.07, 0.05) is None             # eth 17:50
    assert chk("Down", 0.01, 0.03, 0.01) is None             # btc 17:40


def test_falling_favorite_is_blocked():
    # Down favorite (long mid rising 0.04 -> 0.08 over 20s means Down mid fell 0.96 -> 0.92)
    assert "falling" in chk("Down", 0.07, 0.09, 0.05, hist=[(70.0, 0.04), (99.0, 0.08)], now=100.0)
    assert chk("Down", 0.07, 0.09, 0.05, hist=[(60.0, 0.01), (95.0, 0.08)], now=100.0) is not None
    # slow slide: no 20s drop >= 0.03, but 0.05 below the 60s high
    assert "60s high" in chk("Down", 0.07, 0.09, 0.05, hist=[(45.0, 0.01), (75.0, 0.05), (95.0, 0.075)], now=100.0)
    assert chk("Down", 0.04, 0.06, 0.05, hist=[(70.0, 0.05), (99.0, 0.05)], now=100.0) is None


def test_no_last_trade_fails_closed():
    assert "no last trade" in chk("Up", 0.95, 0.96, None)


def _bot(tmp_path, px, px_ok_at):
    import io
    import types
    args = types.SimpleNamespace(assets=["btc"], max_feed_age=8.0, feed_poll=3.0, fresh=True, live=False, stop_on_any_loss=False,
                                 t_min=15.0, t_max=100.0, min_ask=0.95, max_ask=0.99, usd=1.0, slip=0.01, min_edge=0.005,
                                 max_spread=0.08, last_tol=0.03, drop_secs=20.0, max_drop=0.03, max_off_high=0.05,
                                 max_trades=60, max_spend=100.0, max_attempts=3, risk="point")
    late.STATE = tmp_path / "state.json"
    b = late.Bot(args, None, None, io.StringIO())
    b.px = {"btc": px}
    b.px_ok_at = px_ok_at
    return b


def _prices(start, t_now, lead):
    """Dense 1-second prices: flat at 100 through the window's open, then `lead` above it, with a gentle wobble."""
    px = {}
    for t in range(start - 700, t_now):
        px[t] = 100.0 if t < start else 100.0 + lead + (0.05 if t % 2 else -0.05)
    return px


def test_consider_buys_when_book_and_index_price_agree_and_lead_is_big(tmp_path):
    import asyncio
    start = 1_800_000_000
    now = start + 300 - 60                                  # 60 s left
    px = _prices(start, int(now), lead=3.0)                 # 3.0 above the price to beat, calm series
    b = _bot(tmp_path, px, px_ok_at=now)
    slug = late.updown_slug(start, "btc")
    b.books[slug] = Book(slug, "MARKET_STATE_OPEN", [(0.01, 500.0)], [(0.03, 500.0)])   # Down 0.99 ... Up favorite? no: Up ask 0.03
    b.last[slug] = 0.97
    # book says Up is the favorite: long offer 0.97, bid 0.96
    b.books[slug] = Book(slug, "MARKET_STATE_OPEN", [(0.96, 500.0)], [(0.97, 500.0)])
    b.hist[slug] = [(now - 30, 0.965), (now - 1, 0.965)]
    asyncio.run(b.consider("btc", start, now))
    assert "btc" in b.pending and b.pending["btc"]["side"] == "Up"


def test_consider_refuses_when_index_price_says_the_other_side_leads(tmp_path):
    import asyncio
    start = 1_800_000_000
    now = start + 300 - 60
    px = _prices(start, int(now), lead=-3.0)                # Down is clearly ahead on the index
    b = _bot(tmp_path, px, px_ok_at=now)
    slug = late.updown_slug(start, "btc")
    b.books[slug] = Book(slug, "MARKET_STATE_OPEN", [(0.96, 500.0)], [(0.97, 500.0)])  # book thinks Up
    b.last[slug] = 0.97
    b.hist[slug] = [(now - 30, 0.965), (now - 1, 0.965)]
    asyncio.run(b.consider("btc", start, now))
    assert "btc" not in b.pending


def test_consider_fails_closed_without_a_fresh_index_price(tmp_path):
    import asyncio
    start = 1_800_000_000
    now = start + 300 - 60
    b = _bot(tmp_path, _prices(start, int(now), lead=3.0), px_ok_at=now - 60)   # feed 60 s stale
    slug = late.updown_slug(start, "btc")
    b.books[slug] = Book(slug, "MARKET_STATE_OPEN", [(0.96, 500.0)], [(0.97, 500.0)])
    b.last[slug] = 0.97
    b.hist[slug] = [(now - 30, 0.965), (now - 1, 0.965)]
    asyncio.run(b.consider("btc", start, now))
    assert "btc" not in b.pending


def test_limit_price_allows_up_to_the_edge_rule_not_just_one_cent(tmp_path):
    """The XRP miss: quote 0.95, offer repriced to 0.97 within 50 ms. The order's limit must reach 0.97."""
    import asyncio
    import types
    sent = {}

    class FakeTrader:
        def create(self, order):
            sent.update(order)
            return {"id": "X"}

        def order_outcome(self, oid):
            return {"state": "ORDER_STATE_EXPIRED", "cumQuantity": 0}

    start = 1_800_000_000
    now = start + 300 - 60
    b = _bot(tmp_path, _prices(start, int(now), lead=3.0), px_ok_at=now)
    b.a.live, b.trader = True, FakeTrader()
    slug = late.updown_slug(start, "btc")
    b.books[slug] = Book(slug, "MARKET_STATE_OPEN", [(0.94, 500.0)], [(0.95, 500.0)])
    b.last[slug] = 0.95
    b.hist[slug] = [(now - 30, 0.945), (now - 1, 0.945)]

    async def run():
        loop = asyncio.get_running_loop()
        b.run_in_executor = None
        await b.consider("btc", start, now)
    asyncio.run(run())
    assert float(sent["price"]["value"]) >= 0.97                       # reaches past the 0.96 that missed
    assert float(sent["price"]["value"]) <= 0.99
    assert slug not in b.done and b.attempts[slug] == 1                # expired empty: eligible to try again


def test_exchange_price_to_beat_replaces_the_reconstruction(tmp_path):
    start = 1_800_000_000
    now = start + 300 - 60
    b = _bot(tmp_path, _prices(start, int(now), lead=3.0), px_ok_at=now)
    recon = b.margin_view("btc", start, now)
    assert recon["exact"] is False
    b.ptb[("btc", start)] = 102.9              # the exchange says the price to beat was 102.9, nearly where price is
    exch = b.margin_view("btc", start, now)
    assert exch["exact"] is True and abs(exch["z"]) < abs(recon["z"])


def test_loss_budget_stops_everything_when_spent_and_does_not_halt_assets(tmp_path, monkeypatch):
    b = _bot(tmp_path, {}, 0.0)
    b.a.loss_budget = 0.15
    b.stats = {a: [0, 0, 0.0] for a in ("btc", "eth")}
    b.a.assets = ["btc", "eth"]
    monkeypatch.setattr(late, "settle_result", lambda http, ev: "Down")           # both of our Up trades lose
    def trade(slug):
        return {"slug": slug, "ev": "x", "end": 0, "side": "Up", "ask": 0.97, "qty": 0.10, "fee": 0.0, "next_check": 0}
    b.pending = {"btc": trade("cpc-btc-updown-5m-2026-10-29-1900z")}
    b.settle(1e12)                                                                  # loses 0.097
    assert b.budget_pnl == pytest.approx(-0.097) and not b.stop_all and not b.halted
    b.pending = {"eth": trade("cpceth-eth-updown-5m-2026-10-29-1900z")}
    b.settle(1e12)                                                                  # loses another 0.097 -> -0.194 <= -0.15
    assert b.stop_all and not b.halted                                              # budget replaced the per-asset halt


import pytest  # noqa: E402  (used by the budget test above)

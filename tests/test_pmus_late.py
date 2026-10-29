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

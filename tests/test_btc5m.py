import json

import pytest

from pm_edges.btc5m import asks_from_book, find_opportunity, parse_market, window_start

RATE = 0.07


def test_window_start():
    assert window_start(1793292828) == 1793292600


def test_asks_sorted_ascending_even_though_clob_sends_best_last():
    book = {"asks": [{"price": "0.99", "size": "10"}, {"price": "0.97", "size": "5"}, {"price": "0.98", "size": "7"}]}
    assert [p for p, _ in asks_from_book(book)] == [0.97, 0.98, 0.99]


def test_sum_just_under_one_is_not_profitable_after_fees():
    # 0.49 + 0.50 = 0.99 "< 100%", but fees are ~3.5c, so this must NOT trade.
    assert find_opportunity([(0.49, 100)], [(0.50, 100)], fee_rate=RATE) is None


def test_typical_book_sum_above_one_does_not_trade():
    assert find_opportunity([(0.50, 100)], [(0.51, 100)], fee_rate=RATE) is None


def test_real_gap_trades_with_exact_accounting():
    opp = find_opportunity([(0.45, 100)], [(0.50, 100)], fee_rate=RATE, max_shares=20)
    assert opp is not None and opp.shares == 20
    fee = 20 * RATE * (0.45 * 0.55 + 0.5 * 0.5)
    assert opp.fee == pytest.approx(fee)
    assert opp.profit == pytest.approx(20 - 20 * 0.95 - fee)
    assert opp.profit > 0


def test_walk_stops_when_next_level_stops_paying():
    up = [(0.45, 10), (0.52, 100)]
    down = [(0.50, 100)]
    opp = find_opportunity(up, down, fee_rate=RATE, max_shares=100)
    assert opp.shares == 10 and opp.up_limit == 0.45


def test_below_min_order_size_is_skipped():
    assert find_opportunity([(0.40, 3)], [(0.50, 100)], fee_rate=RATE, min_shares=5) is None


def test_zero_fee_market_trades_on_plain_sum():
    opp = find_opportunity([(0.49, 100)], [(0.50, 100)], fee_rate=0.0, max_shares=10)
    assert opp is not None and opp.profit == pytest.approx(0.1)


def test_parse_market_fee_schedule_and_fail_closed():
    raw = {"slug": "btc-updown-5m-1", "outcomes": '["Up", "Down"]', "clobTokenIds": '["u", "d"]',
           "feesEnabled": True, "feeSchedule": {"exponent": 1, "rate": 0.07}, "orderMinSize": 5,
           "orderPriceMinTickSize": 0.01, "acceptingOrders": True, "enableOrderBook": True}
    m = parse_market(raw, 1)
    assert (m.up_token, m.down_token, m.fee_rate, m.accepting) == ("u", "d", 0.07, True)
    del raw["feeSchedule"]
    assert parse_market(raw, 1).fee_rate == 0.07  # missing schedule -> pessimistic default, not 0
    raw["outcomes"] = '["Yes", "No"]'
    with pytest.raises(ValueError):
        parse_market(raw, 1)

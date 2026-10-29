import pytest

from pm_edges.margin import edge_after_risk, lead_z, reversal_ub, sigma_ref


def test_lead_z_sign_and_scale():
    # 90 s left: nothing known yet, the expected average is just the current price
    assert lead_z(100.0, 0.0, 101.0, 90, 0.5) == pytest.approx(2.0)      # Up ahead by 2 sigma
    assert lead_z(100.0, 0.0, 99.0, 90, 0.5) == pytest.approx(-2.0)      # Down ahead
    assert lead_z(100.0, 0.0, 100.0, 90, 0.5) == 0.0


def test_known_seconds_count_toward_the_average():
    # 30 s left: 30 prices already published (all 102), the rest expected at the current 100 -> mean 101
    assert lead_z(100.0, 30 * 102.0, 100.0, 30, 1.0) == pytest.approx(1.0)


def test_zero_volatility_is_never_a_lead():
    assert lead_z(100.0, 0.0, 105.0, 90, 0.0) == 0.0


def test_sigma_ref_needs_enough_data_and_measures_minute_moves():
    assert sigma_ref([100.0] * 100) == 0.0                                # too short
    ramp = [float(i) for i in range(700)]                                 # +60 every 60 s
    assert sigma_ref(ramp) == pytest.approx(60.0)


def test_reversal_bound_falls_with_lead_and_with_less_time():
    assert reversal_ub(90, 0.2) > reversal_ub(90, 1.2) > reversal_ub(90, 3.2)
    assert reversal_ub(120, 2.0) > reversal_ub(60, 2.0) > reversal_ub(30, 2.0)


def test_left_rounds_up_to_a_safer_bucket_only_in_the_cautious_direction():
    assert reversal_ub(34, 1.2) == reversal_ub(45, 1.2)                   # 34 s uses the 45 s bucket
    assert reversal_ub(121, 5.0) == 1.0                                   # no data beyond 120 s


def test_coin_flip_leads_are_unsafe_late_leads_are_safe():
    assert reversal_ub(30, 0.2) > 0.05
    assert reversal_ub(30, 1.2) < 0.10 < reversal_ub(30, 0.2)
    assert reversal_ub(30, 3.0) < 0.01                       # a big lead with little time left is safe


def test_exchange_price_to_beat_is_less_noisy_than_a_reconstruction():
    for left, z in ((90, 1.7), (60, 1.2), (45, 0.9)):
        assert reversal_ub(left, z, exact=True) < reversal_ub(left, z, exact=False)


def test_the_eth_loss_would_have_been_blocked():
    """ETH 19:45 UTC, bought Down at 0.95 with a 1.28 sigma lead and 59 s left; it settled Up (-$0.997).
    The first version of the table priced that at 2.7%; with the measured exchange-vs-candle noise it is 5.5% or worse,
    which is more than the 5% the 0.95 ask can afford."""
    for exact in (True, False):
        ub = reversal_ub(59, 1.28, exact=exact)
        assert ub >= 0.055
        assert edge_after_risk(0.95, 0.0695 * 0.95 * 0.05, ub) < 0


def test_edge_after_risk():
    # ask 0.97, fee ~0.2c, risk 0.5%: 1 - 0.005 - 0.97 - 0.002 = +0.023
    assert edge_after_risk(0.97, 0.002, 0.005) == pytest.approx(0.023)
    # the ETH loss: a 0.95 ask with a 0.73 lead and 34 s left: bucket 45, z in [0.5, 1)
    assert edge_after_risk(0.95, 0.0027, reversal_ub(34, 0.73)) < 0.03
    # the BTC loss: 0.95 ask, lead 0.03 with 88 s left
    assert edge_after_risk(0.95, 0.0027, reversal_ub(88, 0.03)) < 0
    # the SOL loss: 0.97 ask, lead 0.88 with 96 s left
    assert edge_after_risk(0.97, 0.002, reversal_ub(96, 0.88)) < 0


def test_max_price_is_the_highest_ask_that_still_clears_the_edge():
    from pm_edges.margin import max_price
    # the XRP case: reversal risk 0.7%, edge floor 0.5c -> paying 0.97 was still fine (and 0.96 missed the fill)
    assert max_price(0.007, 0.005) >= 0.97
    assert max_price(0.015, 0.005) >= 0.96            # the BTC case
    p = max_price(0.015, 0.005)
    assert edge_after_risk(p, 0.0695 * p * (1 - p), 0.015) >= 0.005
    assert edge_after_risk(p + 0.01, 0.0695 * (p + 0.01) * (0.99 - p), 0.015) < 0.005   # one more cent is too dear
    assert max_price(0.50, 0.005) is not None and max_price(0.50, 0.005) < 0.5           # coin flips only below fair value
    assert max_price(1.0, 0.0) is None                                                    # never buy a sure loser

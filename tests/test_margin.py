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
    assert reversal_ub(30, 1.2) < 0.01


def test_edge_after_risk():
    # ask 0.97, fee ~0.2c, risk 0.5%: 1 - 0.005 - 0.97 - 0.002 = +0.023
    assert edge_after_risk(0.97, 0.002, 0.005) == pytest.approx(0.023)
    # the ETH loss: a 0.95 ask with a 0.73 lead and 34 s left: bucket 45, z in [0.5, 1)
    assert edge_after_risk(0.95, 0.0027, reversal_ub(34, 0.73)) < 0.03
    # the BTC loss: 0.95 ask, lead 0.03 with 88 s left
    assert edge_after_risk(0.95, 0.0027, reversal_ub(88, 0.03)) < 0
    # the SOL loss: 0.97 ask, lead 0.88 with 96 s left
    assert edge_after_risk(0.97, 0.002, reversal_ub(96, 0.88)) < 0

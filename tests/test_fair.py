import pytest

from pm_edges.config import Config
from pm_edges.fair import (ScoreModel, american_to_decimal, american_to_prob, devig,
                           devig_multiplicative, devig_power, devig_shin, kelly_fraction,
                           model_for, prob_to_american, weighted_consensus, BookFair)


def test_american_to_prob():
    assert american_to_prob(-110) == pytest.approx(110 / 210)
    assert american_to_prob(+150) == pytest.approx(0.4)
    assert american_to_prob(+100) == pytest.approx(0.5)
    assert american_to_prob(-400) == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [0, 50, -99.9])
def test_american_rejects_invalid(bad):
    with pytest.raises(ValueError):
        american_to_prob(bad)


@pytest.mark.parametrize("odds", [-455, -200, -110, 100, 120, 335, 1200])
def test_american_round_trip(odds):
    assert prob_to_american(american_to_prob(odds)) == pytest.approx(odds if odds != 100 else -100)


def test_american_to_decimal():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_even_market_is_50_50(method):
    q = american_to_prob(-110)
    assert devig([q, q], method) == pytest.approx([0.5, 0.5])


def test_multiplicative_hand_case():
    # -105 / -115: q = .51220 / .53488, sum 1.04708 -> 48.92% / 51.08%
    p = devig_multiplicative([american_to_prob(-105), american_to_prob(-115)])
    assert p[0] == pytest.approx(0.48917, abs=1e-4)
    assert p[1] == pytest.approx(0.51083, abs=1e-4)


def test_power_hand_case():
    # Solve q1^k + q2^k = 1 for -105/-115: k ~ 1.0690 -> 48.84% / 51.16%
    p = devig_power([american_to_prob(-105), american_to_prob(-115)])
    assert sum(p) == pytest.approx(1)
    assert p[0] == pytest.approx(0.48840, abs=2e-4)


def test_shin_hand_case():
    p = devig_shin([american_to_prob(-105), american_to_prob(-115)])
    assert sum(p) == pytest.approx(1)
    assert p[0] == pytest.approx(0.48866, abs=2e-4)


def test_methods_differ_on_longshots_as_expected():
    # Moneyline SF -425 / WAS +335. Power and Shin take more margin off the
    # longshot than multiplicative does.
    q = [american_to_prob(-425), american_to_prob(335)]
    mult, power, shin = (devig(q, m)[1] for m in ("multiplicative", "power", "shin"))
    assert power < shin < mult
    assert 0.77 < 1 - mult < 0.80


def test_devig_needs_both_sides():
    with pytest.raises(ValueError):
        devig_power([0.6])


def test_unknown_method():
    with pytest.raises(ValueError):
        devig([0.5, 0.52], "magic")


def test_weighted_consensus():
    e = [BookFair("pinnacle", 0.50, 3), BookFair("dk", 0.54, 1)]
    assert weighted_consensus(e) == pytest.approx(0.51)
    assert weighted_consensus([]) is None


# ---- line model -----------------------------------------------------------

def test_model_identity_and_symmetry():
    m = ScoreModel(13.5, {3: 2.7, 7: 1.8}, "spread")
    assert m.convert(-7.5, 0.55, -7.5) == pytest.approx(0.55, abs=1e-6)
    # even game: laying 0.5 and taking 0.5 are complements
    assert m.cover_prob(0.0, -0.5) == pytest.approx(1 - m.cover_prob(0.0, 0.5))
    # a pick'em line at 50% means a mean margin of 0
    assert m.fit_mu(0, 0.5) == pytest.approx(0.0, abs=1e-3)


def test_model_monotone_in_line():
    m = model_for("spread", Config())
    ps = [m.convert(-7.5, 0.5, line) for line in (-10.5, -8.5, -7.5, -6.5, -3.5, -2.5)]
    assert ps == sorted(ps)


def test_key_numbers_make_crossing_7_and_3_expensive():
    flat = ScoreModel(13.5, {}, "spread")
    keyed = model_for("spread", Config())
    def step(model, a, b):
        return model.convert(-7.5, 0.5, b) - model.convert(-7.5, 0.5, a)
    # moving -7.5 -> -6.5 crosses 7; -5.5 -> -4.5 crosses nothing special
    assert step(keyed, -7.5, -6.5) > step(flat, -7.5, -6.5)
    assert step(keyed, -7.5, -6.5) > step(keyed, -5.5, -4.5)
    assert step(keyed, -3.5, -2.5) > step(keyed, -5.5, -4.5)


def test_integer_line_excludes_push():
    m = model_for("spread", Config())
    mu = m.fit_mu(-7.5, 0.5)
    p7, p75, p65 = m.cover_prob(mu, -7), m.cover_prob(mu, -7.5), m.cover_prob(mu, -6.5)
    assert p75 < p7 < p65


def test_total_conversion_direction():
    m = model_for("total", Config())
    over_at_53_5 = m.convert(52.5, 0.489, 53.5)
    assert 0.43 < over_at_53_5 < 0.47     # about 3-5 pts per point near 50
    assert m.convert(52.5, 0.489, 51.5) > 0.489


def test_kelly():
    assert kelly_fraction(0.5, 0.4) == pytest.approx(1 / 6)
    assert kelly_fraction(0.4, 0.5) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0


def test_binomial_tail_matches_hand_cases_and_large_n():
    from pm_edges.angles import binom_p_at_least
    assert binom_p_at_least(15, 15, 0.5) == pytest.approx(0.5 ** 15)
    assert binom_p_at_least(11, 13, 0.5) == pytest.approx((78 + 13 + 1) / 8192)   # C(13,11)+C(13,12)+C(13,13)
    assert binom_p_at_least(0, 10, 0.5) == 1.0
    assert 0.4 < binom_p_at_least(5000, 10000, 0.5) < 0.52                         # no overflow at large n

"""Regression tests on real responses captured 2026-10-19 23:06 UTC for WAS @ SF.

tests/fixtures/live_was_sf_2026-10-19 is a trimmed copy of a live
data/raw run: search results reduced to slugs, and CLOB books kept only for
the moneyline, McCaffrey receptions 4.5 and total 51.5 tokens (other
outcomes fall back to Gamma quotes offline).
"""
import shutil
from collections import Counter
from datetime import date, time
from pathlib import Path

import pytest

from pm_edges.config import Config
from pm_edges.pipeline import GameSpec, analyze, collect

LIVE = Path(__file__).parent / "fixtures" / "live_was_sf_2026-10-19"
SPEC = GameSpec("nfl", "WAS", "SF", date(2026, 10, 19), time(20, 15))


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(LIVE, data / "raw" / SPEC.game_id / "20261019T230625Z")
    cfg = Config(data_dir=data)
    run = collect(SPEC, cfg, offline=True, history=True)
    return cfg, run, analyze(run, None)


def edge(a, question, outcome):
    return next(e for e in a.edges if e.market.question == question and e.outcome.name == outcome)


def test_discovery(live):
    _, run, _ = live
    assert run.base_slug == "nfl-was-sf-2026-10-20"           # UTC next-day slug
    assert {m.event_slug for m in run.markets} == {"nfl-was-sf-2026-10-20", "nfl-was-sf-2026-10-20-player-props"}
    assert len(run.markets) == 494


def test_classification_of_real_markets(live):
    cfg, run, a = live
    by_smt = Counter()
    for m in a.checked:
        priced = all(o.selection is not None for o in m.outcomes)
        by_smt[(m.sports_market_type, priced)] += 1
    for smt in ("moneyline", "spreads", "totals", "team_totals", "first_half_spreads", "q3_totals",
                "passing_yards", "receptions", "anytime_touchdowns", "two_plus_touchdowns"):
        assert by_smt[(smt, False)] == 0, smt
    for smt in ("exact_margin", "safety"):
        assert by_smt[(smt, True)] == 0, smt
    # must not be read as a game total of 2.5
    tds = next(m for m in run.markets if m.question == "Commanders Total Touchdowns: O/U 2.5")
    assert tds.market_type == "other" and tds.outcomes[0].selection is None


def test_period_and_team_total_selections(live):
    _, run, _ = live
    q = {m.question: m for m in run.markets}
    assert q["1Q Spread: Commanders (-2.5)"].outcomes[0].selection.period == "q1"
    tt = q["Commanders 1H Team Total: O/U 6.5"].outcomes[0].selection
    assert (tt.period, tt.player, tt.stat, tt.line, tt.side) == ("h1", "WAS", "team_total", 6.5, "over")
    two = q["Christian McCaffrey: 2+ Touchdowns"].outcomes[0].selection
    assert (two.stat, two.line, two.side) == ("tds_over", 1.5, "over")


def test_hand_checked_prices(live):
    _, _, a = live
    # Pinnacle SF -340 / WAS +289 -> power no-vig 76.17%; CLOB best asks .24 / .77
    sf = edge(a, "Commanders vs. 49ers", "49ers")
    assert sf.fair.book_prob("pinnacle") == pytest.approx(0.7617, abs=1e-4)
    assert sf.ask == 0.77 and sf.outcome.price_source == "clob"
    assert edge(a, "Commanders vs. 49ers", "Commanders").ask == 0.24
    # Pinnacle McCaffrey receptions 4.5: -127 / +105 -> 53.67%
    rec = edge(a, "Christian McCaffrey: Receptions O/U 4.5", "Over")
    assert rec.fair.book_prob("pinnacle") == pytest.approx(0.5367, abs=1e-4)
    assert rec.fee_note.startswith("0.03 (market feeSchedule)")


def test_every_checked_market_has_fair_or_reason(live):
    _, _, a = live
    for e in a.edges:
        assert e.fair.prob is not None or e.fair.reason, e.market.question
    assert not any(e.fair.source == "model" and e.market.outcomes[0].selection.period != "full"
                   for e in a.edges)


def test_no_edge_at_capture_time(live):
    _, _, a = live
    assert not [e for e in a.edges if e.flagged]

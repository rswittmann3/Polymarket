"""Injury and goalie parsing, and the pieces of the Polymarket lag study that have a right answer."""
import importlib.util
from pathlib import Path

import pytest

from pm_edges import injuries as I

INJ = {"injuries": [{"displayName": "New York Knicks", "injuries": [
    {"status": "Out", "date": "2026-10-19T19:38Z", "shortComment": "Surgery.",
     "athlete": {"displayName": "Mitchell Robinson", "position": {"abbreviation": "C"},
                 "links": [{"href": "https://www.espn.com/nba/player/_/id/3908809/mitchell-robinson"}]}},
    {"status": "Day-To-Day", "date": "2026-10-20T10:00Z", "shortComment": "Ankle.",
     "athlete": {"displayName": "Jalen Brunson", "position": {"abbreviation": "PG"},
                 "links": [{"href": "https://www.espn.com/nba/player/_/id/3934672/jalen-brunson"}]}}]}]}

NBA_OVERVIEW = {"statistics": {"names": ["gamesPlayed", "avgMinutes", "avgPoints"],
                               "splits": [{"displayName": "Regular Season", "stats": ["77", "34.1", "26.0"]},
                                          {"displayName": "Career", "stats": ["400", "30.0", "20.0"]}]}}
NHL_OVERVIEW_EARLY = {"statistics": {"names": ["games", "goals", "assists", "points", "timeOnIcePerGame"],
                                     "splits": [{"displayName": "Regular Season", "stats": ["4", "0", "1", "1", "17:30"]},
                                                {"displayName": "Career", "stats": ["400", "150", "250", "400", "19:30"]}]}}


def test_report_reads_team_player_status_and_id():
    rows = I.report("nba", get=lambda url, params=None: INJ)
    assert [(r["team"], r["name"], r["pos"], r["status"], r["athlete_id"]) for r in rows] == [
        ("NYK", "Mitchell Robinson", "C", "Out", "3908809"), ("NYK", "Jalen Brunson", "PG", "Day-To-Day", "3934672")]


def test_player_value_uses_season_line_in_basketball_and_career_rate_early_in_a_hockey_season():
    v = I.player_value("nba", "1", get=lambda url, params=None: NBA_OVERVIEW, cache={})
    assert (v["games"], v["minutes"], v["points"]) == (77, 34.1, 26.0)
    h = I.player_value("nhl", "2", get=lambda url, params=None: NHL_OVERVIEW_EARLY, cache={})
    assert h["points"] == pytest.approx(1.0) and h["minutes"] == pytest.approx(19.5)             # 4 games is not a sample: career rate


def test_key_flag_and_ordering():
    assert I.is_key("nba", "Out", "PG", {"minutes": 34, "points": 26}) and not I.is_key("nba", "Out", "C", {"minutes": 12, "points": 4})
    assert I.is_key("nhl", "Out", "G", {}) and not I.is_key("nba", "Healthy", "PG", {"minutes": 34})
    rows = I.report("nba", get=lambda url, params=None: INJ)
    vals = {"3908809": {"minutes": 12, "points": 4}, "3934672": {"minutes": 34, "points": 26}}
    ctx = I.game_context("nba", "BOS", "NYK", rows, vals)
    assert [r["name"] for r in ctx] == ["Jalen Brunson", "Mitchell Robinson"] and ctx[0]["key"] and not ctx[1]["key"]
    assert I.game_context("nba", "BOS", "PHI", rows, vals) == []


def _lag():
    spec = importlib.util.spec_from_file_location("lag_study", Path(__file__).parent.parent / "tools" / "lag_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lag_study_price_lookup_and_book_fair_value():
    lag = _lag()
    hist = [{"t": 100, "p": 0.40}, {"t": 700, "p": 0.55}, {"t": 5000, "p": 0.60}]
    assert lag.pm_price_at(hist, 800) == 0.55 and lag.pm_price_at(hist, 50) is None
    assert lag.pm_price_at(hist, 2500) is None                         # last trade is older than the allowed age
    total = {"totals": [{"name": "Over", "price": -110, "point": 220.5}, {"name": "Under", "price": -110, "point": 220.5}]}
    assert lag.fair_at(total, "totals", "NYK") == (220.5, pytest.approx(0.5))
    spread = {"spreads": [{"name": "New York Knicks", "price": -120, "point": -4.5}, {"name": "Philadelphia 76ers", "price": 100, "point": 4.5}]}
    line, p = lag.fair_at(spread, "spreads", "NYK")
    assert line == -4.5 and 0.5 < p < 0.55
    assert lag.fair_at({}, "totals", "NYK") is None
    assert lag.team_is_home("Spread: Knicks (-4.5)", "NYK") and not lag.team_is_home("Spread: 76ers (+4.5)", "NYK")

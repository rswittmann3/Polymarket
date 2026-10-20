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


def test_gap_rule_buys_the_cheap_side_and_scores_it_against_the_close():
    import pandas as pd
    lag = _lag()
    rows = pd.DataFrame([
        {"game": "A@B", "kind": "totals", "ts": "2026-10-20T20:00:00+00:00", "market_id": "7", "book": 0.56, "hours_to_tip": 3.0},
        {"game": "A@B", "kind": "totals", "ts": "2026-10-20T20:30:00+00:00", "market_id": "7", "book": 0.58, "hours_to_tip": 2.5},
        {"game": "A@B", "kind": "totals", "ts": "2026-10-20T21:00:00+00:00", "market_id": "7", "book": 0.60, "hours_to_tip": 2.0}])
    books = pd.DataFrame([
        {"ts": "2026-10-20T20:01:00+00:00", "market_id": "7", "buy_avg": 0.52, "buy_filled": True, "sell_avg": 0.51, "sell_filled": True},
        {"ts": "2026-10-20T20:31:00+00:00", "market_id": "7", "buy_avg": 0.57, "buy_filled": True, "sell_avg": 0.56, "sell_filled": True}])
    t = lag.gap_rule(rows, books, threshold=0.02)
    assert len(t) == 1 and t.iloc[0]["side"] == "yes" and t.iloc[0]["edge_at_entry"] == pytest.approx(0.04)
    assert t.iloc[0]["clv"] == pytest.approx(0.60 - 0.52)                       # the book later agreed and then some
    assert lag.gap_rule(rows, books, threshold=0.05).empty                      # nothing clears a 5-point bar
    sell = lag.gap_rule(rows.assign(book=[0.40, 0.42, 0.38]), books.assign(sell_avg=[0.47, 0.48]), threshold=0.02)
    assert sell.iloc[0]["side"] == "no" and sell.iloc[0]["cost"] == pytest.approx(0.53)
    s = lag.summarize_trades(t)
    assert s["trades"] == 1 and s["share_beating_close"] == 1.0 and s["share_of_edge_that_held"] == pytest.approx(2.0)


def test_injury_event_study_orients_the_move_against_the_injured_team(tmp_path):
    import json
    lag = _lag()

    def pin(ts, spread, total):
        return {"ts": ts, "src": "pinnacle", "commence": "2026-10-21T23:00:00+00:00", "home": "NYK", "away": "PHI",
                "spreads": [{"name": "New York Knicks", "price": -110, "point": spread}, {"name": "Philadelphia 76ers", "price": -110, "point": -spread}],
                "totals": [{"name": "Over", "price": -110, "point": total}, {"name": "Under", "price": -110, "point": total}]}
    lines = [pin("2026-10-21T15:00:00+00:00", -4.5, 220.5),
             {"ts": "2026-10-21T15:20:00+00:00", "src": "injury", "baseline": True, "league": "nba", "team": "NYK", "athlete_id": "1", "status": "Out", "pos": "PG"},
             {"ts": "2026-10-21T15:40:00+00:00", "src": "injury", "baseline": False, "league": "nba", "team": "NYK", "athlete_id": "1", "status": "Out", "pos": "PG"},
             pin("2026-10-21T16:10:00+00:00", -2.5, 218.5)]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(l) for l in lines))
    out = lag.injury_event_study(path, windows=(30,), value_of=lambda aid, lg: {"minutes": 34, "points": 26})
    assert out["events"] == 1 and out["key_events"] == 1                     # the baseline row is not an event
    w = out["windows"]["30min"]
    assert w["spread"]["n"] == 1 and w["spread"]["mean_move"] == pytest.approx(2.0)      # Knicks at home: -4.5 to -2.5 hurts them by 2
    assert w["total"]["mean_move"] == pytest.approx(-2.0)
    none = lag.injury_event_study(path, windows=(30,), value_of=lambda aid, lg: {"minutes": 5, "points": 2})
    assert none["key_events"] == 0 and none["windows"]["30min"]["spread"]["n"] == 0

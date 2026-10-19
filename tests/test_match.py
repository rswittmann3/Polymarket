from datetime import datetime, timezone

import pytest

from conftest import load
from pm_edges.config import Config
from pm_edges.fair import american_to_prob, devig
from pm_edges.match import BookIndex, fair_for, match_markets
from pm_edges.models import BookRow, Selection
from pm_edges.polymarket import event_markets
from pm_edges.sportsbooks import (market_keys_for, missing_lines, needed_market_keys,
                                  normalize_event_odds, parse_market_key)

NOW = datetime(2026, 10, 19, 22, tzinfo=timezone.utc)
GAME = "nfl-was-sf-2026-10-19"


def rows():
    return (normalize_event_odds(load("odds_event_main.json"), GAME, NOW)
            + normalize_event_odds(load("odds_event_alt.json"), GAME, NOW))


def row(book, key, sel, price):
    return BookRow(GAME, book, key, sel, price, american_to_prob(price), NOW)


# ---- Odds API normalisation -------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("h2h", ("moneyline", "full", "")),
    ("spreads_h1", ("spread", "h1", "")),
    ("alternate_spreads", ("spread", "full", "")),
    ("alternate_totals_q1", ("total", "q1", "")),
    ("team_totals", ("total", "full", "team_total")),
    ("player_pass_yds", ("player_prop", "full", "pass_yds")),
    ("player_rush_yds_alternate", ("player_prop", "full", "rush_yds")),
    ("outrights", None),
])
def test_parse_market_key(key, expected):
    assert parse_market_key(key) == expected


def test_market_keys_for():
    assert market_keys_for(Selection("spread", "h1", line=-3.5, side="SF")) == ["spreads_h1"]
    assert market_keys_for(Selection("total", line=52.5, side="over"), alternates=True) == ["alternate_totals"]
    assert market_keys_for(Selection("player_prop", player="x", stat="anytime_td", side="yes"), True) == []
    assert market_keys_for(Selection("player_prop", player="x", stat="made_up_stat", side="over")) == []


def test_normalize_event_odds():
    rs = normalize_event_odds(load("odds_event_main.json"), GAME, NOW)
    pin_spread = [r for r in rs if r.book == "pinnacle" and r.market_key == "spreads"]
    assert {(r.selection.side, r.selection.line) for r in pin_spread} == {("SF", -7.5), ("WAS", 7.5)}
    purdy = [r for r in rs if r.selection.player == "brock purdy"]
    assert {r.selection.side for r in purdy} == {"over", "under"}
    td = [r for r in rs if r.selection.stat == "anytime_td"]
    assert td and all(r.selection.side == "yes" and r.selection.line is None for r in td)


def test_needed_and_missing_keys():
    ms = [m for e in ("gamma_event_base.json", "gamma_event_props.json")
          for m in event_markets(load(e), GAME, "SF", "WAS")]
    need = needed_market_keys(ms)
    assert {"h2h", "spreads", "totals", "spreads_h1", "player_pass_yds",
            "player_rush_yds", "player_anytime_td"} <= need
    main = normalize_event_odds(load("odds_event_main.json"), GAME, NOW)
    miss = missing_lines(ms, main)
    # -6.5 spread, 53.5/49.5/56.5 totals and McCaffrey 75.5 are not main lines
    assert {"alternate_spreads", "alternate_totals", "player_rush_yds_alternate"} <= miss
    assert "player_pass_yds_alternate" not in miss   # 245.5 and 249.5 are main lines


# ---- fair lookup ---------------------------------------------------------------

def test_exact_main_line_uses_all_books_weighted():
    cfg = Config()
    idx = BookIndex(rows(), "SF", "WAS")
    f = fair_for(Selection("total", line=52.5, side="under"), idx, cfg)
    assert f.source == "books"
    assert {b.book for b in f.books} == {"pinnacle", "draftkings", "fanduel", "betmgm"}
    dk = f.book_prob("draftkings")
    assert dk == pytest.approx(devig([american_to_prob(-115), american_to_prob(-105)], "power")[0])
    w = {"pinnacle": 3, "draftkings": 1, "fanduel": 1, "betmgm": 1}
    assert f.prob == pytest.approx(sum(b.prob * w[b.book] for b in f.books) / 6)


def test_alt_line_exact_match():
    f = fair_for(Selection("spread", line=-6.5, side="SF"), BookIndex(rows(), "SF", "WAS"), Config())
    assert f.source == "books"
    assert {b.market_key for b in f.books} == {"alternate_spreads"}


def test_opposite_side_of_spread():
    idx = BookIndex(rows(), "SF", "WAS")
    a = fair_for(Selection("spread", line=-7.5, side="SF"), idx, Config())
    b = fair_for(Selection("spread", line=7.5, side="WAS"), idx, Config())
    assert a.prob + b.prob == pytest.approx(1)


def test_line_mismatch_falls_back_to_model():
    f = fair_for(Selection("total", line=53.5, side="over"), BookIndex(rows(), "SF", "WAS"), Config())
    assert f.source == "model"
    assert 0.40 < f.prob < 0.48
    assert "converted" in f.reason


def test_model_not_used_for_props():
    f = fair_for(Selection("player_prop", player="christian mccaffrey", stat="rush_yds", line=75.5, side="over"),
                 BookIndex(rows(), "SF", "WAS"), Config())
    assert f.prob is None and "props" in f.reason


def test_one_sided_market_has_no_fair():
    f = fair_for(Selection("player_prop", player="christian mccaffrey", stat="anytime_td", side="yes"),
                 BookIndex(rows(), "SF", "WAS"), Config())
    assert f.prob is None and "one-sided" in f.reason


def test_one_sided_becomes_two_sided_when_another_book_has_no():
    rs = rows() + [row("caesars", "player_anytime_td",
                       Selection("player_prop", player="christian mccaffrey", stat="anytime_td", side="yes"), -170),
                   row("caesars", "player_anytime_td",
                       Selection("player_prop", player="christian mccaffrey", stat="anytime_td", side="no"), 135)]
    f = fair_for(Selection("player_prop", player="christian mccaffrey", stat="anytime_td", side="yes"),
                 BookIndex(rs, "SF", "WAS"), Config())
    assert f.source == "books" and f.books_label == "caesars"


def test_three_way_moneyline_is_skipped():
    sel = Selection("moneyline", "q1", side="SF")
    rs = [row("dk", "h2h_q1", Selection("moneyline", "q1", side="SF"), 120),
          row("dk", "h2h_q1", Selection("moneyline", "q1", side="WAS"), 250),
          row("dk", "h2h_q1", Selection("moneyline", "q1", side="draw"), 450)]
    f = fair_for(sel, BookIndex(rs, "SF", "WAS"), Config())
    assert f.prob is None


def test_unlisted_market():
    f = fair_for(Selection("player_prop", player="nobody", stat="pass_yds", line=1.5, side="over"),
                 BookIndex(rows(), "SF", "WAS"), Config())
    assert f.prob is None and f.reason == "no book lists this market"


def test_every_market_has_fair_or_reason():
    ms = [m for e in ("gamma_event_base.json", "gamma_event_props.json", "gamma_event_quarter.json")
          for m in event_markets(load(e), GAME, "SF", "WAS")]
    for of in match_markets(ms, rows(), "SF", "WAS", Config()):
        assert of.fair.prob is not None or of.fair.reason, of.market.question


def test_model_not_used_for_periods_or_team_totals():
    cfg = Config()
    rs = [row("dk", "spreads_q3", Selection("spread", "q3", line=-2.5, side="SF"), -110),
          row("dk", "spreads_q3", Selection("spread", "q3", line=2.5, side="WAS"), -110),
          row("dk", "team_totals", Selection("total", player="SF", stat="team_total", line=27.5, side="over"), -110),
          row("dk", "team_totals", Selection("total", player="SF", stat="team_total", line=27.5, side="under"), -110)]
    idx = BookIndex(rs, "SF", "WAS")
    q = fair_for(Selection("spread", "q3", line=-7.5, side="SF"), idx, cfg)
    assert q.prob is None and "full-game only" in q.reason
    t = fair_for(Selection("total", player="SF", stat="team_total", line=30.5, side="over"), idx, cfg)
    assert t.prob is None and "team totals" in t.reason

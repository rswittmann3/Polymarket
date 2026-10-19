from datetime import date, time

import pytest

from conftest import load
from pm_edges.config import Config
from pm_edges.models import Selection
from pm_edges.polymarket import (classify, effective_fee_rate, event_markets, fee_per_share,
                                 game_slug, is_stale, normalize_market, parse_book,
                                 parse_json_list, parse_period, parse_stat, slug_dates)

GAME = "nfl-was-sf-2026-10-19"


def _game_markets():
    return {m["id"]: m for m in load("gamma_event_base.json")[0]["markets"]}


# ---- JSON-string fields ---------------------------------------------------

def test_json_string_fields_are_decoded():
    raw = _game_markets()["101"]
    assert isinstance(raw["outcomes"], str)          # the fixture really is a string
    assert parse_json_list(raw["outcomes"]) == ["Commanders", "49ers"]
    assert parse_json_list(raw["clobTokenIds"]) == ["tok101a", "tok101b"]
    assert parse_json_list('["0.205", "0.795"]') == ["0.205", "0.795"]
    assert parse_json_list(["a", "b"]) == ["a", "b"]
    assert parse_json_list(None) == []
    assert parse_json_list("") == []


def test_normalize_market_fields():
    m = normalize_market(_game_markets()["101"], "nfl-was-sf-2026-10-20", GAME, "SF", "WAS")
    assert m.question == "Commanders vs. 49ers"
    assert m.market_type == "moneyline"
    assert [o.name for o in m.outcomes] == ["Commanders", "49ers"]
    assert [o.token_id for o in m.outcomes] == ["tok101a", "tok101b"]
    assert m.outcome_prices == [0.205, 0.795]
    assert m.liquidity == 250000 and m.volume == 1_000_000
    assert m.fees_enabled is True and m.fee_rate == 0.05


def test_executable_price_from_bid_ask():
    # bestBid .20 / bestAsk .21 for outcome 0 -> outcome 1 costs 1 - .20 = .80
    m = normalize_market(_game_markets()["101"], "", GAME, "SF", "WAS")
    was, sf = m.outcomes
    assert was.ask == pytest.approx(0.21)
    assert sf.ask == pytest.approx(0.80)
    assert sf.bid == pytest.approx(0.79)
    assert m.mid == pytest.approx(0.205)


def test_stale_flags():
    cfg = Config()
    raw = dict(_game_markets()["101"])
    assert is_stale(normalize_market(raw, "", GAME, "SF", "WAS"), cfg) == []
    raw.update(lastTradePrice=0.26)
    assert any("last" in r for r in is_stale(normalize_market(raw, "", GAME, "SF", "WAS"), cfg))
    raw.update(lastTradePrice=0.205, bestBid=0.15, bestAsk=0.25)
    assert any("wide" in r for r in is_stale(normalize_market(raw, "", GAME, "SF", "WAS"), cfg))


# ---- slugs ----------------------------------------------------------------

def test_monday_night_kickoff_uses_next_day_utc_slug():
    # Monday 2026-10-19 8:15 pm ET = Tuesday 00:15 UTC
    dates = slug_dates(date(2026, 10, 19), time(20, 15), "America/New_York")
    assert dates == [date(2026, 10, 20), date(2026, 10, 19)]
    assert game_slug("nfl", "WAS", "SF", dates[0]) == "nfl-was-sf-2026-10-20"


def test_afternoon_kickoff_keeps_local_date_first():
    assert slug_dates(date(2026, 10, 18), time(13, 0))[0] == date(2026, 10, 18)


def test_no_kickoff_tries_both():
    assert slug_dates(date(2026, 10, 19)) == [date(2026, 10, 19), date(2026, 10, 20)]


def test_west_coast_late_kickoff():
    # 5:15 pm PT is also 00:15 UTC next day
    assert slug_dates(date(2026, 10, 19), time(17, 15), "America/Los_Angeles")[0] == date(2026, 10, 20)


# ---- classification -------------------------------------------------------

def _m(question, outcomes, smt=None, line=None):
    d = {"question": question, "outcomes": str(outcomes).replace("'", '"')}
    if smt:
        d["sportsMarketType"] = smt
    if line is not None:
        d["line"] = line
    return d


def test_classify_spread():
    mt, sels, note = classify(_m("Spread: 49ers (-7.5)", ["49ers", "Commanders"], "spreads", -7.5), "SF", "WAS")
    assert mt == "spread"
    assert sels == [Selection("spread", line=-7.5, side="SF"), Selection("spread", line=7.5, side="WAS")]


def test_classify_spread_from_line_field_only():
    mt, sels, note = classify(_m("Commanders vs 49ers spread", ["49ers", "Commanders"], "spreads", -3.5), "SF", "WAS")
    assert sels[0] == Selection("spread", line=-3.5, side="SF")
    assert "assumed" in note


def test_classify_total_and_period():
    mt, sels, _ = classify(_m("Commanders vs. 49ers: 1H O/U 24.5", ["Over", "Under"], "totals"), "SF", "WAS")
    assert mt == "total"
    assert sels[0] == Selection("total", "h1", line=24.5, side="over")


def test_classify_team_total():
    mt, sels, _ = classify(_m("49ers Team Total O/U 30.5", ["Over", "Under"]), "SF", "WAS")
    assert sels[1] == Selection("total", player="SF", stat="team_total", line=30.5, side="under")


def test_classify_moneyline():
    mt, sels, _ = classify(_m("Commanders vs. 49ers", ["Commanders", "49ers"], "moneyline"), "SF", "WAS")
    assert mt == "moneyline" and [s.side for s in sels] == ["WAS", "SF"]


@pytest.mark.parametrize("question,outcomes,expected", [
    ("Brock Purdy: Passing Yards O/U 245.5", ["Over", "Under"],
     Selection("player_prop", player="brock purdy", stat="pass_yds", line=245.5, side="over")),
    ("Amon-Ra St. Brown: Receiving Yards O/U 80.5", ["Over", "Under"],
     Selection("player_prop", player="amon ra st brown", stat="reception_yds", line=80.5, side="over")),
    ("Will Jayden Daniels throw for 250+ passing yards?", ["Yes", "No"],
     Selection("player_prop", player="jayden daniels", stat="pass_yds", line=249.5, side="over")),
    ("Will Christian McCaffrey score a touchdown?", ["Yes", "No"],
     Selection("player_prop", player="christian mccaffrey", stat="anytime_td", side="yes")),
    ("Deebo Samuel Sr.: Anytime Touchdown", ["Yes", "No"],
     Selection("player_prop", player="deebo samuel", stat="anytime_td", side="yes")),
    ("Will Brian Robinson Jr. have over 60.5 rushing yards?", ["Yes", "No"],
     Selection("player_prop", player="brian robinson", stat="rush_yds", line=60.5, side="over")),
    ("Terry McLaurin: Rushing + Receiving Yards O/U 70.5", ["Over", "Under"],
     Selection("player_prop", player="terry mclaurin", stat="rush_reception_yds", line=70.5, side="over")),
    ("Will George Kittle score 2+ touchdowns?", ["Yes", "No"],
     Selection("player_prop", player="george kittle", stat="tds_over", line=1.5, side="over")),
])
def test_classify_props(question, outcomes, expected):
    mt, sels, _ = classify(_m(question, outcomes), "SF", "WAS")
    assert mt == "player_prop"
    assert sels[0] == expected
    assert sels[1].side in ("under", "no")


def test_prop_no_maps_to_under():
    _, sels, _ = classify(_m("Will Jayden Daniels throw for 250+ passing yards?", ["Yes", "No"]), "SF", "WAS")
    assert sels[1].side == "under" and sels[1].line == 249.5


def test_unrecognised_prop_has_reason():
    mt, sels, note = classify(_m("Who will score first?", ["Yes", "No"], "player_props"), "SF", "WAS")
    assert mt == "player_prop" and sels == [None, None] and note


def test_quarter_novelty():
    mt, sels, note = classify(_m("Will the 4th quarter be the highest scoring quarter?", ["Yes", "No"]), "SF", "WAS")
    assert mt == "quarter" and note == "no sportsbook equivalent"


def test_multi_outcome_is_other():
    mt, sels, note = classify(_m("Highest scoring quarter", ["Q1", "Q2", "Q3", "Q4"]), "SF", "WAS")
    assert mt == "other" and "4 outcomes" in note


def test_parse_period_and_stat():
    assert parse_period("first_half_spreads") == "h1"
    assert parse_period("1st Quarter Moneyline") == "q1"
    assert parse_period("Commanders vs 49ers") == "full"
    assert parse_stat("Rushing TDs") == "rush_tds"
    assert parse_stat("Passing Touchdowns") == "pass_tds"
    assert parse_stat("Receptions") == "receptions"


# ---- order book and fees --------------------------------------------------

def test_parse_book_sorts_unordered_levels():
    book = load("clob_books.json")["tok204a"]   # asks listed .47, .40, .42
    parsed = parse_book(book)
    assert parsed["ask"] == 0.40 and parsed["ask_size"] == 300
    assert parsed["asks"] == [(0.40, 300), (0.42, 500), (0.47, 1000)]
    assert parsed["bid"] == 0.38


def test_parse_book_empty():
    assert parse_book({"bids": [], "asks": []})["ask"] is None


def test_fee_per_share():
    assert fee_per_share(0.5, 0.05) == pytest.approx(0.0125)     # $1.25 per 100 shares
    assert fee_per_share(0.9, 0.05) == pytest.approx(0.0045)
    assert fee_per_share(0.5, 0.0) == 0.0


def test_effective_fee_rate_priority():
    cfg = Config()
    raw = dict(_game_markets()["101"])
    m = normalize_market(raw, "", GAME, "SF", "WAS")
    assert effective_fee_rate(m, cfg) == (0.05, "market feeSchedule")
    raw.update(feesEnabled=False, feeSchedule=None)
    assert effective_fee_rate(normalize_market(raw, "", GAME, "SF", "WAS"), cfg)[0] == 0.0
    raw.pop("feesEnabled")
    assert effective_fee_rate(normalize_market(raw, "", GAME, "SF", "WAS"), cfg)[1] == "default (unverified)"
    cfg.fee_rate_override = 0.0
    assert effective_fee_rate(m, cfg) == (0.0, "override")


def test_event_markets_dedupes():
    events = load("gamma_event_base.json") * 2
    ms = event_markets(events, GAME, "SF", "WAS")
    assert len(ms) == len(load("gamma_event_base.json")[0]["markets"])

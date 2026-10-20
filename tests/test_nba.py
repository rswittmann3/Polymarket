"""NBA support: team lookup, Polymarket market classification, Odds API keys, and a real-data regression."""
import shutil
from datetime import date, time
from pathlib import Path

import pytest

from pm_edges.config import Config
from pm_edges.fair import model_for
from pm_edges.models import Selection
from pm_edges.pipeline import GameSpec, analyze, collect
from pm_edges.polymarket import classify
from pm_edges.sportsbooks import market_keys_for, normalize_event_odds, parse_market_key
from pm_edges.teams import canonical_abbr, full_name, team_from_text

LIVE = Path(__file__).parent / "fixtures" / "live_nba_phi_nyk_2026-10-20"


def test_nba_team_lookup():
    assert canonical_abbr("PHO", "nba") == "PHX"
    assert canonical_abbr("sas", "nba") == "SAS"
    assert team_from_text("76ers", "nba") == "PHI"
    assert team_from_text("Philadelphia 76ers", "nba") == "PHI"
    assert team_from_text("Los Angeles Clippers", "nba") == "LAC"
    assert team_from_text("Los Angeles", "nba") is None                  # two teams share the city
    assert full_name("NYK", "nba") == "New York Knicks"
    # the same code means different teams in the two leagues
    assert full_name("WAS", "nfl") == "Washington Commanders" and full_name("WAS", "nba") == "Washington Wizards"
    with pytest.raises(KeyError):
        canonical_abbr("XYZ", "nba")


def _m(question, outcomes, smt, line=None):
    import json
    d = {"question": question, "outcomes": json.dumps(outcomes), "sportsMarketType": smt, "slug": "x"}
    if line is not None:
        d["line"] = line
    return d


def test_classify_nba_game_markets():
    mt, sels, _ = classify(_m("76ers vs. Knicks", ["76ers", "Knicks"], "moneyline"), "NYK", "PHI", "nba")
    assert mt == "moneyline" and [s.side for s in sels] == ["PHI", "NYK"]
    mt, sels, _ = classify(_m("Spread: Knicks (-5.5)", ["Knicks", "76ers"], "spreads", -5.5), "NYK", "PHI", "nba")
    assert sels == [Selection("spread", line=-5.5, side="NYK"), Selection("spread", line=5.5, side="PHI")]
    mt, sels, _ = classify(_m("76ers vs. Knicks: O/U 231.5", ["Over", "Under"], "totals", 231.5), "NYK", "PHI", "nba")
    assert mt == "total" and sels[0] == Selection("total", line=231.5, side="over")
    mt, sels, _ = classify(_m("76ers vs. Knicks: 1H O/U 118.5", ["Over", "Under"], "first_half_totals", 118.5), "NYK", "PHI", "nba")
    assert sels[0].period == "h1" and sels[0].line == 118.5
    mt, sels, _ = classify(_m("1H Spread: Knicks (-2.5)", ["Knicks", "76ers"], "first_half_spreads", -2.5), "NYK", "PHI", "nba")
    assert sels[0] == Selection("spread", "h1", line=-2.5, side="NYK")


def test_classify_nba_props_and_unpriceable():
    mt, sels, _ = classify(_m("LeBron James: Assists O/U 7.5", ["Yes", "No"], "assists", 7.5), "NYK", "PHI", "nba")
    assert mt == "player_prop"
    assert sels[0] == Selection("player_prop", player="lebron james", stat="assists", line=7.5, side="over")
    assert sels[1].side == "under"
    mt, sels, note = classify(_m("76ers vs. Knicks: Odd/Even Score", ["Odd", "Even"], "basketball_odd_even"), "NYK", "PHI", "nba")
    assert sels == [None, None] and "no sportsbook equivalent" in note


def test_odds_api_nba_keys_and_normalisation():
    assert parse_market_key("player_points_rebounds_assists") == ("player_prop", "full", "pra")
    assert parse_market_key("player_points") == ("player_prop", "full", "points")
    assert market_keys_for(Selection("player_prop", player="x", stat="pra", line=30.5, side="over")) == ["player_points_rebounds_assists"]
    assert market_keys_for(Selection("player_prop", player="x", stat="points", line=20.5, side="over"), alternates=True) == ["player_points_alternate"]
    body = {"bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "totals", "outcomes": [{"name": "Over", "price": -107, "point": 231.5}, {"name": "Under", "price": -105, "point": 231.5}]},
        {"key": "h2h", "outcomes": [{"name": "New York Knicks", "price": -200}, {"name": "Philadelphia 76ers", "price": 170}]},
        {"key": "player_points", "outcomes": [{"name": "Over", "description": "Jalen Brunson", "price": -115, "point": 26.5},
                                              {"name": "Under", "description": "Jalen Brunson", "price": -105, "point": 26.5}]}]}]}
    from datetime import datetime, timezone
    rows = normalize_event_odds(body, "g", datetime(2026, 10, 20, tzinfo=timezone.utc), "nba")
    sides = {(r.market_key, r.selection.side) for r in rows}
    assert ("h2h", "NYK") in sides and ("h2h", "PHI") in sides and ("totals", "over") in sides
    brunson = [r for r in rows if r.selection.player == "jalen brunson"]
    assert {r.selection.stat for r in brunson} == {"points"} and len(brunson) == 2


def test_nba_scoring_model_uses_league_parameters():
    cfg = Config().apply_league("nba")
    assert cfg.total_sigma == pytest.approx(18.2) and cfg.margin_sigma == pytest.approx(14.0) and not cfg.key_number_weights
    over_at_232_5 = model_for("total", cfg).convert(231.5, 0.50, 232.5)
    assert 0.46 < over_at_232_5 < 0.49                                   # about 2 points of probability per point of total
    assert Config().apply_league("nfl").total_sigma == pytest.approx(13.4)


# ---- real responses captured 2026-10-20 18:37 UTC, 76ers @ Knicks -------------------------------

@pytest.fixture(scope="module")
def live(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    spec = GameSpec("nba", "PHI", "NYK", date(2026, 10, 20), time(19, 0))
    shutil.copytree(LIVE, data / "raw" / spec.game_id / "20261020T183658Z")
    cfg = Config(data_dir=data, scope="halves")
    run = collect(spec, cfg, offline=True)
    return cfg, run, analyze(run, None)


def test_live_discovery_and_mapping(live):
    _, run, a = live
    assert run.base_slug == "nba-phi-nyk-2026-10-20" and len(run.markets) == 45
    types = {m.sports_market_type: m.market_type for m in run.markets}
    assert types["moneyline"] == "moneyline" and types["spreads"] == "spread" and types["totals"] == "total"
    assert types["assists"] == "player_prop" and types["first_half_totals"] == "total"
    assert types["basketball_odd_even"] == "other"
    assert all(m.outcomes[0].selection is not None for m in run.markets if m.sports_market_type in ("totals", "spreads", "points"))


def test_live_totals_match_the_books(live):
    _, _, a = live
    over = next(e for e in a.edges if e.market.question.endswith("O/U 231.5") and e.outcome.name == "Over")
    assert over.fair.source == "books" and 0.49 < over.fair.prob < 0.52
    assert over.fair.book_prob("pinnacle") is not None and over.fee == 0          # the market's own fee schedule says 0
    assert abs(over.edge) < 0.03 and not over.flagged
    assert not any(e.flagged for e in a.edges)


def test_live_every_priced_row_has_a_reason_or_value(live):
    _, _, a = live
    assert sum(e.fair.prob is not None for e in a.edges) > 60
    assert all(e.fair.prob is not None or e.fair.reason for e in a.edges)

"""History parsing and the angle engine, on small synthetic data with answers worked out by hand."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pm_edges import angles as A
from pm_edges import history as H

TYPES = {2008: {2: {"start": "2007-10-29T07:00Z", "end": "2008-04-17T07:00Z"},
                3: {"start": "2008-04-17T07:00Z", "end": "2008-06-23T07:00Z"}}}

SBR = """<table><tr><td>Date</td><td>Rot</td><td>VH</td><td>Team</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>Final</td><td>Open</td><td>Close</td><td>ML</td><td>2H</td></tr>
<tr><td>1030</td><td>501</td><td>V</td><td>Portland</td><td>26</td><td>23</td><td>28</td><td>20</td><td>97</td><td>184</td><td>189.5</td><td>900</td><td>95</td></tr>
<tr><td>1030</td><td>502</td><td>H</td><td>SanAntonio</td><td>29</td><td>30</td><td>22</td><td>25</td><td>106</td><td>12.5</td><td>13</td><td>-1400</td><td>5</td></tr>
<tr><td>1030</td><td>503</td><td>V</td><td>Seattle</td><td>28</td><td>34</td><td>24</td><td>31</td><td>117</td><td>2</td><td>1</td><td>100</td><td>105</td></tr>
<tr><td>1030</td><td>504</td><td>H</td><td>NewYork</td><td>30</td><td>21</td><td>21</td><td>24</td><td>96</td><td>214.5</td><td>212</td><td>-120</td><td>3</td></tr>
<tr><td>0415</td><td>505</td><td>V</td><td>NewJersey</td><td>20</td><td>20</td><td>20</td><td>20</td><td>80</td><td>pk</td><td>pk</td><td>100</td><td>3</td></tr>
<tr><td>0415</td><td>506</td><td>H</td><td>LALakers</td><td>20</td><td>20</td><td>20</td><td>20</td><td>90</td><td>190</td><td>191</td><td>-110</td><td>3</td></tr>
<tr><td>0420</td><td>507</td><td>V</td><td>Utah</td><td>20</td><td>20</td><td>20</td><td>20</td><td>100</td><td>3</td><td>4</td><td>150</td><td>3</td></tr>
<tr><td>0420</td><td>508</td><td>H</td><td>Denver</td><td>20</td><td>20</td><td>20</td><td>20</td><td>101</td><td>200</td><td>201</td><td>-170</td><td>3</td></tr>
</table>"""


def test_parse_sbr_pairs_lines_and_dates(tmp_path):
    f = tmp_path / "2007-08.htm"
    f.write_text(SBR)
    g = H.parse_sbr(f, 2007, TYPES)
    assert len(g) == 4
    a = g.iloc[0]       # Portland @ San Antonio: the home row has the small number, so San Antonio is the favourite
    assert (a.away, a.home, a.date) == ("POR", "SAS", date(2007, 10, 30))
    assert a.total_close == 189.5 and a.home_spread_close == -13.0 and a.total_open == 184 and a.home_spread_open == -12.5
    b = g.iloc[1]       # Seattle @ New York: the visitor row has the small number, so the visitor is favoured (home +1)
    assert (b.away, b.home) == ("OKC", "NYK") and b.total_close == 212 and b.home_spread_close == 1.0
    c = g.iloc[2]       # "pk" is a pick'em and the April date rolls into the next calendar year
    assert c.date == date(2008, 4, 15) and c.home_spread_close == 0.0 and c.total_close == 191.0 and c.away == "BKN"
    assert list(g["kind"]) == ["reg", "reg", "reg", "post"]
    assert g.iloc[3].home_spread_close == 4.0                      # Utah favoured away: home Denver +4


def _games():
    """Season 2025: a playoff meeting, then season 2026 openers. Lines are home-spread, total."""
    rows = [
        # (date, kind, away, home, away_pts, home_pts, home_spread, total)
        ("2026-05-04", "post", "PHI", "NYK", 98, 137, -7.5, 212.5),
        ("2026-05-06", "post", "PHI", "NYK", 102, 108, -10.5, 213.5),
        ("2026-05-08", "post", "NYK", "PHI", 108, 94, 3.5, 214.5),
        ("2026-10-20", "reg", "PHI", "NYK", 100, 110, -4.5, 231.5),       # NYK home opener: 210 points, Under
        ("2026-10-22", "reg", "NYK", "BOS", 100, 120, 6.0, 225.0),         # NYK game 2; BOS opener at home
        ("2026-10-21", "reg", "OKC", "SAS", 120, 118, 1.0, 230.0),         # SAS opener; 238 points, Over
    ]
    g = pd.DataFrame(rows, columns=["date", "kind", "away", "home", "away_pts", "home_pts", "home_spread_close", "total_close"])
    g["date"] = pd.to_datetime(g["date"])
    g["season"] = np.where(g["date"] < "2026-08-01", 2025, 2026)
    g["total_open"] = g["total_close"]
    g["home_spread_open"] = g["home_spread_close"]
    return g


def test_team_games_features():
    t = A.team_games(_games())
    nyk = t[(t.team == "NYK") & (t.kind == "reg")].sort_values("date")
    assert list(nyk.game_num) == [1, 2] and list(nyk.is_opener) == [1, 0]
    opener = nyk.iloc[0]
    assert opener.at_home == 1 and opener.spread == -4.5 and opener.total == 231.5
    assert opener.margin == 10 and opener.cover == 1 and opener.over == -1          # 110-100: covers -4.5, 210 < 231.5
    assert opener.prev_vs_opp_post == 1 and opener.prev_vs_opp_days == 165
    assert opener.prev_total == 214.5 and opener.total_vs_prev == pytest.approx(17.0)
    assert opener.prev_po_wins == 3 and opener.prev_po_games == 3                   # won all 3 playoff games last season
    away = t[(t.team == "PHI") & (t.game_num == 1)].iloc[0]
    assert away.at_home == 0 and away.spread == 4.5 and away.prev_po_wins == 0 and away.prev_po_games == 3
    assert nyk.iloc[1].rest_days == 2 and nyk.iloc[1].b2b == 0


def test_evaluate_counts_totals_once_per_game_and_roi():
    t = A.team_games(_games())
    r = A.evaluate(t, "game_num == 1", "under")        # all three openers' rows: PHI, NYK in one game, + BOS? no, + OKC/SAS
    assert (r.wins, r.losses, r.pushes) == (1, 1, 0) or r.n == 3       # one game counted once even though both teams match
    only_nyk = A.evaluate(t, "game_num == 1 and team == 'NYK'", "under")
    assert only_nyk.record() == "1-0" and only_nyk.roi == pytest.approx(100 / 110)
    sas = A.evaluate(t, "game_num == 1 and team == 'SAS'", "over")
    assert sas.record() == "1-0"
    ats = A.evaluate(t, "game_num == 1 and team == 'NYK'", "ats")
    assert ats.record() == "1-0"


def test_leave_one_out_and_active_check():
    t = A.team_games(_games())
    conds = ["at_home == 1", "game_num == 1", "prev_vs_opp_post == 1", "prev_po_wins > 1"]
    full = A.evaluate(t, " and ".join(conds), "under")
    assert full.record() == "1-0"
    lo = dict(A.leave_one_out(t, conds, "under"))
    assert set(lo) == set(conds) and all(r.n >= full.n for r in lo.values())
    # without the playoff-wins condition the NYK game is still the only home opener that followed a playoff meeting
    assert lo["prev_po_wins > 1"].n == 1
    # without the playoff-meeting condition, the other home openers need >1 playoff wins and have none
    assert lo["prev_vs_opp_post == 1"].n == 1
    g = _games()
    row = A.upcoming(g, "BOS", "NYK", "2026-10-30", -3.0, 228.0).query("team == 'NYK'").iloc[0]
    assert row.game_num == 3 and row.prev_vs_opp_post == 0 and row.prev_po_wins == 3
    assert not A.is_active(row, " and ".join(conds))
    both = A.upcoming(g, "PHI", "NYK", "2026-10-20", -4.5, 231.5)       # as of the morning of the opener
    opener = both.query("team == 'NYK'").iloc[0]
    assert opener.game_num == 1 and opener.prev_vs_opp_post == 1 and A.is_active(opener, " and ".join(conds))
    assert A.active_teams(both, " and ".join(conds)) == ["NYK"] and set(both.team) == {"NYK", "PHI"}


def test_binomial_tail_and_roi_math():
    assert A.binom_p_at_least(13, 13, 0.5) == pytest.approx(0.5 ** 13)
    assert A.PAYOUT == pytest.approx(0.90909, abs=1e-4)
    assert A.BREAKEVEN == pytest.approx(0.5238, abs=1e-4)


def test_null_search_returns_comparable_angles():
    t = A.team_games(_games())
    out = A.null_search(t, "under", conditions=2, trials=50, n_min=1, n_max=10, seed=3, base="season > 2000")
    assert out["trials"] == 50 and out["kept"] >= 0
    s = A.summarize_null([(11, 2, "q"), (6, 6, "q"), (13, 0, "q")], target_w=11, target_n=13)
    assert s["comparable"] == 3 and s["as_good"] == 2 and s["perfect"] == 1


# ---- hockey and football ---------------------------------------------------------------------------

NHL_TYPES = {2009: {2: {"start": "2008-10-04T07:00Z", "end": "2009-04-12T07:00Z"},
                    3: {"start": "2009-04-12T07:00Z", "end": "2009-06-20T07:00Z"}}}


def _nhl_html(wide: bool) -> str:
    """Two games (one neutral-site pair is skipped). 14 columns: ML open/close, total open, price, total close, price.
    16 columns: a puck line and its price sit before the totals."""
    def row(*cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    def line(date, rot, vh, team, final, mo, mc, to, tc, over_price, under_price, side):
        price = over_price if side == "V" else under_price
        puck = (1.5 if vh == "V" else -1.5, -200 if vh == "V" else 170)
        tail = [*puck, to, -110, tc, price] if wide else [to, -110, tc, price]
        return row(date, rot, vh, team, 0, 1, 1, final, mo, mc, *tail)
    head = row("Date", "Rot", "VH", "Team", "1st", "2nd", "3rd", "Final", "Open", "Close", *(["PuckLine"] if wide else []), "OpenOU", "CloseOU")
    rows = [line("1004", 1, "V", "NYRangers", 2, 100, -120, 5.5, 6, -125, None, "V"),
            line("1004", 2, "H", "TampaBay", 1, -120, 100, 5.5, 6, None, 105, "H"),
            line("1004", 3, "N", "SanJose", 3, 100, 100, 5.5, 5.5, -110, -110, "V"),
            line("1004", 4, "N", "Nashville", 1, -110, -110, 5.5, 5.5, -110, -110, "H"),
            line("0412", 5, "V", "Phoenix", 4, 150, 160, 5, 5, -115, None, "V"),
            line("0412", 6, "H", "Atlanta", 3, -170, -180, 5, 5, None, -105, "H")]
    return "<table>" + head + "".join(rows) + "</table>"


@pytest.mark.parametrize("wide", [False, True])
def test_parse_sbr_nhl_both_layouts(tmp_path, wide):
    f = tmp_path / "2008-09.htm"
    f.write_text(_nhl_html(wide))
    g = H.parse_sbr_nhl(f, 2008, NHL_TYPES)
    assert len(g) == 2                                          # the neutral-site pair is skipped
    a = g.iloc[0]
    assert (a.away, a.home, a.date, a.kind) == ("NYR", "TBL", date(2008, 10, 4), "reg")
    assert (a.away_ml, a.home_ml, a.away_ml_open, a.home_ml_open) == (-120, 100, 100, -120)
    assert (a.total_open, a.total_close, a.over_odds, a.under_odds) == (5.5, 6, -125, 105)
    b = g.iloc[1]                                               # Phoenix and Atlanta are today's Utah and Winnipeg
    assert (b.away, b.home, b.date, b.kind) == ("UTA", "WPG", date(2009, 4, 12), "post")


def test_nhl_bubble_playoffs_get_the_right_year(tmp_path):
    f = tmp_path / "2019-20.htm"
    f.write_text(_nhl_html(True).replace("1004", "0902"))
    types = {2020: {2: {"start": "2019-10-02T07:00Z"}, 3: {"start": "2020-08-01T07:00Z"}}}
    assert H.parse_sbr_nhl(f, 2019, types).iloc[0].date == date(2020, 9, 2)


def _espn_day(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": path.stem, "events": events}))


def _event(eid, day, away, home, a, h, detail, season=2009, typ=2):
    return {"id": eid, "date": day + "T00:00Z", "season": {"year": season, "type": typ}, "status": "STATUS_FINAL",
            "detail": detail, "neutral": False,
            "teams": [{"homeAway": "home", "abbr": home, "score": str(h)}, {"homeAway": "away", "abbr": away, "score": str(a)}]}


def test_build_nhl_removes_the_shootout_goal_and_labels_overtime(tmp_path, monkeypatch):
    root = tmp_path / "nhl"
    (root / "sbr").mkdir(parents=True)
    (root / "sbr" / "2008-09.htm").write_text(_nhl_html(False))
    (root / "season_types.json").write_text(json.dumps(NHL_TYPES))
    # NYR 2 @ TBL 1 went to a shootout: the box score has 2-1, the sportsbook total counts 2 goals; the second game is regulation
    _espn_day(root / "espn_scoreboard" / "20081005.json", [_event("1", "2008-10-05", "NYR", "TB", 2, 1, "Final/SO")])
    _espn_day(root / "espn_scoreboard" / "20090413.json", [_event("2", "2009-04-13", "ARI", "ATL", 4, 3, "Final", typ=3)])
    monkeypatch.setattr(H, "RAW", tmp_path)
    g = H.build_nhl()
    so = g[g.away == "NYR"].iloc[0]
    assert so.ot == "SO" and so.points == 2 and so.home_pts + so.away_pts == 3
    assert g[g.away == "UTA"].iloc[0].ot == "REG" and g[g.away == "UTA"].iloc[0].points == 7


def test_parse_espn_drops_bad_feed_values(tmp_path, monkeypatch):
    root = tmp_path / "nhl"
    _espn_day(root / "espn_scoreboard" / "20241105.json", [_event("9", "2024-11-05", "BOS", "TB", 3, 2, "Final/OT", season=2025)])
    (root / "espn_odds").mkdir(parents=True)
    bad = {"provider": "DraftKings", "overUnder": -135.0, "overOdds": -110, "underOdds": -110, "open_total": None, "close_total": None,
           "home_ml": 280, "away_ml": -145}                      # a total that is really a price, and a "moneyline" from the puck line
    good = {"provider": "Caesars Sportsbook", "overUnder": 6.0, "overOdds": -115, "underOdds": -105, "open_total": 5.5,
            "close_total": -135, "home_ml": -150, "away_ml": 130}
    (root / "espn_odds" / "9.json").write_text(json.dumps({"items": [bad, good]}))
    monkeypatch.setattr(H, "RAW", tmp_path)
    e = H.parse_espn({}, "nhl").iloc[0]
    assert e.provider == "Caesars Sportsbook" and e.total_close == 6.0 and e.total_open == 5.5
    assert (e.home_ml, e.away_ml, e.over_odds) == (-150, 130, -115) and e.ot == "OT"


def _nfl_csv(path):
    cols = ["season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "away_score", "home_team", "home_score",
            "location", "overtime", "spread_line", "total_line", "away_moneyline", "home_moneyline", "away_spread_odds",
            "home_spread_odds", "over_odds", "under_odds", "div_game", "roof", "temp", "wind", "away_rest", "home_rest",
            "away_qb_name", "home_qb_name"]
    rows = [
        [2015, "REG", 1, "2015-09-13", "Sunday", "13:00", "OAK", 13, "CIN", 33, "Home", 0, 9.0, 41.5, 400, -550, -110, -110, -110, -110, 0, "outdoors", 75, 5, 7, 7, "A", "B"],
        [2023, "SB", 22, "2024-02-11", "Sunday", "18:30", "SF", 22, "KC", 25, "Neutral", 1, -1.5, 47.5, -125, 105, -108, -112, -110, -110, 0, "dome", None, None, 14, 14, "C", "D"],
        [2026, "REG", 7, "2026-10-22", "Thursday", "20:15", "NE", None, "CHI", None, "Home", None, 3.0, 50.5, 124, -148, -118, -102, -115, -105, 0, "outdoors", None, None, 4, 4, "E", "F"],
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_build_nfl_orients_lines_to_the_home_team(tmp_path, monkeypatch):
    (tmp_path / "nfl").mkdir()
    _nfl_csv(tmp_path / "nfl" / "games.csv")
    monkeypatch.setattr(H, "RAW", tmp_path)
    g = H.build_nfl()
    assert len(g) == 2                                           # the unplayed game is not history
    a, b = g.iloc[0], g.iloc[1]
    assert (a.away, a.home, a.home_spread_close, a.margin, a.points) == ("LV", "CIN", -9.0, 20, 46)    # nflverse: positive = home favoured
    assert (b.kind, b.playoff_round, b.neutral, b.ot) == ("post", 4, 1, "OT") and b.home_spread_close == 1.5
    up = H.upcoming_nfl("NE", "CHI", date(2026, 10, 22))
    assert up["home_spread_close"] == -3.0 and up["total_close"] == 50.5 and up["hour"] == 20 and up["weekday"] == "Thursday"
    assert H.upcoming_nfl("NE", "CHI", date(2026, 11, 22)) is None


def _frame(lg: str, seasons=range(2010, 2016), per_season=60, seed=1):
    """Random games with the columns a league's history has, to exercise every library query."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for s in seasons:
        day = pd.Timestamp(f"{s}-09-15")
        for i in range(per_season):
            a, h = rng.choice(teams, 2, replace=False)
            day += pd.Timedelta(days=int(rng.integers(0, 3)))
            hp, ap = int(rng.integers(1, 7)), int(rng.integers(1, 7))
            rows.append({"date": day, "season": s, "kind": "post" if i > per_season - 8 else "reg", "away": a, "home": h,
                         "away_pts": ap, "home_pts": hp + (hp == ap), "points": ap + hp + (hp == ap),
                         "home_spread_close": float(rng.choice([-7, -3, -1, 1, 3, 7, 10])) if lg != "nhl" else np.nan,
                         "total_close": float(rng.choice([5.5, 6.0, 6.5])) if lg == "nhl" else float(rng.choice([38.5, 44, 50.5])),
                         "home_ml": float(rng.choice([-250, -150, 120, 180])), "away_ml": float(rng.choice([-250, -150, 120, 180])),
                         "div_game": int(rng.integers(0, 2)), "roof": rng.choice(["dome", "outdoors"]),
                         "temp": float(rng.integers(20, 80)), "wind": float(rng.integers(0, 25)),
                         "weekday": rng.choice(["Sunday", "Thursday", "Monday"]), "neutral": 0})
    g = pd.DataFrame(rows)
    g["margin"] = g["home_pts"] - g["away_pts"]
    return g


@pytest.mark.parametrize("lg", ["nba", "nhl", "nfl"])
def test_every_library_angle_runs_on_that_leagues_columns(lg):
    from pm_edges import trends
    t = A.team_games(_frame(lg))
    for name, q, kind in trends.LIBRARY[lg]:
        for bet in (("under", "over") if kind == "total" else (kind,)):
            r = A.evaluate(t, q, bet)
            assert r.n >= 0, name


def test_moneyline_roi_uses_posted_prices_and_totals_use_theirs():
    g = pd.DataFrame([
        {"date": "2026-01-01", "season": 2025, "kind": "reg", "away": "AAA", "home": "BBB", "away_pts": 2, "home_pts": 3, "points": 5,
         "home_spread_close": np.nan, "total_close": 5.5, "home_ml": 200, "away_ml": -240, "over_odds": -130, "under_odds": 110},
        {"date": "2026-01-03", "season": 2025, "kind": "reg", "away": "BBB", "home": "AAA", "away_pts": 1, "home_pts": 4, "points": 5,
         "home_spread_close": np.nan, "total_close": 4.5, "home_ml": -150, "away_ml": 130, "over_odds": 100, "under_odds": -120}])
    g["date"] = pd.to_datetime(g["date"])
    t = A.team_games(g)
    r = A.evaluate(t, "at_home == 1", "ml")                      # BBB at +200 wins, AAA at -150 wins
    assert r.record() == "2-0" and r.roi == pytest.approx((2.0 + 100 / 150) / 2)
    under = A.evaluate(t, "total <= 5.5", "under")               # 5 goals: under 5.5 wins at +110, over 4.5 loses; under 4.5 loses
    assert under.record() == "1-1" and under.roi == pytest.approx((1.10 - 1) / 2)
    assert A.evaluate(t, "at_home == 1", "ats").n == 0           # no spread in hockey: nothing to cover


def test_rest_edge_win_prob_and_weather_flags():
    g = _frame("nfl", seasons=[2020], per_season=30)
    t = A.team_games(g)
    row = t.iloc[40]
    twin = t[(t.game_id == row.game_id) & (t.team == row.opp)].iloc[0]
    assert row.rest_edge == pytest.approx(-twin.rest_edge) or np.isnan(row.rest_edge)
    assert row.opp_rest_days == twin.rest_days or np.isnan(row.opp_rest_days)
    both = t[t.game_id == row.game_id]
    assert both.win_prob.sum() == pytest.approx(1.0)
    assert set(t.loc[t.roof == "dome", "dome"]) == {1} and set(t.loc[t.roof == "outdoors", "dome"]) == {0}
    assert set(t.loc[t.weekday == "Thursday", "thu"]) == {1}


def test_benjamini_hochberg_adjustment():
    q = A.bh_adjust([0.001, 0.01, 0.03, 0.5])
    assert q == pytest.approx([0.004, 0.02, 0.04, 0.5])
    assert A.bh_adjust([0.04]) == [0.04]
    q = A.bh_adjust([0.2, 0.01, 0.2])
    assert q[1] == pytest.approx(0.03) and all(v >= p for v, p in zip(q, [0.2, 0.01, 0.2]))

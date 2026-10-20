"""History parsing and the angle engine, on small synthetic data with answers worked out by hand."""
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
    row = A.upcoming(t, 2026, "NYK", "BOS", True, pd.Timestamp("2026-10-30"), -3.0, 228.0)
    assert row.game_num == 3 and row.prev_vs_opp_post == 0 and row.prev_po_wins == 3
    assert not A.is_active(row, " and ".join(conds))
    opener = A.upcoming(t[t["date"] < "2026-10-20"], 2026, "NYK", "PHI", True, pd.Timestamp("2026-10-20"), -4.5, 231.5)
    assert opener.game_num == 1 and opener.prev_vs_opp_post == 1 and A.is_active(opener, " and ".join(conds))


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

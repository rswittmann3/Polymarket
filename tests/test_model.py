"""Walk-forward modelling: scoring math, leak guards, the slate parser."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pm_edges import angles as A
from pm_edges import model as M
from pm_edges import slate as S


def _oos(n=2000, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = signal * x + rng.normal(size=n)
    return pd.DataFrame({"season": np.repeat([2020, 2021], n // 2), "y": y, "p_m": x * signal})


def test_skill_is_positive_only_when_the_prediction_carries_signal():
    real = M.skill_regression(_oos(signal=0.5), "m")
    assert real.gain > 0 and real.t > 4 and all(v > 0 for v in real.by_season.values())
    noise = M.skill_regression(_oos(signal=0.0), "m")
    assert abs(noise.t) < 0.5                                   # predicting zero is exactly the baseline
    wrong = _oos(signal=0.5)
    wrong["p_m"] = -wrong["p_m"]
    assert M.skill_regression(wrong, "m").gain < 0


def test_simulate_settles_spread_total_and_move_questions():
    oos = pd.DataFrame({"season": 2020, "y": [3.0, -2.0, 1.0, -1.0], "p_m": [2.0, 1.5, -0.5, -3.0],
                        "over_odds": [100, -120, -110, 150], "under_odds": [-120, 100, -110, -170]})
    spread = M.simulate(oos, "side", "m", top_fracs=(1.0,))[0]          # +,+,-,- vs outcomes +,-,+,-: 2 wins 2 losses at -110
    assert (spread["wins"], spread["bets"]) == (2, 4)
    assert spread["roi"] == pytest.approx((2 * M.PAYOUT - 2) / 4)
    total = M.simulate(oos, "total", "m", top_fracs=(1.0,))[0]          # over@+100 wins, over@-120 loses, under@-110 loses, under@-170 wins
    assert total["wins"] == 2 and total["roi"] == pytest.approx((1.0 + 100 / 170 - 2) / 4)
    move = M.simulate(oos, "move_total", "m", top_fracs=(1.0,))[0]
    assert move["clv_pts"] == pytest.approx(np.mean(np.sign(oos["p_m"]) * oos["y"])) and move["toward"] == pytest.approx(0.5)


def test_status_labels_follow_the_evidence():
    def ev(t, up, n, recent, roi=-0.02, task="total"):
        return {"available": True, "models": {"avg": {"t": t, "seasons_positive": up, "seasons": n, "recent5_gain": recent,
                                                      "bets": [{"top": 0.1, "roi": roi}]}}}
    assert M.status("total", ev(0.5, 9, 18, 0.1))["label"] == "no skill beyond the line"
    assert "not enough to pay" in M.status("total", ev(7, 14, 15, 0.1))["label"]
    assert M.status("total", ev(7, 14, 15, 0.1, roi=0.03))["label"] == "beats the line and pays"
    assert M.status("move_side", ev(16, 13, 13, 0.2))["label"] == "predicts the line move"
    assert M.status("move_side", ev(16, 13, 13, -0.2))["label"] == "no reliable skill"      # a model that stopped working


def test_move_models_never_see_the_closing_line_or_the_move():
    banned = {"total_rel", "spread", "win_prob", "total_move", "spread_move", "total_vs_avg", "total_vs_prev"}
    assert not banned & set(M.OPEN_FEATURES)
    assert {"total_open_rel", "spread_open"} <= set(M.OPEN_FEATURES)
    assert "ats_resid" not in M.FEATURES and "tot_resid" not in M.FEATURES      # same-game outcomes


def test_move_outliers_from_bad_source_rows_are_dropped():
    f = pd.DataFrame({"season": [2020] * 4, "total_open": [220.0, 219.0, 18.0, 221.0], "total": [221.0, 219.0, 220.0, 440.0],
                      "total_move": [1.0, 0.0, 202.0, 219.0]})
    d, _ = M.task_data(f, "move_total", "nba")
    assert list(d["total_move"]) == [1.0, 0.0]


def _games(n=60):
    rng = np.random.default_rng(1)
    rows, day = [], pd.Timestamp("2020-10-01")
    for i in range(n):
        a, h = rng.choice(["AAA", "BBB", "CCC", "DDD"], 2, replace=False)
        day += pd.Timedelta(days=int(rng.integers(1, 3)))
        rows.append({"date": day, "season": 2020, "kind": "reg", "away": a, "home": h, "away_pts": int(rng.integers(90, 120)),
                     "home_pts": int(rng.integers(90, 120)), "home_spread_close": -3.0, "total_close": 215.0})
    return pd.DataFrame(rows)


def test_form_features_use_only_earlier_games():
    g = _games()
    base = A.team_games(g)
    changed = g.copy()
    changed.loc[30, ["away_pts", "home_pts"]] = [200, 50]                # rewrite one game's result
    alt = A.team_games(changed)
    # that game's own row is identical before and after the change...
    gid = base[(base["date"] == g.loc[30, "date"]) & (base["team"] == g.loc[30, "home"])].iloc[0]["game_id"]
    cols = ["ats_5", "tot_5", "marg_5", "pf_5", "winres_20"]
    b = base[base["game_id"] == gid].set_index("team")[cols]
    a = alt[alt["game_id"] == gid].set_index("team")[cols]
    pd.testing.assert_frame_equal(b, a)
    # ...and a later game involving the same team moves
    later = base[(base["team"] == g.loc[30, "home"]) & (base["date"] > g.loc[30, "date"])].iloc[0]
    later_alt = alt[(alt["team"] == later["team"]) & (alt["date"] == later["date"])].iloc[0]
    assert later["marg_5"] != later_alt["marg_5"]


def test_game_frame_is_home_side_with_relative_totals():
    f = M.game_frame(A.team_games(_games(200)))
    assert (f["at_home"] == 1).all() and (f["home"] == f["team"]).all() and (f["away"] == f["opp"]).all()
    assert "total_rel" in f and f["total_rel"].dropna().abs().max() < 1e-9            # constant 215 total: nothing "high"


DK_ITEM = {"provider": {"name": "DraftKings"}, "spread": -3.5, "overUnder": 235.5, "overOdds": -115.0, "underOdds": -105.0,
           "open": {"total": {"american": "232.5"}},
           "homeTeamOdds": {"favorite": True, "moneyLine": -142, "open": {"pointSpread": {"american": "-2.5"}},
                            "current": {"pointSpread": {"american": "-3.5"}}},
           "awayTeamOdds": {"moneyLine": 120}}


def test_parse_odds_orients_everything_to_the_home_side():
    o = S.parse_odds([{"provider": {"name": "Live"}, "overUnder": 1, "homeTeamOdds": {}}, DK_ITEM])
    assert (o["total"], o["total_open"], o["home_spread"], o["home_spread_open"]) == (235.5, 232.5, -3.5, -2.5)
    assert (o["home_ml"], o["away_ml"], o["over_odds"], o["provider"]) == (-142, 120, -115, "DraftKings")
    dog = {**DK_ITEM, "homeTeamOdds": {"favorite": False}}
    assert S.parse_odds([dog])["home_spread"] == 3.5                 # ESPN's sign is the favourite's; the home underdog gets +
    assert S.parse_odds([]) is None


def test_espn_slate_skips_games_already_played_and_unknown_teams():
    board = {"events": [
        {"id": "1", "date": "2026-10-21T23:30Z", "competitions": [{"status": {"type": {"name": "STATUS_SCHEDULED"}},
         "competitors": [{"homeAway": "home", "team": {"abbreviation": "ORL"}}, {"homeAway": "away", "team": {"abbreviation": "ATL"}}]}]},
        {"id": "2", "date": "2026-10-21T23:30Z", "competitions": [{"status": {"type": {"name": "STATUS_FINAL"}},
         "competitors": [{"homeAway": "home", "team": {"abbreviation": "ORL"}}, {"homeAway": "away", "team": {"abbreviation": "ATL"}}]}]}]}
    def fake(url, params=None):
        return {"items": [DK_ITEM]} if "odds" in url else board
    out = S.espn_slate("nba", date(2026, 10, 21), 1, get=fake)
    assert len(out) == 1 and (out[0]["away"], out[0]["home"], out[0]["date"], out[0]["total"]) == ("ATL", "ORL", "2026-10-21", 235.5)


def test_upcoming_many_builds_both_teams_for_each_game():
    g = _games()
    specs = [{"away": "AAA", "home": "BBB", "date": "2020-12-20", "total_close": 215.0, "home_spread_close": -2.0},
             {"away": "CCC", "home": "DDD", "date": "2020-12-20", "total_close": 220.0, "home_spread_close": 1.0}]
    t = A.upcoming_many(g, specs)
    assert len(t) == 4 and set(t["team"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert t["win"].isna().all() and t["game_num"].min() > 1               # unplayed, but they know their season so far

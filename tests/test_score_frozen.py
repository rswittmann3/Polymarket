import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _mod():
    spec = importlib.util.spec_from_file_location("score_frozen", Path(__file__).parent.parent / "tools" / "score_frozen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FROZEN = {"games": [
    {"away": "A", "home": "B", "total_open": 220.0, "total": 221.0, "pred_close_total": 222.0, "home_spread_open": -3.0, "home_spread": -3.5, "pred_close_spread": -4.0},
    {"away": "C", "home": "D", "total_open": 230.0, "total": 230.0, "pred_close_total": 228.0, "home_spread_open": 2.0, "home_spread": 2.0, "pred_close_spread": 1.0},
    {"away": "E", "home": "F", "total_open": 210.0, "total": 210.0, "pred_close_total": 211.0, "home_spread_open": 0.0, "home_spread": 0.0, "pred_close_spread": 0.5}]}
CLOSES = {("A", "B"): {"tip": "2026-10-20T23:00:00Z", "pin_total": 223.0, "pin_spread": -4.5},
          ("C", "D"): {"tip": "2026-10-20T23:00:00Z", "pin_total": 229.0, "pin_spread": 2.0},
          ("E", "F"): {"tip": "2026-10-22T23:00:00Z", "pin_total": 215.0, "pin_spread": 1.0}}       # has not tipped yet
NOW = datetime(2026, 10, 21, 6, 0, tzinfo=timezone.utc)


def test_scoring_counts_only_tipped_games_and_signs_clv_by_the_predicted_direction():
    r = _mod().score(FROZEN, CLOSES, now=NOW)
    t = r["total"]
    assert t["games"] == 2
    # game 1 predicted +2 from open and the line rose 3: +3 points; game 2 predicted -2 and it fell 1: +1 point
    assert t["from_open"]["mean_clv_points"] == pytest.approx(2.0) and t["from_open"]["direction_right"] == 2
    # from the line at prediction time: game 1 predicted +1, actual +2 (+2); game 2 predicted -2, actual -1 (+1)
    assert t["from_then"]["mean_clv_points"] == pytest.approx(1.5)
    s = r["spread"]
    assert s["from_open"]["mean_clv_points"] == pytest.approx((1.5 + 0.0) / 2) and s["games"] == 2   # game 2 predicted -1 but the line did not move


def test_nothing_to_score_before_any_game_tips():
    assert _mod().score(FROZEN, CLOSES, now=datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)) == {}

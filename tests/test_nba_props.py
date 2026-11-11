import numpy as np
import pandas as pd

from pm_edges import nba_props as NP
from tools import score_props as SP


def test_norm_name_strips_suffix_accents_punctuation():
    assert NP.norm_name("Kelly Oubre Jr.") == "kelly oubre"
    assert NP.norm_name("Nikola Jokić") == "nikola jokic"
    assert NP.norm_name("Trey Murphy III") == "trey murphy"


def test_p_over_nbinom_properties():
    p = NP.p_over(np.array([5.0, 10.0, 20.0]), 0.1, 9.5)
    assert p[0] < p[1] < p[2] and 0 < p[0] < p[2] < 1
    assert abs(NP.p_over(10.0, 0.05, 9.5) - 0.5) < 0.12                 # mean sits near a half-point line just below it
    assert NP.p_over(8.0, 0.3, 8.5) < NP.p_over(8.0, 0.02, 8.5) + 0.2    # heavier tails do not flip the sign of the edge


def test_fit_alpha_recovers_overdispersion():
    rng = np.random.default_rng(0)
    mu = rng.uniform(3, 25, 40000); alpha = 0.08
    y = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mu))
    assert abs(NP.fit_alpha(y, mu) - alpha) < 0.02


def _box(rows):
    cols = ["game_id", "date", "team", "player_id", "player", "starter", "dnp", "min", "pts", "reb", "ast", "tpm", "stl", "blk", "played"]
    d = pd.DataFrame(rows, columns=cols); d["date"] = pd.to_datetime(d.date); d["stype"] = 2; return d


def test_team_absences_flags_recent_rotation_player_only():
    rows = []
    for g in range(1, 7):                                       # six games: A and B play every game, C plays the first five and sits the sixth
        for pid, mins in (("A", 34), ("B", 28), ("C", 20)):
            if pid == "C" and g == 6:
                continue
            rows.append([f"g{g}", f"2026-11-0{g}", "XYZ", pid, pid, pid == "A", False, mins, 15, 4, 3, 1, 1, 0, True])
    ab = NP.team_absences(_box(rows)).set_index("game_id")
    assert ab.loc["g6", "miss_min"] == 20 and ab.loc["g6", "miss_n"] == 1
    assert ab.loc["g5", "miss_min"] == 0                          # C played game 5
    assert ab.loc["g2", "miss_min"] == 0                          # too little history to call anyone a rotation player yet


def test_ratio_model_extrapolates_beyond_training_range():
    rng = np.random.default_rng(1)
    n = 6000; base = rng.uniform(5, 28, n)
    d = pd.DataFrame({f"f_{s}_{h}": 1.0 for s in NP.STATS for h in (6, 20)}, index=range(n))
    d["f_pts_20"] = base; d["f_pts_6"] = base; d["f_min_20"] = 30.0; d["f_start_5"] = 1.0; d["f_rest"] = 2.0; d["f_min_sd"] = 4.0; d["f_games"] = 50
    for c in NP.ABS:
        d[c] = 0.0
    d["pts"] = rng.poisson(base)
    m = NP.fit_stat(d, "pts")
    hot = d.iloc[:5].copy(); hot["f_pts_20"] = 42.0; hot["f_pts_6"] = 42.0
    assert NP.predict_mean(m, hot, "pts").mean() > 36              # a 42-point scorer is projected near 42, not capped at the training maximum


def test_summarize_brier_and_bets():
    df = pd.DataFrame({"event": ["e1", "e1", "e2", "e2"], "player": list("abcd"), "stat": "pts", "line": 10.5,
                       "p_over": [0.8, 0.2, 0.7, 0.3], "mid_over": [0.5, 0.5, 0.5, 0.5], "over": [1.0, 0.0, 1.0, 0.0],
                       "edge_over": [0.25, -0.35, 0.15, -0.25], "edge_under": [-0.35, 0.25, -0.25, 0.15],
                       "ask_over": [0.55, 0.55, 0.55, 0.55], "ask_under": [0.55, 0.55, 0.55, 0.55], "close_mid": [0.6, 0.4, 0.55, 0.45]})
    r = SP.summarize(df, "t")
    assert r["brier_gain"] > 0 and r["bets_edge10"]["n"] == 4 and r["bets_edge10"]["win_rate"] == 1.0
    assert abs(r["bets_edge10"]["mean_gain_prob_pts"] - 0.45) < 1e-9
    assert r["close_moved_toward_model"]["mean"] > 0

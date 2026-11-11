import gzip, importlib.util, json
from pathlib import Path

spec = importlib.util.spec_from_file_location("ladder_scan", Path(__file__).resolve().parents[1] / "research/novig/01_ladder_arb_scan.py")
scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)


def _write(d, name, rows):
    with gzip.open(d / name, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mk(tmp_path, bids_a, bids_b):
    meta = [{"k": "ev", "e": "E1", "lg": "NBA", "d": "AAA @ BBB", "st": 9_000_000_000_000},
            {"k": "mk", "m": "A", "e": "E1", "d": "AAA @ BBB t220.5", "mt": "TOTAL", "s": "220.5", "n": ["Over 220.5", "Under 220.5"]},
            {"k": "mk", "m": "B", "e": "E1", "d": "AAA @ BBB t225.5", "mt": "TOTAL", "s": "225.5", "n": ["Over 225.5", "Under 225.5"]}]
    touch = [{"ts": 1_000_000 + i, "m": m, "e": "E1", "lg": "NBA", "mt": "TOTAL", "k": k, "st": 9_000_000_000_000, "b": b, "q": [1000, 1000], "o": ["x", "y"]}
             for i, (m, k, b) in enumerate((("A", "220.5", bids_a), ("B", "225.5", bids_b)))]
    _write(tmp_path, "meta_2026-01-01.jsonl.gz", meta); _write(tmp_path, "touch_2026-01-01.jsonl.gz", touch)
    return scan.main([str(tmp_path)], save=False, quiet=True)


def test_planted_crossed_ladder_is_found(tmp_path):
    # Over 220.5 ask = 1 - 0.60 = 0.40 ; Under 225.5 ask = 1 - 0.65 = 0.35 ; the two cover every total and cost 0.75
    F = _mk(tmp_path, ["0.30", "0.60"], ["0.65", "0.25"])
    assert len(F) == 1 and abs(F.edge.iloc[0] - 0.25) < 1e-9 and F.qty.iloc[0] == 1000


def test_normal_ladder_is_not_flagged(tmp_path):
    # tight, consistent books: Over 220.5 ask 0.62, Under 225.5 ask 0.58 -> cost 1.20
    F = _mk(tmp_path, ["0.38", "0.36"], ["0.42", "0.40"])
    assert (F.edge <= 0).all()


def test_team_totals_of_different_teams_are_not_one_ladder(tmp_path):
    meta = [{"k": "ev", "e": "E1", "lg": "NBA", "d": "A @ B", "st": 9_000_000_000_000},
            {"k": "mk", "m": "A", "e": "E1", "d": "Toronto Raptors 105.5 TEAM_TOTAL", "mt": "TEAM_TOTAL", "s": "105.5", "n": ["Over 105.5", "Under 105.5"]},
            {"k": "mk", "m": "B", "e": "E1", "d": "Boston Celtics 110.5 TEAM_TOTAL", "mt": "TEAM_TOTAL", "s": "110.5", "n": ["Over 110.5", "Under 110.5"]}]
    touch = [{"ts": 1_000_000 + i, "m": m, "e": "E1", "lg": "NBA", "mt": "TEAM_TOTAL", "k": k, "st": 9_000_000_000_000, "b": ["0.60", "0.65"], "q": [1000, 1000], "o": ["x", "y"]}
             for i, (m, k) in enumerate((("A", "105.5"), ("B", "110.5")))]
    _write(tmp_path, "meta_2026-01-01.jsonl.gz", meta); _write(tmp_path, "touch_2026-01-01.jsonl.gz", touch)
    F = scan.main([str(tmp_path)], save=False, quiet=True)
    assert F.empty or (F.edge <= 0).all()

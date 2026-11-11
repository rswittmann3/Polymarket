import gzip
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _mod():
    spec = importlib.util.spec_from_file_location("injury_lag", Path(__file__).parent.parent / "tools" / "injury_lag.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _w(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_fair_line_and_event_study_on_a_synthetic_game(tmp_path):
    m = _mod()
    t0 = int(datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc).timestamp() * 1000)
    tip = t0 + 5 * 3600_000
    meta = [{"k": "ev", "e": "E1", "lg": "NBA", "d": "MIA @ GSW", "st": tip},
            {"k": "mk", "m": "T", "e": "E1", "d": "MIA @ GSW t230.5", "mt": "TOTAL", "s": 230.5, "n": ["Over 230.5", "Under 230.5"]}]
    touch = []
    # spread markets for home GSW at lines -5.5, -4.5, -3.5: outcome 0 is the home side. Before the report GSW -4.5 is a coin flip
    # (fair line -4.5); 10 minutes after, the fair line is -2.5.
    for k, line in enumerate((-5.5, -4.5, -3.5, -2.5, -1.5)):
        meta.append({"k": "mk", "m": f"S{k}", "e": "E1", "d": f"GSW {line}", "mt": "SPREAD", "s": abs(line), "n": [f"GSW {line}", f"MIA +{abs(line)}"]})
        p_before = 0.5 + 0.03 * (line + 4.5)           # home cover prob rises with the line
        p_after = 0.5 + 0.03 * (line + 2.5)
        touch.append({"ts": t0 - 3600_000, "m": f"S{k}", "e": "E1", "lg": "NBA", "mt": "SPREAD", "k": abs(line), "st": tip, "b": [p_before - 0.01, 1 - p_before - 0.01]})
        touch.append({"ts": t0 + 600_000, "m": f"S{k}", "e": "E1", "lg": "NBA", "mt": "SPREAD", "k": abs(line), "st": tip, "b": [p_after - 0.01, 1 - p_after - 0.01]})
    _w(tmp_path / "meta_2026-11-05.jsonl.gz", meta)
    _w(tmp_path / "touch_2026-11-05.jsonl.gz", touch)
    df, ev, mk = m.load_novig(tmp_path)
    assert ev["E1"][:2] == ("MIA", "GSW")
    before, after = m.fair_line_series(df, mk, "E1", [t0 - 600_000, t0 + 900_000])
    assert before == pytest.approx(-4.5, abs=0.05) and after == pytest.approx(-2.5, abs=0.05)
    inj = [{"team": "MIA", "name": "X", "status": "Out", "date": "2026-11-05T18:00:00Z", "baseline": False},
           {"team": "MIA", "name": "Y", "status": "Out", "date": "2026-11-05T18:00:00Z", "baseline": True}]    # a baseline row is ignored
    res = m.event_study(df, ev, mk, inj, horizon_h=24, minutes=(5, 15), min_move=1.0)
    assert len(res) == 1 and res.iloc[0]["total_move"] == pytest.approx(2.0, abs=0.1)
    assert res.iloc[0]["done_5m"] == pytest.approx(0.0, abs=0.05)         # nothing had moved yet at +5 min
    assert res.iloc[0]["done_15m"] == pytest.approx(1.0, abs=0.05)        # fully moved by +15 min

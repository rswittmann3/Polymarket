import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent.parent / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _edge(flagged=False, edge=0.04, pin=0.04, books=4, why="", prob=0.55):
    fair = SimpleNamespace(prob=prob, books=["a"] * books, books_label="a, b", source="books")
    market = SimpleNamespace(market_id="m", question="Over 220.5", market_type="total", liquidity=5000.0)
    sel = SimpleNamespace(label=lambda: "OVER 220.5")
    return SimpleNamespace(flagged=flagged, edge=edge, pin_edge=pin, fair=fair, why_not=why, market=market, ask=0.5, fee=0.0,
                           size_at_ask=100.0, size_with_edge=400.0, signals=[],
                           outcome=SimpleNamespace(index=0, name="Over", selection=sel))


def test_classification_and_confidence():
    s = _load("scan_slate")
    assert s.classify(_edge(flagged=True)) == "flagged"
    assert s.classify(_edge(why="model-converted line (use --allow-model to flag)")) == "model"
    assert s.classify(_edge(why="Pinnacle does not confirm")) == "unconfirmed"
    assert s.classify(_edge(edge=0.02, why="edge below 3.0 pts")) == "below"
    assert s.classify(_edge(edge=-0.01)) == "none" and s.classify(_edge(prob=None, edge=None)) == "unpriced"
    assert s.confidence(_edge(flagged=True, pin=0.04, books=3)) == "high"
    assert s.confidence(_edge(flagged=True, pin=0.02, books=5)) == "medium"
    assert s.confidence(_edge(flagged=False)) == "low"


def test_edge_rows_keep_positive_gaps_and_cap_the_dollars():
    s = _load("scan_slate")
    cfg = SimpleNamespace(max_stake=50.0)
    game = {"league": "nba", "date": "2026-10-22", "away": "A", "home": "B", "kickoff": "2026-10-22T23:00:00+00:00"}
    analysis = SimpleNamespace(edges=[_edge(flagged=True), _edge(edge=-0.02), _edge(edge=0.005, why="edge below 3.0 pts")])
    rows = s.edge_rows(game, analysis, cfg)
    assert len(rows) == 1 and rows[0]["status"] == "flagged" and rows[0]["edge_pts"] == pytest.approx(4.0)
    assert rows[0]["dollars_with_edge"] == 50.0 and rows[0]["dollars_at_ask"] == pytest.approx(50.0) and rows[0]["game"] == "A @ B"


def test_export_summary_says_nothing_found_when_nothing_flagged(tmp_path):
    import json
    ex = _load("export_model_lab")
    (tmp_path / "scan").mkdir()
    row = {"id": "x", "league": "NBA", "game": "A @ B", "date": "d", "bet": "OVER", "market": "m", "ask": 0.5, "fair": 0.52, "edge_pts": 2.0,
           "pin_edge_pts": None, "books": 3, "dollars_with_edge": None, "dollars_at_ask": 40.0, "liquidity": 100, "status": "below", "confidence": "low",
           "why_not": "edge below 3.0 pts"}
    (tmp_path / "scan" / "nba.json").write_text(json.dumps({"scanned_at": "2026-10-22T18:13:37+00:00", "league": "nba", "min_edge_pts": 3.0, "max_stake": 50.0,
                                                           "games": [{"game": "A @ B", "markets_checked": 30, "outcomes_priced": 20, "flagged": 0}], "edges": [row]}))
    edges, summary, games = ex.scan_rows(tmp_path)
    assert summary[0]["flagged"] == 0 and summary[0]["markets"] == 30 and summary[0]["best_gap"] == 2.0 and summary[0]["best_flagged"] is None
    assert edges[0]["why"] == "Gap is under the minimum edge" and edges[0]["dollars"] == 40.0

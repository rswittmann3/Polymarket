"""End-to-end on synthetic fixtures: discovery -> books -> fair -> edges -> report."""
import csv

from rich.console import Console

from pm_edges import cli
from pm_edges.pipeline import analyze, collect
from pm_edges.report import print_edges, write_csv, write_html
from pm_edges.store import Store


def run_all(spec, cfg, fake_apis, **kw):
    run = collect(spec, cfg, transport=fake_apis.transport(), history=True, **kw)
    store = Store(cfg.db_path)
    try:
        return run, analyze(run, store)
    finally:
        store.close()


def render(analysis, cfg):
    console = Console(record=True, width=250)
    print_edges(console, analysis.edges, len(analysis.checked), cfg, analysis.signal_notes)
    return console.export_text()


def test_discovery_finds_all_sibling_events(spec, cfg, fake_apis):
    run = collect(spec, cfg, transport=fake_apis.transport(), books=False)
    assert run.base_slug == "nfl-was-sf-2026-10-20"
    events = {m.event_slug for m in run.markets}
    assert events == {"nfl-was-sf-2026-10-20", "nfl-was-sf-2026-10-20-player-props",
                      "nfl-was-sf-2026-10-20-highest-scoring-quarter"}   # last one only via search
    assert run.clob_priced == 6
    assert any("no CLOB book" in w for w in run.warnings)


def test_end_to_end_flags_designed_edge(spec, cfg, fake_apis, tmp_path):
    run, a = run_all(spec, cfg, fake_apis)
    assert run.quota_remaining is not None
    flagged = [e for e in a.edges if e.flagged]
    assert [(e.market.question, e.outcome.name) for e in flagged] == [
        ("Will Jayden Daniels throw for 250+ passing yards?", "Yes")]
    e = flagged[0]
    assert e.ask == 0.40 and e.fair.prob == 0.5
    assert abs(e.edge - (0.5 - 0.40 - 0.05 * 0.4 * 0.6)) < 1e-9
    assert e.size_at_ask == 300 and e.size_with_edge == 800    # .40 and .42 levels clear 3 pts

    # liquidity filter and closed markets
    questions = {m.question for m in a.checked}
    assert "Commanders vs. 49ers: O/U 56.5" not in questions   # liquidity 40
    assert "Commanders vs. 49ers: O/U 49.5" not in questions   # closed

    # reference check: Under 52.5 at 53.5c vs ~51% fair -> no edge
    under = next(x for x in a.edges if x.market.question.endswith("O/U 52.5") and x.outcome.name == "Under")
    assert under.ask == 0.535 and 0.50 < under.fair.prob < 0.52 and not under.flagged

    # model-converted line shows but cannot flag
    model = [x for x in a.edges if x.fair.source == "model"]
    assert model and all(not x.flagged for x in model)

    text = render(a, cfg)
    assert "11 markets checked, 1 flagged" in text
    assert "No edge found" not in text
    assert "No fair value" in text

    path = write_csv(a.edges, tmp_path / "r.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(a.edges)
    assert sum(r["flagged"] == "True" for r in rows) == 1
    html = write_html(a.edges, len(a.checked), tmp_path / "r.html", "WAS @ SF", cfg, run.fetched_at).read_text()
    assert "1 flagged" in html and "<table>" in html


def test_no_edge_when_fair_equals_price(spec, cfg, fake_apis):
    props = fake_apis.data["gamma_event_props.json"][0]["markets"]
    daniels = next(m for m in props if "Daniels" in m["question"])
    daniels.update(bestBid=0.49, bestAsk=0.50, lastTradePrice=0.50)
    fake_apis.data["clob_books.json"].pop("tok204a")
    fake_apis.data["clob_books.json"].pop("tok204b")
    cfg.fee_rate_override = 0.0
    _, a = run_all(spec, cfg, fake_apis)
    e = next(x for x in a.edges if "Daniels" in x.market.question and x.outcome.name == "Yes")
    assert e.fair.prob == 0.5 and e.ask == 0.5 and e.edge == 0.0 and not e.flagged
    assert not any(x.flagged for x in a.edges)
    assert "No edge found." in render(a, cfg)


def test_allow_model_lets_model_rows_flag(spec, cfg, fake_apis):
    game = fake_apis.data["gamma_event_base.json"][0]["markets"]
    m = next(m for m in game if m["question"].endswith("O/U 53.5"))
    m.update(bestBid=0.30, bestAsk=0.31, lastTradePrice=0.305)
    _, a = run_all(spec, cfg, fake_apis)
    row = next(x for x in a.edges if x.market.question.endswith("O/U 53.5") and x.outcome.name == "Over")
    assert row.fair.source == "model" and row.edge > cfg.min_edge and not row.flagged
    cfg.allow_model = True
    _, a = run_all(spec, cfg, fake_apis)
    row = next(x for x in a.edges if x.market.question.endswith("O/U 53.5") and x.outcome.name == "Over")
    assert row.flagged


def test_offline_rerun_matches_online(spec, cfg, fake_apis):
    _, online = run_all(spec, cfg, fake_apis)
    offline_run = collect(spec, cfg, offline=True, history=True)
    store = Store(cfg.db_path)
    offline = analyze(offline_run, store)
    store.close()
    key = lambda a: [(e.market.market_id, e.outcome.index, e.ask, e.fair.prob, e.flagged) for e in a.edges]
    assert key(offline) == key(online)


def test_cli_offline_and_snapshot(spec, cfg, fake_apis, capsys):
    run_all(spec, cfg, fake_apis)        # populate data/raw
    base = ["--away", "WAS", "--home", "SF", "--date", "2026-10-19", "--kickoff", "20:15",
            "--data-dir", str(cfg.data_dir), "--offline"]
    assert cli.main(["markets", *base]) == 0
    assert cli.main(["snapshot", *base]) == 0
    assert cli.main(["snapshot", *base]) == 0
    assert cli.main(["edges", *base, "--html", "--kelly", "--bankroll", "1000"]) == 0
    out = capsys.readouterr().out
    assert "markets checked" in out and "snapshot 2 saved" in out
    assert list((cfg.data_dir / "reports").glob("*.csv")) and list((cfg.data_dir / "reports").glob("*.html"))


def test_missing_api_key_is_reported_not_fatal(spec, cfg, fake_apis):
    cfg.odds_api_key = None
    run = collect(spec, cfg, transport=fake_apis.transport())
    assert run.markets and not run.book_rows
    assert any("ODDS_API_KEY" in w for w in run.warnings)


def test_no_event_found(spec, cfg, fake_apis):
    from datetime import date
    from pm_edges.httpclient import ApiError
    from pm_edges.pipeline import GameSpec
    import pytest
    with pytest.raises(ApiError, match="no Polymarket event"):
        collect(GameSpec("nfl", "DAL", "NYG", date(2026, 10, 19)), cfg, transport=fake_apis.transport())


def test_line_specific_signals_attach_to_their_line(spec, cfg, fake_apis):
    _, a = run_all(spec, cfg, fake_apis)
    def sigs(question, outcome):
        return next(x for x in a.edges if x.market.question == question and x.outcome.name == outcome).signals
    assert any(s.startswith("juice") for s in sigs("Spread: 49ers (-7.5)", "Commanders"))
    assert not any(s.startswith("juice") for s in sigs("Spread: 49ers (-6.5)", "49ers"))
    assert any(s.startswith("juice") for s in sigs("Commanders vs. 49ers: O/U 52.5", "Under"))
    assert not any(s.startswith("juice") for s in sigs("Commanders vs. 49ers: O/U 53.5", "Under"))
    assert any(s.startswith("PM Commanders") for s in sigs("Commanders vs. 49ers", "49ers"))


def test_offline_replays_saved_4xx(spec, cfg, fake_apis):
    run_all(spec, cfg, fake_apis)
    off = collect(spec, cfg, offline=True)
    w = next(w for w in off.warnings if "no CLOB book" in w)
    assert "HTTP 404" in w and "no saved response" not in w

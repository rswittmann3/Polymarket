"""The paper-trade ledger: entry, refresh before kickoff, settlement and the CLV arithmetic."""
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pm_edges import cli
from pm_edges.config import Config
from pm_edges.ledger import Ledger, resolution

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
KICK = "2026-10-20 23:00:00+00"


def _edge(ask=0.50, fair=0.55, fee=0.0, flagged=True, mid="m1", idx=0, size=None, pin=0.56):
    fair_obj = SimpleNamespace(prob=fair, source="books", book_prob=lambda book: pin if book == "pinnacle" else None)
    market = SimpleNamespace(market_id=mid, question="Over 220.5", market_type="total", game_start_time=KICK)
    return SimpleNamespace(market=market, outcome=SimpleNamespace(index=idx, name="Over"), fair=fair_obj, ask=ask, fee=fee,
                           edge=fair - ask - fee, flagged=flagged, size_with_edge=size)


def _ledger():
    return Ledger(sqlite3.connect(":memory:"))


def test_flagged_edge_is_recorded_once_with_a_capped_paper_stake():
    led, cfg = _ledger(), Config()
    assert led.record("g", [_edge(), _edge(mid="m2", flagged=False)], cfg, T0) == (1, 0)
    row = led.rows()[0]
    assert row["stake"] == pytest.approx(cfg.default_stake) and row["shares"] == pytest.approx(cfg.default_stake / 0.50)
    assert row["kickoff"] == "2026-10-20T23:00:00+00:00" and row["pin_entry"] == 0.56
    assert led.record("g", [_edge()], cfg, T0) == (0, 1)                      # a second scan refreshes instead of duplicating
    assert len(led.rows()) == 1


def test_thin_book_limits_the_shares_and_the_cap_binds_the_stake():
    led = _ledger()
    cfg = Config()
    cfg.default_stake, cfg.max_stake = 500.0, 50.0
    led.record("g", [_edge(size=20.0)], cfg, T0)
    assert led.rows()[0]["shares"] == pytest.approx(20.0)                     # only 20 shares clear the minimum edge
    led2 = _ledger()
    led2.record("g", [_edge()], cfg, T0)
    assert led2.rows()[0]["stake"] == pytest.approx(50.0)                     # never above max_stake


def test_refresh_stops_at_kickoff_so_the_last_pre_game_scan_is_the_close():
    led, cfg = _ledger(), Config()
    led.record("g", [_edge()], cfg, T0)
    led.record("g", [_edge(fair=0.58)], cfg, datetime(2026, 10, 20, 22, 50, tzinfo=timezone.utc))
    led.record("g", [_edge(fair=0.40)], cfg, datetime(2026, 10, 20, 23, 10, tzinfo=timezone.utc))      # after tip: in-play, ignored
    assert led.rows()[0]["fair_last"] == pytest.approx(0.58)


def test_settlement_computes_clv_and_pnl():
    led, cfg = _ledger(), Config()
    led.record("g", [_edge(ask=0.50, fee=0.01)], cfg, T0)
    led.record("g", [_edge(ask=0.50, fee=0.01, fair=0.58)], cfg, T0)
    led.settle_entry(1, 1.0, T0)
    r = led.rows()[0]
    assert r["clv"] == pytest.approx(0.08)                                    # closed 8 points better than the ask
    assert r["pnl"] == pytest.approx(r["shares"] * (1.0 - 0.50 - 0.01))
    s = led.summary()
    assert (s.entries, s.settled, s.hit_rate) == (1, 1, 1.0) and s.mean_clv == pytest.approx(0.08) and s.clv_t is None   # one bet: no se


def test_summary_t_stat_over_several_trades():
    led, cfg = _ledger(), Config()
    for i, close in enumerate([0.56, 0.52, 0.47, 0.58]):
        led.record("g", [_edge(mid=f"m{i}")], cfg, T0)
        led.record("g", [_edge(mid=f"m{i}", fair=close)], cfg, T0)
        led.settle_entry(i + 1, float(i % 2), T0)
    s = led.summary()
    assert s.mean_clv == pytest.approx((0.06 + 0.02 - 0.03 + 0.08) / 4) and s.clv_se > 0 and s.beat_close == 0.75


def test_resolution_reads_final_prices_only():
    done = {"closed": True, "outcomePrices": '["1", "0"]'}
    assert resolution(done, 0) == 1.0 and resolution(done, 1) == 0.0
    assert resolution({"closed": False, "outcomePrices": '["1", "0"]'}, 0) is None
    assert resolution({"closed": True, "outcomePrices": '["0.62", "0.38"]'}, 0) is None       # closed to trading, not yet resolved
    assert resolution({"closed": True, "outcomePrices": '["0.5", "0.5"]'}, 0) == 0.5            # a refund
    assert resolution(None, 0) is None


def test_settle_open_uses_the_lookup_and_skips_games_not_started():
    led, cfg = _ledger(), Config()
    led.record("g", [_edge()], cfg, T0)
    assert led.settle_open(T0, get=lambda url: {"closed": True, "outcomePrices": '["1","0"]'}) == 0              # before kickoff
    after = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
    assert led.settle_open(after, get=lambda url: {"closed": True, "outcomePrices": '["0","1"]'}) == 1
    assert led.rows()[0]["result"] == 0.0 and led.rows()[0]["status"] == "settled"


def test_ledger_command_on_an_empty_database(tmp_path, capsys):
    code = cli.main(["ledger", "--data-dir", str(tmp_path)])
    assert code == 0 and "ledger is empty" in capsys.readouterr().out


def test_manual_trade_is_scored_like_a_flagged_edge():
    led = _ledger()
    assert led.add_manual("nba-den-okc", "5769221", 1, "Spread: Thunder (-8.5)", "Nuggets", "spread", KICK, 0.52, 0.0, 5.0, 0.515, T0)
    assert not led.add_manual("nba-den-okc", "5769221", 1, "x", "y", "spread", KICK, 0.52, 0.0, 5.0, 0.515, T0)       # one entry per outcome
    r = led.rows()[0]
    assert r["shares"] == pytest.approx(5.0 / 0.52) and r["edge_entry"] == pytest.approx(-0.005) and r["kickoff"] == "2026-10-20T23:00:00+00:00"
    led.settle_entry(1, 1.0, T0)
    r = led.rows()[0]
    assert r["pnl"] == pytest.approx(r["shares"] * 0.48) and r["clv"] == pytest.approx(-0.005)

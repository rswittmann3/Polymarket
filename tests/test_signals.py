"""Each signal fires on the case built for it and stays silent otherwise."""
from datetime import datetime, timedelta, timezone

from pm_edges.config import Config
from pm_edges.fair import american_to_prob
from pm_edges.models import BookRow, Selection
from pm_edges.signals import compute_signals, pm_drift, reverse_line_moves
from pm_edges.store import Snapshot, Store

T0 = datetime(2026, 10, 19, 20, 0, tzinfo=timezone.utc)
G = "nfl-was-sf-2026-10-19"
SPREAD = ("spread", "full", "", "")
TOTAL = ("total", "full", "", "")
ML = ("moneyline", "full", "", "")
BOOKS = ("pinnacle", "draftkings", "fanduel", "betmgm")


def r(book, key, sel, price, t):
    return BookRow(G, book, key, sel, price, american_to_prob(price), t)


def snapshot(t, sf_line, total_prices, ml_prices, sid):
    rows = []
    for b in BOOKS:
        rows += [r(b, "spreads", Selection("spread", line=sf_line, side="SF"), -110, t),
                 r(b, "spreads", Selection("spread", line=-sf_line, side="WAS"), -110, t)]
        o, u = total_prices.get(b, (-110, -110))
        rows += [r(b, "totals", Selection("total", line=52.5, side="over"), o, t),
                 r(b, "totals", Selection("total", line=52.5, side="under"), u, t)]
        sf, was = ml_prices[b]
        rows += [r(b, "h2h", Selection("moneyline", side="SF"), sf, t),
                 r(b, "h2h", Selection("moneyline", side="WAS"), was, t)]
    return Snapshot(sid, t, rows, [])


ML_EVEN = {b: (-300, 250) for b in BOOKS}
# Pinnacle has SF ~80% no-vig, the soft books ~74%
ML_SPLIT = {"pinnacle": (-420, 380), "draftkings": (-300, 250), "fanduel": (-290, 240), "betmgm": (-300, 250)}


def kinds(sigs):
    return {(s.kind, s.group) for s in sigs}


def test_two_snapshots_fire_only_designed_signals():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=8), -7.5, {"draftkings": (-105, -115)}, ML_SPLIT, 2)
    sigs, notes = compute_signals([a, b], Config(), "SF", "WAS")
    got = kinds(sigs)
    assert ("line_move", SPREAD) in got
    assert ("steam", SPREAD) in got
    assert ("juice", TOTAL) in got
    assert ("divergence", ML) in got
    # silent elsewhere
    assert ("steam", TOTAL) not in got and ("steam", ML) not in got
    assert ("juice", SPREAD) not in got
    assert ("divergence", TOTAL) not in got and ("divergence", SPREAD) not in got
    assert ("line_move", TOTAL) not in got
    steam = next(s for s in sigs if s.kind == "steam")
    assert "toward" in steam.text and len(steam.books) == 4
    juice = next(s for s in sigs if s.kind == "juice")
    assert "under" in juice.text.lower() and juice.books == ["draftkings"]
    move = next(s for s in sigs if s.kind == "line_move" and s.group == SPREAD)
    assert "-5.5 → -7.5" in move.text


def test_identical_snapshots_are_silent():
    a = snapshot(T0, -7.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=8), -7.5, {}, ML_EVEN, 2)
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    assert sigs == []


def test_steam_needs_three_books():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=8), -5.5, {}, ML_EVEN, 2)
    for row in b.book_rows:      # only two books move the spread
        if row.book in ("pinnacle", "draftkings") and row.market_key == "spreads":
            line = -7.5 if row.selection.side == "SF" else 7.5
            object.__setattr__(row, "selection", Selection("spread", line=line, side=row.selection.side))
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    assert ("steam", SPREAD) not in kinds(sigs)
    assert ("line_move", SPREAD) in kinds(sigs)


def test_steam_outside_window_is_not_reported():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=25), -7.5, {}, ML_EVEN, 2)
    sigs, notes = compute_signals([a, b], Config(), "SF", "WAS")
    assert ("steam", SPREAD) not in kinds(sigs)
    assert any("steam not checked" in n for n in notes)


def test_single_snapshot_note():
    sigs, notes = compute_signals([snapshot(T0, -7.5, {}, ML_EVEN, 1)], Config(), "SF", "WAS")
    assert not any(s.kind in ("line_move", "steam") for s in sigs)
    assert any("one snapshot" in n for n in notes)


def test_pm_drift_toward_and_away():
    t = 1_800_000_000
    falling = [(t - 7 * 3600 + i * 300, 0.25 - 0.04 * i / 84) for i in range(85)]
    toward = pm_drift(falling, 0.20, ("pm", "1"), "Commanders")
    assert toward and "toward fair" in toward[0].text
    away = pm_drift(falling, 0.30, ("pm", "1"), "Commanders")
    assert "away from fair" in away[0].text
    flat = [(t - 7 * 3600 + i * 300, 0.25) for i in range(85)]
    assert pm_drift(flat, 0.20, ("pm", "1"), "Commanders") == []
    assert pm_drift([], 0.2, ("pm", "1"), "x") == []


def test_reverse_line_move_hook():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=30), -5.5, {}, {b: (-250, 210) for b in BOOKS}, 2)
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    move = [s for s in sigs if s.kind == "line_move" and s.group == ML]
    assert move and move[0].value < 0        # SF got cheaper
    rlm = reverse_line_moves(move, {ML: {"side": "SF", "bets_pct": 0.75}}, "SF")
    assert rlm and "75%" in rlm[0].text
    assert reverse_line_moves(move, {ML: {"side": "WAS", "bets_pct": 0.75}}, "SF") == []


def test_store_round_trip(tmp_path):
    store = Store(tmp_path / "s.db")
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    store.write(G, a.taken_at, a.book_rows, [])
    b = snapshot(T0 + timedelta(minutes=8), -7.5, {}, ML_EVEN, 2)
    store.write(G, b.taken_at, b.book_rows, [])
    snaps = store.snapshots(G)
    assert [s.taken_at for s in snaps] == [a.taken_at, b.taken_at]
    assert {(x.book, x.selection, x.price_american) for x in snaps[1].book_rows} == \
        {(x.book, x.selection, x.price_american) for x in b.book_rows}
    sigs, _ = compute_signals(snaps, Config(), "SF", "WAS")
    assert ("steam", SPREAD) in kinds(sigs)
    store.close()


def test_line_move_ignores_change_in_which_books_are_quoted():
    a = snapshot(T0, -7.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=30), -7.5, {}, ML_EVEN, 2)
    # a single book is added with a very different line, and another is dropped
    a.book_rows[:] = [x for x in a.book_rows if x.book != "betmgm"]
    for row in b.book_rows:
        if row.book == "betmgm" and row.market_key == "spreads":
            line = -3.5 if row.selection.side == "SF" else 3.5
            object.__setattr__(row, "selection", Selection("spread", line=line, side=row.selection.side))
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    assert not [s for s in sigs if s.kind == "line_move"]


def test_line_move_needs_two_books_and_reports_median():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=30), -5.5, {}, ML_EVEN, 2)
    def move(book, to):
        for row in b.book_rows:
            if row.book == book and row.market_key == "spreads":
                line = to if row.selection.side == "SF" else -to
                object.__setattr__(row, "selection", Selection("spread", line=line, side=row.selection.side))
    move("pinnacle", -6.5)
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    assert ("line_move", SPREAD) not in kinds(sigs)          # one book is not a market move
    move("fanduel", -6.5)
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    mv = next(s for s in sigs if s.kind == "line_move" and s.group == SPREAD)
    assert "-5.5 → -6.5" in mv.text and "2 of 4 books" in mv.text


def test_pm_drift_distrusts_history_that_disagrees_with_the_book():
    t = 1_800_000_000
    hist = [(t - 7 * 3600 + i * 300, 0.50) for i in range(84)] + [(t, 0.09)]
    assert pm_drift(hist, 0.2, ("pm", "1"), "Yes", now=0.135) == []       # last point is off the book
    assert pm_drift(hist, 0.2, ("pm", "1"), "Yes") != []                  # without the live mid it would fire
    steady = [(t - 7 * 3600 + i * 300, 0.50) for i in range(84)] + [(t, 0.455)]
    sig = pm_drift(steady, 0.40, ("pm", "1"), "Yes", now=0.45)
    assert sig and "toward fair" in sig[0].text


def test_line_move_ignores_books_moving_opposite_ways():
    a = snapshot(T0, -5.5, {}, ML_EVEN, 1)
    b = snapshot(T0 + timedelta(minutes=30), -5.5, {}, ML_EVEN, 2)
    for row in b.book_rows:
        if row.market_key == "spreads" and row.book in ("pinnacle", "fanduel", "draftkings", "betmgm"):
            to = {"pinnacle": -4.5, "fanduel": -6.5, "draftkings": -4.5, "betmgm": -6.5}[row.book]
            line = to if row.selection.side == "SF" else -to
            object.__setattr__(row, "selection", Selection("spread", line=line, side=row.selection.side))
    sigs, _ = compute_signals([a, b], Config(), "SF", "WAS")
    assert ("line_move", SPREAD) not in kinds(sigs)

"""Market signals computed from our own snapshots (and CLOB price history).

These are context for a human reading the report. None of them feeds the
fair probability.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Protocol

from .config import Config
from .fair import devig
from .models import BookRow
from .store import Snapshot


@dataclass
class Signal:
    kind: str           # line_move | steam | juice | divergence | pm_drift | rlm
    group: tuple        # Selection.group_key()
    text: str
    books: list[str] = field(default_factory=list)
    value: float | None = None
    line: float | None = None   # set when the signal is about one line only

    def applies_to(self, line: float | None) -> bool:
        if self.line is None:
            return True
        if line is None:
            return False
        return abs(self.line) == abs(line) if self.group[0] == "spread" else self.line == line


def is_main_key(key: str) -> bool:
    return "alternate" not in key


def canonical_side(group: tuple, home: str, sides: set[str]) -> str:
    market_type = group[0]
    if market_type in ("moneyline", "spread"):
        return home
    for s in ("over", "yes"):
        if s in sides:
            return s
    return sorted(sides)[0]


def _main_quotes(rows: list[BookRow], home: str, away: str) -> dict[tuple, dict[str, tuple]]:
    """group -> book -> (canonical side, line, no-vig prob, q_side, q_other, prices)
    from main markets only. One entry per book: its main line."""
    by: dict[tuple, dict[str, list[BookRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if is_main_key(r.market_key):
            by[r.selection.group_key()][r.book].append(r)
    out: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for group, books in by.items():
        for book, rs in books.items():
            sides = {r.selection.side for r in rs}
            if "draw" in sides:
                continue
            side = canonical_side(group, home, sides)
            best = None
            for r in rs:
                if r.selection.side != side:
                    continue
                opp = r.selection.opposite(home, away)
                other = next((x for x in rs if x.selection == opp), None)
                if other is None:
                    continue
                p = devig([r.prob_implied, other.prob_implied], "multiplicative")[0]
                cand = (side, r.selection.line, p, r.prob_implied, other.prob_implied,
                        (r.price_american, other.price_american), other.selection.side)
                if best is None or abs(p - 0.5) < abs(best[2] - 0.5):
                    best = cand
            if best is not None:
                out[group][book] = best
    return out


def _group_label(group: tuple, side: str, line: float | None = None) -> str:
    market_type, period, player, stat = group
    bits = [] if period == "full" else [period.upper()]
    if player:
        bits.append(player.title() if not player.isupper() else player)
    bits.append(stat or market_type)
    bits.append(side if len(side) <= 3 else side.title())
    if line is not None:
        bits.append(f"{line:+g}" if market_type == "spread" else f"{line:g}")
    return " ".join(bits)


def _toward(group: tuple, d_line: float, d_prob: float) -> int:
    """+1 if the move makes the canonical side more expensive, -1 if cheaper."""
    if d_line:
        # spread: canonical side's line more negative = it is laying more points.
        # total/prop over: higher number = market moved toward the over.
        return (-1 if d_line > 0 else 1) if group[0] == "spread" else (1 if d_line > 0 else -1)
    if abs(d_prob) >= 0.005:
        return 1 if d_prob > 0 else -1
    return 0


def line_moves(open_q, now_q, cfg: Config) -> list[Signal]:
    """Per-book comparison with the first snapshot. A book that changed its main
    line is a line move; a book that kept it is compared on no-vig probability.
    Needs at least two books, so one book's quirk or a change in which books
    were quoted is not reported as a market move."""
    out = []
    for group, now_books in now_q.items():
        open_books = open_q.get(group, {})
        common = sorted(set(open_books) & set(now_books))
        if len(common) < 2:
            continue
        side = now_books[common[0]][0]
        moved = [b for b in common if open_books[b][1] != now_books[b][1]]
        if len(moved) >= 2:
            o_line = median(open_books[b][1] for b in moved)
            n_line = median(now_books[b][1] for b in moved)
            direction = _toward(group, n_line - o_line, 0.0)
            if direction == 0:      # books moved both ways; nothing net
                continue
            out.append(Signal("line_move", group,
                              f"{_group_label(group, side)}: line {o_line:+g} → {n_line:+g} since open "
                              f"({len(moved)} of {len(common)} books)", moved, direction * 0.01))
            continue
        same = [b for b in common if b not in moved]
        d_p = median(now_books[b][2] - open_books[b][2] for b in same) if len(same) >= 2 else 0.0
        if abs(d_p) >= cfg.line_move_min_prob:
            o_p = median(open_books[b][2] for b in same)
            n_p = median(now_books[b][2] for b in same)
            out.append(Signal("line_move", group,
                              f"{_group_label(group, side)}: no-vig {o_p:.1%} → {n_p:.1%} since open "
                              f"({len(same)} books, median)", same, d_p))
    return out


def steam(prev_q, now_q, gap_min: float, cfg: Config) -> list[Signal]:
    if gap_min > cfg.steam_window_min:
        return []
    out = []
    for group, now_books in now_q.items():
        prev_books = prev_q.get(group, {})
        moved: dict[int, list[str]] = defaultdict(list)
        for book in set(prev_books) & set(now_books):
            p, n = prev_books[book], now_books[book]
            d_line = (n[1] - p[1]) if (n[1] is not None and p[1] is not None) else 0.0
            direction = _toward(group, d_line, n[2] - p[2])
            if direction:
                moved[direction].append(book)
        for direction, books in moved.items():
            if len(books) >= cfg.steam_min_books:
                side = next(iter(now_books.values()))[0]
                word = "toward" if direction > 0 else "against"
                out.append(Signal("steam", group,
                                  f"steam {word} {_group_label(group, side)}: {len(books)} books moved "
                                  f"within {gap_min:.0f} min", sorted(books), float(direction)))
    return out


def juice_asymmetry(now_q, cfg: Config) -> list[Signal]:
    out = []
    for group, books in now_q.items():
        if group[0] == "moneyline":
            continue
        paid: dict[tuple, list[str]] = defaultdict(list)
        detail = {}
        for book, (side, line, _p, q_side, q_other, prices, other_side) in books.items():
            gap = q_side - q_other
            if abs(gap) < cfg.juice_gap_min:
                continue
            fav = side if gap > 0 else other_side
            paid[(fav, line)].append(book)
            at = "" if line is None else (f" {line:+g}" if group[0] == "spread" else f" {line:g}")
            detail[book] = f"{book}{at}: {side} {prices[0]:+.0f} / {other_side} {prices[1]:+.0f}"
        if not paid:
            continue
        by_line: dict[float | None, list[tuple]] = defaultdict(list)
        for (fav, line), bks in paid.items():
            by_line[line].append((fav, bks))
        for line, favs in by_line.items():
            fav, fav_books = max(favs, key=lambda kv: len(kv[1]))
            conflict = sum(len(b) for f, b in favs if f != fav)
            text = (f"juice: {_group_label(group, fav)} being paid for at {len(fav_books)} book(s) "
                    f"({'; '.join(detail[b] for b in sorted(fav_books))})")
            if conflict:
                text += f"; {conflict} book(s) lean the other way"
            out.append(Signal("juice", group, text, sorted(fav_books), line=line))
    return out


def sharp_soft_divergence(rows: list[BookRow], home: str, away: str, cfg: Config,
                          sharp: str = "pinnacle") -> list[Signal]:
    by: dict[tuple, dict[str, list[BookRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r.selection.group_key(), r.selection.line if r.selection.market_type != "spread"
            else abs(r.selection.line or 0))][r.book].append(r)
    out = []
    for (group, _absline), books in by.items():
        if sharp not in books:
            continue
        sides = {r.selection.side for rs in books.values() for r in rs}
        if "draw" in sides:
            continue
        side = canonical_side(group, home, sides)

        def novig(rs):
            for r in rs:
                if r.selection.side != side:
                    continue
                other = next((x for x in rs if x.selection == r.selection.opposite(home, away)), None)
                if other:
                    return r.selection.line, devig([r.prob_implied, other.prob_implied], cfg.devig_method)[0]
            return None
        sharp_q = novig(books[sharp])
        if sharp_q is None:
            continue
        soft = []
        for book, rs in books.items():
            if cfg.is_sharp(book):
                continue
            q = novig(rs)
            if q is not None and q[0] == sharp_q[0]:
                soft.append((book, q[1]))
        if not soft:
            continue
        soft_avg = mean(p for _b, p in soft)
        diff = sharp_q[1] - soft_avg
        if abs(diff) > cfg.divergence_min:
            line = f" {sharp_q[0]:+g}" if group[0] == "spread" else (f" {sharp_q[0]:g}" if sharp_q[0] is not None else "")
            out.append(Signal("divergence", group,
                              f"{_group_label(group, side)}{line}: {sharp} {sharp_q[1]:.1%} vs soft avg "
                              f"{soft_avg:.1%} ({diff * 100:+.1f} pts, {len(soft)} soft books)",
                              [sharp] + sorted(b for b, _ in soft), diff, line=sharp_q[0]))
    return out


def pm_drift(history: list[tuple[int, float]], fair0: float | None, group: tuple,
             label: str, min_move: float = 0.01, now: float | None = None,
             max_gap: float = 0.03) -> list[Signal]:
    """Change in Polymarket price of outcome 0 over 1h and 6h, and whether it
    moved toward the books' fair probability. `now` is the live mid; a history
    whose last point disagrees with it by more than `max_gap` is not trusted."""
    if not history:
        return []
    t_now, p_last = history[-1]
    if now is not None and abs(p_last - now) > max_gap:
        return []
    p_now = now if now is not None else p_last
    parts = []
    biggest = 0.0
    for hours in (1, 6):
        cutoff = t_now - hours * 3600
        before = [p for t, p in history if t <= cutoff]
        if not before:
            continue
        p_then = before[-1]
        d = p_now - p_then
        biggest = max(biggest, abs(d))
        part = f"{hours}h {d * 100:+.1f}¢"
        if fair0 is not None and abs(d) >= min_move:
            part += " toward fair" if abs(p_now - fair0) < abs(p_then - fair0) else " away from fair"
        parts.append(part)
    if not parts or biggest < min_move:
        return []
    return [Signal("pm_drift", group, f"PM {label}: {', '.join(parts)}", value=biggest)]


class BetSplitsProvider(Protocol):
    """Hook for public bet/money percentages. Not free; nothing is scraped.

    Return {group_key: {"side": str, "bets_pct": float}} for whatever data
    you have legitimate access to.
    """
    def splits(self, game_id: str) -> dict[tuple, dict]: ...


class NoSplits:
    def splits(self, game_id: str) -> dict[tuple, dict]:
        return {}


def reverse_line_moves(moves: list[Signal], splits: dict[tuple, dict], home: str) -> list[Signal]:
    out = []
    for s in moves:
        info = splits.get(s.group)
        if not info or s.value is None:
            continue
        public_on_canonical = info.get("side") in (home, "over", "yes")
        if info.get("bets_pct", 0) >= 0.6 and ((s.value < 0) == public_on_canonical):
            out.append(Signal("rlm", s.group, f"reverse line move: {info['bets_pct']:.0%} of bets on "
                                             f"{info['side']}, price moved the other way"))
    return out


def compute_signals(snaps: list[Snapshot], cfg: Config, home: str, away: str,
                    splits: dict[tuple, dict] | None = None) -> tuple[list[Signal], list[str]]:
    """Book-side signals from stored snapshots. Returns (signals, notes)."""
    notes: list[str] = []
    snaps = [s for s in snaps if s.book_rows]
    if not snaps:
        return [], ["no sportsbook snapshots yet; run `pm_edges snapshot` to start tracking"]
    now = snaps[-1]
    now_q = _main_quotes(now.book_rows, home, away)
    sigs: list[Signal] = []
    if len(snaps) >= 2:
        open_q = _main_quotes(snaps[0].book_rows, home, away)
        moves = line_moves(open_q, now_q, cfg)
        sigs += moves
        prev = snaps[-2]
        gap = (now.taken_at - prev.taken_at).total_seconds() / 60
        if gap > cfg.steam_window_min:
            notes.append(f"steam not checked: last two snapshots are {gap:.0f} min apart "
                         f"(window {cfg.steam_window_min:.0f} min)")
        sigs += steam(_main_quotes(prev.book_rows, home, away), now_q, gap, cfg)
        if splits:
            sigs += reverse_line_moves(moves, splits, home)
    else:
        notes.append("one snapshot so far: line movement and steam need at least two")
    sigs += juice_asymmetry(now_q, cfg)
    sigs += sharp_soft_divergence(now.book_rows, home, away, cfg)
    return sigs, notes


def by_group(signals: list[Signal]) -> dict[tuple, list[Signal]]:
    out: dict[tuple, list[Signal]] = defaultdict(list)
    for s in signals:
        out[s.group].append(s)
    return out

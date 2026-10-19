"""Map each Polymarket outcome to sportsbook prices and derive a fair probability.

Lookup order for one selection:
1. Books that quote both sides at the exact line (main or alternate market).
2. Spreads/totals only: each book's two-way line nearest to the target,
   converted with the score model. Labelled ``fair_source = model``.
3. Otherwise no fair value, with the reason.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import Config
from .fair import BookFair, Fair, devig, model_for, weighted_consensus
from .models import BookRow, PMMarket, Selection


@dataclass
class _Pair:
    book: str
    market_key: str
    line: float | None      # the target side's line
    q_side: float           # implied prob of the target side
    q_other: list[float]    # implied probs of the other outcome(s)


class BookIndex:
    def __init__(self, rows: list[BookRow], home: str, away: str):
        self.home, self.away = home, away
        # group_key -> book -> market_key -> [rows]
        self.by_group: dict[tuple, dict[str, dict[str, list[BookRow]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows:
            self.by_group[r.selection.group_key()][r.book][r.market_key].append(r)

    def books_for(self, sel: Selection) -> dict[str, dict[str, list[BookRow]]]:
        return self.by_group.get(sel.group_key(), {})

    def pairs(self, sel: Selection, exact: bool) -> tuple[list[_Pair], list[str]]:
        """Two-way prices per book for `sel`. With exact=False, every line the
        book quotes for this side is returned (for model conversion)."""
        pairs: list[_Pair] = []
        notes: list[str] = []
        for book, by_key in self.books_for(sel).items():
            # main markets before alternates, so a book's main line wins
            for key in sorted(by_key, key=lambda k: k.startswith("alternate_") or k.endswith("_alternate")):
                rows = by_key[key]
                if sel.market_type == "moneyline":
                    sides = {r.selection.side: r for r in rows}
                    if "draw" in sides:
                        notes.append(f"{book}: three-way ({key})")
                        continue
                    me = sides.get(sel.side)
                    others = [r for s, r in sides.items() if s != sel.side]
                    if me and len(others) == 1:
                        pairs.append(_Pair(book, key, None, me.prob_implied, [others[0].prob_implied]))
                    continue
                for r in rows:
                    if r.selection.side != sel.side:
                        continue
                    if exact and r.selection.line != sel.line:
                        continue
                    opp = r.selection.opposite(self.home, self.away)
                    other = next((x for x in rows if x.selection == opp), None)
                    if other is None:
                        at = "" if r.selection.line is None else f" at {r.selection.line:g}"
                        notes.append(f"{book}: only {r.selection.side}{at}")
                        continue
                    pairs.append(_Pair(book, key, r.selection.line, r.prob_implied, [other.prob_implied]))
        return pairs, notes


def _dedupe(pairs: list[_Pair]) -> list[_Pair]:
    """One price per (book, line): the first, i.e. the main market."""
    seen, out = set(), []
    for p in pairs:
        k = (p.book, p.line)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def fair_for(sel: Selection, index: BookIndex, cfg: Config, allow_model: bool = True) -> Fair:
    pairs, notes = index.pairs(sel, exact=True)
    entries = []
    for p in _dedupe(pairs):
        fair_side = devig([p.q_side, *p.q_other], cfg.devig_method)[0]
        entries.append(BookFair(p.book, fair_side, cfg.weight(p.book), p.line, p.market_key))
    if entries:
        return Fair(weighted_consensus(entries), "books", entries)

    if not index.books_for(sel):
        return Fair(None, "none", reason="no book lists this market")

    model = model_for(sel.market_type, cfg) if allow_model and sel.line is not None else None
    if model is not None:
        all_pairs, _ = index.pairs(sel, exact=False)
        best: dict[str, _Pair] = {}
        for p in _dedupe(all_pairs):
            if p.line is None:
                continue
            cur = best.get(p.book)
            if cur is None or abs(p.line - sel.line) < abs(cur.line - sel.line):
                best[p.book] = p
        for p in best.values():
            fair_side = devig([p.q_side, *p.q_other], cfg.devig_method)[0]
            converted = _convert(model, sel, p.line, fair_side)
            entries.append(BookFair(p.book, converted, cfg.weight(p.book), p.line, p.market_key))
        if entries:
            lines = sorted({e.line for e in entries})
            return Fair(weighted_consensus(entries), "model", entries,
                        reason=f"converted from line(s) {', '.join(f'{x:g}' for x in lines)}")

    quoted = {r.selection.side for by_key in index.books_for(sel).values()
              for rows in by_key.values() for r in rows
              if r.selection.line == sel.line or sel.market_type == "moneyline"}
    if len(quoted) == 1:
        return Fair(None, "none", reason=f"one-sided at books (only {quoted.pop()} quoted"
                    + ("" if sel.line is None else f" at {sel.line:g}")
                    + "); no fair value without inventing a vig")
    if notes:
        return Fair(None, "none", reason="; ".join(sorted(set(notes)))[:160] + " (one-sided)")
    if sel.line is not None:
        return Fair(None, "none", reason=f"no book quotes both sides at {sel.line:g}"
                    + ("" if model_for(sel.market_type, cfg) else "; no line model for props"))
    return Fair(None, "none", reason="no two-way price at any book")


def _convert(model, sel: Selection, line_from: float, prob_from: float) -> float:
    if sel.market_type == "total" and sel.side == "under":
        return 1 - model.convert(line_from, 1 - prob_from, sel.line)
    return model.convert(line_from, prob_from, sel.line)


@dataclass
class OutcomeFair:
    market: PMMarket
    outcome_index: int
    fair: Fair


def match_markets(markets: list[PMMarket], rows: list[BookRow], home: str, away: str,
                  cfg: Config) -> list[OutcomeFair]:
    index = BookIndex(rows, home, away)
    out = []
    for m in markets:
        for o in m.outcomes:
            if o.selection is None:
                fair = Fair(None, "none", reason=m.parse_note or "not mapped to a sportsbook market")
            else:
                fair = fair_for(o.selection, index, cfg)
            out.append(OutcomeFair(m, o.index, fair))
    return out

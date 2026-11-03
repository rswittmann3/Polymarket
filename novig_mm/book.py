"""Price grid, binary-market touch and order-book helpers (Novig).

A resting order is a *buy* of one outcome at a price in (0, 1). In a two-outcome market, buying A at p is the same
as selling B at 1 - p, so for outcome A:

    bid_A = best resting buy price of A            (from A's ladder)
    ask_A = 1 - best resting buy price of B        (from B's ladder)
    spread = 1 - bid_A - bid_B                     (the same number from either side)

All prices are `Decimal`; the API sends them as strings. Quantities are contracts that pay $1 at settlement.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Sequence

D = Decimal


def _build_grid() -> tuple[Decimal, ...]:
    """0.001..0.050 step 0.001, 0.055..0.945 step 0.005, 0.950..0.999 step 0.001 (docs: place-an-order)."""
    g = [D(i) / 1000 for i in range(1, 51)]
    g += [D(i) / 1000 for i in range(55, 946, 5)]
    g += [D(i) / 1000 for i in range(950, 1000)]
    return tuple(g)


GRID = _build_grid()
_GRID_LIST = list(GRID)
ONE = D(1)


def is_valid_price(p: Decimal) -> bool:
    i = bisect_left(_GRID_LIST, p)
    return i < len(_GRID_LIST) and _GRID_LIST[i] == p


def round_down(p: Decimal) -> Decimal | None:
    """Highest grid price <= p (for bids). None if p is below the lowest price."""
    i = bisect_right(_GRID_LIST, p)
    return _GRID_LIST[i - 1] if i else None


def round_up(p: Decimal) -> Decimal | None:
    """Lowest grid price >= p (for asks expressed as buys of the other side). None if above the top price."""
    i = bisect_left(_GRID_LIST, p)
    return _GRID_LIST[i] if i < len(_GRID_LIST) else None


def step_up(p: Decimal, n: int = 1) -> Decimal | None:
    """The grid price n ticks above p (p must be on the grid)."""
    i = bisect_left(_GRID_LIST, p) + n
    return _GRID_LIST[i] if 0 <= i < len(_GRID_LIST) else None


def tick_size(p: Decimal) -> Decimal:
    return D("0.005") if D("0.050") < p < D("0.950") else D("0.001")


@dataclass(frozen=True)
class Level:
    price: Decimal
    qty: int
    orders: int = 1


def ladder(orders: Iterable[dict[str, Any]]) -> list[Level]:
    """Aggregate a REST/WS list of resting orders (best price first) into price levels, best first."""
    out: list[Level] = []
    for o in orders:
        p, q = D(str(o["price"])), int(o["qty"])
        if out and out[-1].price == p:
            out[-1] = Level(p, out[-1].qty + q, out[-1].orders + 1)
        else:
            out.append(Level(p, q, 1))
    return out


def parse_book(resp: dict[str, Any]) -> dict[str, list[Level]]:
    """`GET .../markets/{id}/book` -> {outcomeId: [Level, ...]} best price first."""
    return {oid: ladder(orders) for oid, orders in (resp.get("orders") or {}).items()}


@dataclass(frozen=True)
class Touch:
    """Top of book for one binary market, from the listing's per-outcome `bid` fields."""
    market_id: str
    outcome_ids: tuple[str, str]
    bid: tuple[Decimal | None, Decimal | None]       # best buy price per outcome
    bid_qty: tuple[int, int]

    @property
    def spread(self) -> Decimal | None:
        if self.bid[0] is None or self.bid[1] is None:
            return None
        return ONE - self.bid[0] - self.bid[1]

    @property
    def crossed(self) -> bool:
        s = self.spread
        return s is not None and s < 0

    def mid(self, i: int) -> Decimal | None:
        """Mid price of outcome i, only when both sides are quoted."""
        if self.bid[0] is None or self.bid[1] is None:
            return None
        j = 1 - i
        return (ONE + self.bid[i] - self.bid[j]) / 2

    def ask(self, i: int) -> Decimal | None:
        j = 1 - i
        return None if self.bid[j] is None else ONE - self.bid[j]

    def ask_qty(self, i: int) -> int:
        return self.bid_qty[1 - i]


def touch_from_market(m: dict[str, Any]) -> Touch | None:
    """Build a Touch from a catalog market item. None unless the market has exactly two outcomes."""
    outs = m.get("outcomes") or []
    if len(outs) != 2:
        return None
    bids, qtys = [], []
    for o in outs:
        b = o.get("bid")
        bids.append(D(str(b["price"])) if b else None)
        qtys.append(int(b["qty"]) if b else 0)
    return Touch(m["marketId"], (outs[0]["outcomeId"], outs[1]["outcomeId"]), (bids[0], bids[1]), (qtys[0], qtys[1]))


def depth_within(levels: Sequence[Level], best: Decimal | None, ticks: int) -> int:
    """Contracts resting within `ticks` grid steps of the best price (inclusive)."""
    if best is None:
        return 0
    floor = best
    for _ in range(ticks):
        lower = step_up(floor, -1)
        if lower is None:
            break
        floor = lower
    return sum(l.qty for l in levels if floor <= l.price <= best)

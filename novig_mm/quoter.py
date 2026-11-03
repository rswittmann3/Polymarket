"""Dry-run quote engine for a binary Novig market. Pure functions: nothing here touches the network or places orders.

Model. Let A, B be the two outcomes and `fair` the fair probability of A (B is 1 - fair). Orders are buys:
    bid on A at   round_down(fair' - h)          fair' = fair - skew * inv_A / 100
    bid on B at   round_down(1 - fair' - h)      (= selling A at fair' + h)
where inv_A is net contracts long A (holding B counts as short A) and h the half spread. Both quotes are
post-only (`PO`), so they never take liquidity, and are clamped to stay strictly below the opposing best price.
No quotes inside `pull_minutes` of the start or when a stand-down condition holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from .book import D, Touch, ONE, round_down, step_up, is_valid_price


@dataclass(frozen=True)
class QuoteConfig:
    half_spread: Decimal = D("0.015")          # minimum half spread around fair, in probability
    k_sd: Decimal = D("1.0")                   # half spread is at least k_sd * fair_sd
    skew_per_100: Decimal = D("0.005")         # fair shift per 100 contracts of net inventory
    size: int = 100                            # contracts per quote
    max_inventory: int = 300                   # stop adding to one side beyond this net position
    pull_minutes: float = 45.0                 # stand down this close to the start
    min_price: Decimal = D("0.05")             # do not quote outcomes with fair below this ...
    max_price: Decimal = D("0.95")             # ... or above this
    max_book_disagreement: Decimal = D("0.06") # stand down if fair differs from Novig mid by more than this


@dataclass(frozen=True)
class Quote:
    outcome_id: str
    price: Decimal
    qty: int
    tif: str = "PO"
    ticks_from_best: int | None = None         # price vs current best bid for this outcome, in grid steps (0 = joins the touch)


@dataclass(frozen=True)
class QuoteSet:
    quotes: tuple[Quote, ...] = ()
    skipped: str | None = None
    meta: dict = field(default_factory=dict)


def _ticks(a: Decimal, b: Decimal | None) -> int | None:
    """Grid steps from b up to a (negative if a is below b)."""
    if b is None:
        return None
    if a == b:
        return 0
    lo, hi, sign = (b, a, 1) if a > b else (a, b, -1)
    n, p = 0, lo
    while p < hi:
        nxt = step_up(p)
        if nxt is None:
            return None
        p, n = nxt, n + 1
    return sign * n


def make_quotes(touch: Touch, fair: Decimal, fair_sd: Decimal, inv_a: int, minutes_to_start: float,
                cfg: QuoteConfig = QuoteConfig(), market_open: bool = True) -> QuoteSet:
    """Quotes for market `touch` (outcome 0 = A) given fair P(A)."""
    if not market_open:
        return QuoteSet(skipped="market not open")
    if minutes_to_start < cfg.pull_minutes:
        return QuoteSet(skipped=f"inside {cfg.pull_minutes:g} min of start")
    if not (cfg.min_price <= fair <= cfg.max_price):
        return QuoteSet(skipped="fair outside quoting range")
    mid = touch.mid(0)
    if mid is not None and abs(mid - fair) > cfg.max_book_disagreement:
        return QuoteSet(skipped=f"fair {fair} vs Novig mid {mid}: disagreement above {cfg.max_book_disagreement}",
                        meta={"mid": mid})
    h = max(cfg.half_spread, cfg.k_sd * fair_sd)
    fair_adj = fair - cfg.skew_per_100 * D(inv_a) / 100
    want = [round_down(fair_adj - h), round_down(ONE - fair_adj - h)]       # buy A, buy B
    quotes = []
    for i, price in enumerate(want):
        if price is None:
            continue
        if i == 0 and inv_a >= cfg.max_inventory:            # already long A: do not buy more A
            continue
        if i == 1 and inv_a <= -cfg.max_inventory:
            continue
        # a buy of i fills immediately if price + best bid of the other outcome >= 1, so stay strictly below that
        other = touch.bid[1 - i]
        if other is not None and price + other >= ONE:
            cap = round_down(ONE - other)
            price = cap if cap is not None and cap + other < ONE else (step_up(cap, -1) if cap is not None else None)
            if price is None:
                continue
        assert is_valid_price(price), price
        quotes.append(Quote(touch.outcome_ids[i], price, cfg.size, "PO", _ticks(price, touch.bid[i])))
    if not quotes:
        return QuoteSet(skipped="inventory limits", meta={"h": h})
    return QuoteSet(tuple(quotes), None, {"h": h, "fair_adj": fair_adj, "mid": mid})


# --- fill replay ----------------------------------------------------------------------------------------------

@dataclass
class SimOrder:
    outcome_id: str
    price: Decimal
    qty: int
    queue_ahead: int = 0          # contracts resting at our price ahead of us when we joined
    filled: int = 0
    _seen_at_price: int = 0

    @property
    def remaining(self) -> int:
        return self.qty - self.filled


def apply_trade(order: SimOrder, maker_outcome_id: str, maker_price: Decimal, qty: int) -> int:
    """Fill our resting buy against one historical trade. Returns contracts filled by this trade.

    A trade whose maker side is our outcome at price m means every resting buy of that outcome priced above m was
    consumed first. So: m < our price -> we are fully filled; m == our price -> we fill only after `queue_ahead`
    contracts have traded at that price; m > our price -> no effect. Cancels ahead of us are ignored (conservative)."""
    if order.remaining <= 0 or maker_outcome_id != order.outcome_id:
        return 0
    if maker_price < order.price:
        got = order.remaining
    elif maker_price == order.price:
        order._seen_at_price += qty
        # contracts traded at our price beyond the queue ahead of us and beyond what we already got
        got = min(order.remaining, qty, max(0, order._seen_at_price - order.queue_ahead - order.filled))
    else:
        return 0
    order.filled += got
    return got

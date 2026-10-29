"""Bitcoin Up/Down 5-minute markets: find windows where buying both sides costs less than $1.

Up and Down settle to $1 and $0 in opposite order, so one share of each pays exactly $1 whatever
BTC does. Buying both at asks `a_up + a_down` plus the taker fee on each leg locks in
`1 - a_up - a_down - fee_up - fee_down` per pair. The fee is `rate * (p * (1 - p)) ** exponent`
per share (about 1.75c a leg at p = 0.50), so "the two prices add up to under 100%" is not
enough: the ask-sum has to be under 100% by more than the fees. Prices are executable asks from
the CLOB order book, never `outcomePrices`, and the size is found by walking the book.

Nothing here talks to a wallet. Order placement lives in tools/btc5m_arb.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import CLOB_BASE
from .polymarket import fee_per_share

GAMMA_BASE = "https://gamma-api.polymarket.com"
WINDOW = 300  # seconds


def window_start(now_ts: float) -> int:
    return int(now_ts // WINDOW) * WINDOW


def slug_for(start_ts: int) -> str:
    return f"btc-updown-5m-{start_ts}"


@dataclass(frozen=True)
class Market:
    slug: str
    start_ts: int
    up_token: str
    down_token: str
    fee_rate: float
    fee_exponent: float
    min_size: float
    tick: float
    accepting: bool
    neg_risk: bool

    @property
    def end_ts(self) -> int:
        return self.start_ts + WINDOW


def parse_market(raw: dict, start_ts: int) -> Market:
    outcomes = json.loads(raw["outcomes"])
    tokens = json.loads(raw["clobTokenIds"])
    by_name = {str(o).lower(): t for o, t in zip(outcomes, tokens)}
    if set(by_name) != {"up", "down"}:
        raise ValueError(f"unexpected outcomes {outcomes!r}")
    sched = raw.get("feeSchedule") or {}
    if raw.get("feesEnabled") is False:
        rate = 0.0
    else:
        # Fail closed: an unreadable schedule gets a pessimistic fee, not zero.
        rate = float(sched.get("rate", 0.07))
    return Market(
        slug=raw["slug"], start_ts=start_ts, up_token=by_name["up"], down_token=by_name["down"],
        fee_rate=rate, fee_exponent=float(sched.get("exponent", 1)),
        min_size=float(raw.get("orderMinSize") or 5), tick=float(raw.get("orderPriceMinTickSize") or 0.01),
        accepting=bool(raw.get("acceptingOrders")) and bool(raw.get("enableOrderBook")) and not raw.get("closed"),
        neg_risk=bool(raw.get("negRisk")),
    )


def asks_from_book(book: dict) -> list[tuple[float, float]]:
    """(price, size) ascending by price. The CLOB returns asks best-last, so never trust the order."""
    lv = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
    return sorted(p_s for p_s in lv if 0 < p_s[0] < 1 and p_s[1] > 0)


@dataclass(frozen=True)
class Opportunity:
    shares: float
    cost_up: float          # USDC for the Up leg, before fee
    cost_down: float
    fee: float              # total taker fee, both legs
    up_limit: float         # worst Up price touched, for a limit order that sweeps the book
    down_limit: float
    best_up: float
    best_down: float

    @property
    def cost(self) -> float:
        return self.cost_up + self.cost_down + self.fee

    @property
    def profit(self) -> float:
        return self.shares - self.cost

    @property
    def profit_per_share(self) -> float:
        return self.profit / self.shares


def find_opportunity(up_asks, down_asks, *, fee_rate: float, fee_exponent: float = 1.0,
                     min_edge: float = 0.005, max_shares: float = 50.0,
                     min_shares: float = 5.0) -> Opportunity | None:
    """Walk both books, adding one pair at a time while the marginal pair still clears `min_edge`.

    `min_edge` is USDC per pair after fees; it is the cushion against fee rounding and a book that
    moves between the read and the order. Returns None if even the first pair does not clear it, or
    the profitable depth is smaller than the market's minimum order size.
    """
    ui = di = 0
    u_left = up_asks[0][1] if up_asks else 0.0
    d_left = down_asks[0][1] if down_asks else 0.0
    shares = cost_u = cost_d = fee = 0.0
    while ui < len(up_asks) and di < len(down_asks) and shares < max_shares:
        pu, pd = up_asks[ui][0], down_asks[di][0]
        fu = fee_per_share(pu, fee_rate, fee_exponent)
        fd = fee_per_share(pd, fee_rate, fee_exponent)
        if 1.0 - pu - pd - fu - fd < min_edge:
            break
        take = min(u_left, d_left, max_shares - shares)
        shares += take
        cost_u += take * pu
        cost_d += take * pd
        fee += take * (fu + fd)
        u_left -= take
        d_left -= take
        if u_left <= 1e-9:
            ui += 1
            u_left = up_asks[ui][1] if ui < len(up_asks) else 0.0
        if d_left <= 1e-9:
            di += 1
            d_left = down_asks[di][1] if di < len(down_asks) else 0.0
    shares = float(int(shares * 100) // 1) / 100  # floor to 2 dp: order sizes are 2 dp
    if shares < min_shares:
        return None
    # Re-price at the floored size so reported cost matches what will be bought.
    return _reprice(up_asks, down_asks, shares, fee_rate, fee_exponent)


def _fill(asks, shares):
    left, cost, fee_levels, worst = shares, 0.0, [], 0.0
    for p, s in asks:
        t = min(left, s)
        cost += t * p
        fee_levels.append((p, t))
        worst = p
        left -= t
        if left <= 1e-9:
            break
    return cost, fee_levels, worst


def _reprice(up_asks, down_asks, shares, rate, exp) -> Opportunity:
    cu, lu, wu = _fill(up_asks, shares)
    cd, ld, wd = _fill(down_asks, shares)
    fee = sum(t * fee_per_share(p, rate, exp) for p, t in lu + ld)
    return Opportunity(shares=shares, cost_up=cu, cost_down=cd, fee=fee, up_limit=wu, down_limit=wd,
                       best_up=up_asks[0][0], best_down=down_asks[0][0])


# -- network ---------------------------------------------------------------

def fetch_market(client, start_ts: int) -> Market | None:
    r = client.get(f"{GAMMA_BASE}/markets", params={"slug": slug_for(start_ts)})
    r.raise_for_status()
    rows = r.json()
    return parse_market(rows[0], start_ts) if rows else None


def fetch_asks(client, token: str) -> list[tuple[float, float]]:
    r = client.get(f"{CLOB_BASE}/book", params={"token_id": token})
    r.raise_for_status()
    return asks_from_book(r.json())

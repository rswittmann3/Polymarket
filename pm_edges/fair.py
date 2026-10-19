"""Odds conversion, de-vigging, consensus and line conversion.

De-vig methods take the implied probabilities of every outcome of one
market at one book and return probabilities that sum to 1:

* multiplicative: p_i = q_i / sum(q)
* power:          p_i = q_i ** k with k chosen so sum(p) = 1 (removes more
                  margin from longshots; default)
* shin:           Shin (1993) insider-trading model, solved for z
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

_N = NormalDist()


# --------------------------------------------------------------------------
# Odds conversion
# --------------------------------------------------------------------------


def american_to_prob(american: float) -> float:
    if american == 0 or -100 < american < 100:
        raise ValueError(f"invalid American odds: {american}")
    if american > 0:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)


def prob_to_american(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError(f"probability out of range: {p}")
    if p >= 0.5:
        return -100.0 * p / (1 - p)
    return 100.0 * (1 - p) / p


def american_to_decimal(american: float) -> float:
    return 1 + (american / 100.0 if american > 0 else 100.0 / -american)


# --------------------------------------------------------------------------
# De-vig
# --------------------------------------------------------------------------


def _check(probs: list[float]) -> None:
    if len(probs) < 2:
        raise ValueError("de-vig needs at least two outcomes; never invent the missing side")
    if any(not 0 < q < 1 for q in probs):
        raise ValueError(f"implied probabilities must be in (0, 1): {probs}")


def devig_multiplicative(probs: list[float]) -> list[float]:
    _check(probs)
    s = sum(probs)
    return [q / s for q in probs]


def devig_power(probs: list[float]) -> list[float]:
    _check(probs)
    lo, hi = 0.01, 100.0
    for _ in range(200):
        k = (lo + hi) / 2
        if sum(q ** k for q in probs) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    out = [q ** k for q in probs]
    s = sum(out)
    return [p / s for p in out]


def devig_shin(probs: list[float]) -> list[float]:
    _check(probs)
    s = sum(probs)
    if s <= 1:
        return devig_multiplicative(probs)

    def shin_p(z: float) -> list[float]:
        return [(math.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z)) for q in probs]

    lo, hi = 0.0, 0.999
    for _ in range(200):
        z = (lo + hi) / 2
        if sum(shin_p(z)) > 1:
            lo = z
        else:
            hi = z
    out = shin_p((lo + hi) / 2)
    t = sum(out)
    return [p / t for p in out]


DEVIG = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(probs: list[float], method: str = "power") -> list[float]:
    try:
        return DEVIG[method](probs)
    except KeyError:
        raise ValueError(f"unknown de-vig method {method!r}; use one of {sorted(DEVIG)}") from None


# --------------------------------------------------------------------------
# Consensus
# --------------------------------------------------------------------------


@dataclass
class BookFair:
    book: str
    prob: float
    weight: float
    line: float | None = None
    market_key: str = ""


@dataclass
class Fair:
    prob: float | None
    source: str                   # books | model | none
    books: list[BookFair] = field(default_factory=list)
    reason: str = ""

    @property
    def books_label(self) -> str:
        return ",".join(sorted({b.book for b in self.books}))

    def book_prob(self, book: str) -> float | None:
        for b in self.books:
            if b.book == book:
                return b.prob
        return None


def weighted_consensus(entries: list[BookFair]) -> float | None:
    total_w = sum(e.weight for e in entries)
    if not entries or total_w <= 0:
        return None
    return sum(e.prob * e.weight for e in entries) / total_w


# --------------------------------------------------------------------------
# Line conversion (normal approximation over integer outcomes)
# --------------------------------------------------------------------------


class ScoreModel:
    """Discretised normal model of an integer quantity (final margin or game
    total), with optional extra mass on key numbers.

    `cover_prob(mu, line)` is P(X + line > 0) for spreads (X = this side's
    margin) or P(X > line) for totals, conditioned on no push, which is how
    sportsbooks price integer lines.
    """

    def __init__(self, sigma: float, key_weights: dict[int, float] | None = None,
                 kind: str = "spread", span: int = 120):
        if kind not in ("spread", "total"):
            raise ValueError(kind)
        self.sigma = sigma
        self.key_weights = key_weights or {}
        self.kind = kind
        self.span = span

    def pmf(self, mu: float) -> dict[int, float]:
        lo = int(math.floor(mu - self.span)) if self.kind == "spread" else 0
        hi = int(math.ceil(mu + self.span))
        out = {}
        for k in range(lo, hi + 1):
            p = _N.cdf((k + 0.5 - mu) / self.sigma) - _N.cdf((k - 0.5 - mu) / self.sigma)
            if self.kind == "spread":
                p *= self.key_weights.get(abs(k), 1.0)
            out[k] = p
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}

    def cover_prob(self, mu: float, line: float) -> float:
        win = lose = 0.0
        for k, p in self.pmf(mu).items():
            v = (k + line) if self.kind == "spread" else (k - line)
            if v > 1e-9:
                win += p
            elif v < -1e-9:
                lose += p
        return win / (win + lose) if win + lose > 0 else 0.5

    def fit_mu(self, line: float, prob: float) -> float:
        """Mean that makes `cover_prob(mu, line) == prob` (bisection)."""
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        if self.kind == "spread":
            lo, hi = -80.0, 80.0
        else:
            lo, hi = max(0.0, line - 80.0), line + 80.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.cover_prob(mid, line) < prob:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def convert(self, line_from: float, prob_from: float, line_to: float) -> float:
        return self.cover_prob(self.fit_mu(line_from, prob_from), line_to)


def model_for(market_type: str, cfg) -> ScoreModel | None:
    if market_type == "spread":
        return ScoreModel(cfg.margin_sigma, cfg.key_number_weights, "spread")
    if market_type == "total":
        return ScoreModel(cfg.total_sigma, None, "total")
    return None


# --------------------------------------------------------------------------
# Kelly
# --------------------------------------------------------------------------


def kelly_fraction(p: float, cost: float) -> float:
    """Full-Kelly fraction for a $1-payout contract bought at `cost` (price + fee)."""
    if not 0 < cost < 1:
        return 0.0
    return max(0.0, (p - cost) / (1 - cost))

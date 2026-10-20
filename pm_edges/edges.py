"""Edge = fair probability - executable price - taker fee, per outcome."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .fair import Fair, kelly_fraction
from .match import OutcomeFair
from .models import PMMarket, PMOutcome
from .polymarket import effective_fee_rate, fee_per_share, is_stale
from .signals import Signal


@dataclass
class EdgeRow:
    market: PMMarket
    outcome: PMOutcome
    fair: Fair
    ask: float | None
    fee: float | None
    edge: float | None
    pin_edge: float | None      # same edge, but against Pinnacle's no-vig price alone
    flagged: bool
    why_not: str                 # empty when flagged
    fee_note: str
    size_at_ask: float | None    # shares at the best ask (CLOB only)
    size_with_edge: float | None # shares across ask levels that still clear min_edge
    stale: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    kelly_stake: float | None = None

    @property
    def cost(self) -> float | None:
        return None if self.ask is None else self.ask + (self.fee or 0.0)


def checked(markets: list[PMMarket], cfg: Config) -> list[PMMarket]:
    """Markets the edge scan looks at: open, active and liquid enough."""
    return [m for m in markets if m.active and not m.closed and m.liquidity >= cfg.min_liquidity]


def compute_edges(ofs: list[OutcomeFair], cfg: Config, signals: dict[tuple, list[Signal]] | None = None,
                  bankroll: float | None = None) -> list[EdgeRow]:
    signals = signals or {}
    rows = []
    for of in ofs:
        m, fair = of.market, of.fair
        o = m.outcomes[of.outcome_index]
        rate, fee_note = effective_fee_rate(m, cfg)
        ask = o.ask
        fee = fee_per_share(ask, rate, m.fee_exponent) if ask is not None else None
        edge = None if (fair.prob is None or ask is None) else fair.prob - ask - (fee or 0.0)
        pin = fair.book_prob("pinnacle") if fair.source == "books" else None
        pin_edge = None if (pin is None or ask is None) else pin - ask - (fee or 0.0)

        why = ""
        if not m.active or m.closed:
            why = "closed"
        elif m.liquidity < cfg.min_liquidity:
            why = f"liquidity {m.liquidity:.0f} < {cfg.min_liquidity:.0f}"
        elif fair.prob is None:
            why = fair.reason or "no fair value"
        elif ask is None or not 0 < ask < 1:
            why = "no ask"
        elif fair.source == "model" and not cfg.allow_model:
            why = "model-converted line (use --allow-model to flag)"
        elif edge < cfg.min_edge:
            why = f"edge below {cfg.min_edge * 100:.1f} pts"
        elif fair.source == "books" and len(fair.books) < cfg.min_books_to_flag \
                and not any(cfg.is_sharp(b.book) for b in fair.books):
            why = f"fair value rests on one soft book ({fair.books_label})"
        elif cfg.confirm_sharp and pin_edge is not None and pin_edge < cfg.min_edge:
            why = f"Pinnacle does not confirm ({pin_edge * 100:+.1f} pts vs Pinnacle)"

        # shares that still clear min_edge, counting at most max_stake dollars
        size_with_edge = None
        if fair.prob is not None and o.ask_levels:
            size_with_edge, budget = 0.0, cfg.max_stake
            for p, s in o.ask_levels:
                if fair.prob - p - fee_per_share(p, rate, m.fee_exponent) < cfg.min_edge or budget <= 0:
                    continue
                take = min(s, budget / p)
                size_with_edge += take
                budget -= take * p

        stake = None
        if bankroll and not why and fair.prob is not None and ask is not None:
            f = kelly_fraction(fair.prob, ask + (fee or 0.0)) * cfg.kelly_fraction
            stake = round(min(bankroll * min(f, cfg.kelly_cap), cfg.max_stake), 2)

        line = o.selection.line if o.selection else None
        sigs = (signals.get(o.selection.group_key(), []) if o.selection else []) \
            + signals.get(("pm", m.market_id), [])
        rows.append(EdgeRow(
            market=m, outcome=o, fair=fair, ask=ask, fee=fee, edge=edge, pin_edge=pin_edge,
            flagged=not why, why_not=why, fee_note=f"{rate:g} ({fee_note})",
            size_at_ask=o.ask_size, size_with_edge=size_with_edge,
            stale=is_stale(m, cfg),
            signals=[s.text for s in sigs if s.applies_to(line)],
            kelly_stake=stake,
        ))
    return rows

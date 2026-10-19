"""Shared record types.

A ``Selection`` names one side of one wager the same way on both sources
(e.g. SF -7.5 full game, or Over 245.5 Brock Purdy passing yards), so a
Polymarket outcome can be looked up among the sportsbook rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

MARKET_TYPES = ("moneyline", "spread", "total", "player_prop", "quarter", "other")


def norm_player(name: str) -> str:
    """'Brian Robinson Jr.' -> 'brian robinson'; for matching only."""
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Selection:
    market_type: str
    period: str = "full"        # full | h1 | h2 | q1 | q2 | q3 | q4
    player: str = ""            # normalised player name, or team abbr for team totals
    stat: str = ""              # pass_yds, rush_yds, anytime_td, team_total, ...
    line: float | None = None   # spread: this side's handicap; total/prop: the number
    side: str = ""              # team abbr | over | under | yes | no

    def opposite(self, home: str, away: str) -> "Selection":
        flip = {"over": "under", "under": "over", "yes": "no", "no": "yes",
                home: away, away: home}
        line = -self.line if (self.market_type == "spread" and self.line is not None) else self.line
        return Selection(self.market_type, self.period, self.player, self.stat,
                         line, flip.get(self.side, self.side))

    def group_key(self) -> tuple:
        """Identifies the market, ignoring line and side."""
        return (self.market_type, self.period, self.player, self.stat)

    def label(self) -> str:
        bits = [] if self.period == "full" else [self.period.upper()]
        if self.player:
            bits.append(self.player.title() if not self.player.isupper() else self.player)
        if self.stat and self.stat != "team_total":
            bits.append(self.stat)
        bits.append(self.side.upper() if len(self.side) <= 3 else self.side.title())
        if self.line is not None:
            bits.append(f"{self.line:+g}" if self.market_type == "spread" else f"{self.line:g}")
        return " ".join(bits)


@dataclass
class BookRow:
    game_id: str
    book: str
    market_key: str             # Odds API market key, e.g. spreads, alternate_totals
    selection: Selection
    price_american: float
    prob_implied: float
    fetched_at: datetime
    source: str = "book"
    liquidity: float | None = None


@dataclass
class PMOutcome:
    index: int
    name: str
    token_id: str | None
    selection: Selection | None
    bid: float | None = None    # best price someone will pay for this outcome
    ask: float | None = None    # best price you can buy this outcome at
    ask_size: float | None = None   # shares available at `ask`
    ask_levels: list[tuple[float, float]] = field(default_factory=list)  # (price, size) ascending
    bid_levels: list[tuple[float, float]] = field(default_factory=list)  # (price, size) descending
    price_source: str = "gamma"     # gamma | clob


@dataclass
class PMMarket:
    game_id: str
    event_slug: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    market_type: str
    sports_market_type: str | None
    line: float | None
    outcomes: list[PMOutcome]
    outcome_prices: list[float]
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    liquidity: float
    volume: float
    game_start_time: str | None
    end_date: str | None
    active: bool
    closed: bool
    fees_enabled: bool | None
    fee_rate: float | None
    fee_exponent: float
    parse_note: str = ""
    fetched_at: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

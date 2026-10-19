"""Thresholds, sportsbook weights and .env loading.

Every number that changes an output lives here, so a run can be reproduced
from its config. CLI flags override the defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
ODDS_BASE = "https://api.the-odds-api.com/v4"

LEAGUE_SPORT_KEYS = {"nfl": "americanfootball_nfl"}

# Odds API bookmaker keys. Check them against the `key` values in a real
# response; a book that is missing from the response is skipped.
DEFAULT_BOOK_WEIGHTS: dict[str, float] = {
    "pinnacle": 3.0,
    "circasports": 2.0,
    "betonlineag": 2.0,
    "bookmaker": 2.0,
    "lowvig": 2.0,
}
DEFAULT_BOOK_WEIGHT = 1.0
SHARP_BOOK = "pinnacle"


@dataclass
class Config:
    data_dir: Path = Path("data")
    odds_api_key: str | None = None

    # Polymarket filters
    min_liquidity: float = 100.0
    stale_trade_gap: float = 0.03   # |lastTradePrice - mid|
    stale_spread: float = 0.05      # bestAsk - bestBid

    # Fees. Polymarket's documented taker fee is shares * rate * p * (1 - p)
    # (per-market `feeSchedule`); makers pay none. `default_fee_rate` is used
    # only when a market does not expose its own schedule and does not say
    # `feesEnabled: false`. It is a conservative guess; check it.
    default_fee_rate: float = 0.05
    fee_rate_override: float | None = None

    # Fair probability
    devig_method: str = "power"     # multiplicative | power | shin
    book_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BOOK_WEIGHTS))
    default_book_weight: float = DEFAULT_BOOK_WEIGHT
    margin_sigma: float = 13.5      # NFL final margin, points
    total_sigma: float = 10.0       # NFL game total, points
    # Multipliers on the probability mass of exact final margins |m| in the
    # discretised normal margin model. With sigma 13.5 they put about 14% of
    # close games on a 3-point margin and about 8% on 7, roughly the long-run
    # NFL rates (approximate; tune if you have data). Ties are rare.
    key_number_weights: dict[int, float] = field(
        default_factory=lambda: {0: 0.1, 3: 2.7, 7: 1.8}
    )

    # Odds API
    regions: str = "us,eu"          # eu is where Pinnacle is listed
    fetch_alternates: bool = True

    # Edge flagging
    min_edge: float = 0.03
    allow_model: bool = False

    # Kelly (off unless --kelly)
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.02         # max fraction of bankroll per market

    # Signals
    steam_window_min: float = 10.0
    steam_min_books: int = 3
    line_move_min_prob: float = 0.01
    juice_gap_min: float = 0.02     # implied-prob gap between sides at one book
    divergence_min: float = 0.02    # Pinnacle vs soft-book average no-vig

    # HTTP
    timeout_s: float = 20.0
    retries: int = 3
    concurrency: int = 8

    def weight(self, book: str) -> float:
        return self.book_weights.get(book, self.default_book_weight)

    def is_sharp(self, book: str) -> bool:
        return self.weight(book) > self.default_book_weight

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "snapshots.db"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal KEY=VALUE loader; existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(**overrides) -> Config:
    load_dotenv()
    cfg = Config(odds_api_key=os.environ.get("ODDS_API_KEY") or None)
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg

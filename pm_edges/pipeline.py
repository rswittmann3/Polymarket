"""Glue: fetch both sources for one game, match, compute signals and edges."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone

import httpx

from .config import LEAGUE_SPORT_KEYS, Config
from .edges import EdgeRow, checked, compute_edges
from .httpclient import ApiError, JsonClient, RawStore, utcnow
from .match import OutcomeFair, match_markets
from .models import BookRow, PMMarket
from .polymarket import PolymarketClient, event_markets
from .signals import NoSplits, Signal, by_group, compute_signals, pm_drift
from .sportsbooks import OddsApiClient, missing_lines, needed_market_keys, normalize_event_odds
from .store import Snapshot, Store
from .teams import canonical_abbr


@dataclass
class GameSpec:
    league: str
    away: str
    home: str
    date: date
    kickoff: time | None = None
    tz: str = "America/New_York"

    def __post_init__(self):
        self.league = self.league.lower()
        self.away = canonical_abbr(self.away)
        self.home = canonical_abbr(self.home)

    @property
    def game_id(self) -> str:
        return f"{self.league}-{self.away}-{self.home}-{self.date.isoformat()}".lower()

    @property
    def sport_key(self) -> str:
        try:
            return LEAGUE_SPORT_KEYS[self.league]
        except KeyError:
            raise ApiError(f"league {self.league!r} not supported yet") from None


@dataclass
class Run:
    spec: GameSpec
    cfg: Config
    base_slug: str | None = None
    markets: list[PMMarket] = field(default_factory=list)
    book_rows: list[BookRow] = field(default_factory=list)
    clob_priced: int = 0
    histories: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    quota_remaining: str | None = None
    odds_calls: list[tuple[str, str]] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=utcnow)
    raw_dir: str = ""


def make_http(cfg: Config, spec: GameSpec, offline: bool, offline_run: str | None = None,
              transport: httpx.BaseTransport | None = None) -> JsonClient:
    store = RawStore(cfg.raw_dir, spec.game_id)
    return JsonClient(store, offline=offline, offline_run=offline_run, timeout=cfg.timeout_s,
                      retries=cfg.retries, transport=transport)


def fetch_polymarket(run: Run, http: JsonClient, use_clob: bool = True, history: bool = False) -> None:
    spec, cfg = run.spec, run.cfg
    pm = PolymarketClient(http, cfg)
    base, events = pm.discover(spec.league, spec.away, spec.home, spec.date, spec.kickoff, spec.tz)
    run.base_slug = base
    if not events:
        run.warnings += pm.warnings
        raise ApiError(f"no Polymarket event found for {spec.away} @ {spec.home} on {spec.date} "
                       f"(tried slugs for {spec.date} and the next day)")
    fetched = http.latest_fetch or utcnow()
    run.markets = event_markets(events, spec.game_id, spec.home, spec.away, fetched)
    live = [m for m in run.markets if m.active and not m.closed]
    if use_clob:
        run.clob_priced = pm.attach_books(live)
    if history:
        run.histories = pm.histories(checked(run.markets, cfg))
    run.warnings += pm.warnings


def fetch_books(run: Run, http: JsonClient) -> None:
    spec, cfg = run.spec, run.cfg
    odds = OddsApiClient(http, cfg, spec.sport_key)
    try:
        event = odds.find_event(spec.home, spec.away, spec.date)
        live = [m for m in run.markets if m.active and not m.closed]
        keys = needed_market_keys(live)
        resp = _event_odds_with_fallback(odds, event["id"], keys, run)
        rows = normalize_event_odds(resp.body, spec.game_id, resp.fetched_at)
        if cfg.fetch_alternates:
            alt = missing_lines(live, rows)
            if alt:
                resp2 = _event_odds_with_fallback(odds, event["id"], alt, run)
                rows += normalize_event_odds(resp2.body, spec.game_id, resp2.fetched_at)
        run.book_rows = rows
    except ApiError as exc:
        run.warnings.append(f"sportsbooks: {exc}")
    run.quota_remaining = odds.remaining
    run.odds_calls = odds.calls


def _event_odds_with_fallback(odds: OddsApiClient, event_id: str, keys: set[str], run: Run):
    try:
        return odds.event_odds(event_id, keys)
    except ApiError as exc:
        featured = {k for k in keys if not k.startswith("player_")}
        if "HTTP 422" not in str(exc) or featured == keys or not featured:
            raise
        run.warnings.append(f"odds request rejected ({exc}); retrying without player props")
        return odds.event_odds(event_id, featured)


def collect(spec: GameSpec, cfg: Config, *, offline: bool = False, offline_run: str | None = None,
            books: bool = True, use_clob: bool = True, history: bool = False,
            transport: httpx.BaseTransport | None = None) -> Run:
    run = Run(spec, cfg)
    http = make_http(cfg, spec, offline, offline_run, transport)
    try:
        fetch_polymarket(run, http, use_clob=use_clob, history=history)
        if books:
            fetch_books(run, http)
    finally:
        http.close()
    run.fetched_at = http.latest_fetch or utcnow()
    run.raw_dir = str(http.store.run_dir if not offline else http.store.game_dir)
    return run


@dataclass
class Analysis:
    run: Run
    outcome_fairs: list[OutcomeFair]
    edges: list[EdgeRow]
    signals: list[Signal]
    signal_notes: list[str]
    checked: list[PMMarket]


def analyze(run: Run, store: Store | None, bankroll: float | None = None,
            splits_provider=None) -> Analysis:
    spec, cfg = run.spec, run.cfg
    live = checked(run.markets, cfg)
    ofs = match_markets(live, run.book_rows, spec.home, spec.away, cfg)

    snaps = store.snapshots(spec.game_id) if store else []
    if run.book_rows and not (snaps and snaps[-1].taken_at == run.fetched_at):
        snaps = snaps + [Snapshot(-1, run.fetched_at, run.book_rows, [])]
    splits = (splits_provider or NoSplits()).splits(spec.game_id)
    sigs, notes = compute_signals(snaps, cfg, spec.home, spec.away, splits)

    fair0 = {of.market.market_id: of.fair.prob for of in ofs if of.outcome_index == 0}
    for m in live:
        hist = run.histories.get(m.market_id) or _history_from_snapshots(snaps, m.market_id)
        o0 = m.outcomes[0] if m.outcomes else None
        sigs += pm_drift(hist, fair0.get(m.market_id), ("pm", m.market_id),
                         o0.name if o0 else m.question)

    edges = compute_edges(ofs, cfg, by_group(sigs), bankroll)
    return Analysis(run, ofs, edges, sigs, notes, live)


def _history_from_snapshots(snaps: list[Snapshot], market_id: str) -> list[tuple[int, float]]:
    out = []
    for s in snaps:
        for r in s.pm_rows:
            if r["market_id"] == market_id and r["outcome_index"] == 0 \
                    and r["bid"] is not None and r["ask"] is not None:
                out.append((int(s.taken_at.replace(tzinfo=s.taken_at.tzinfo or timezone.utc).timestamp()),
                            (r["bid"] + r["ask"]) / 2))
    return sorted(out)

"""Polymarket Gamma + CLOB clients, slug discovery and market normalisation.

Executable prices: to buy outcome 0 ("Yes") you pay ``bestAsk``; to buy
outcome 1 ("No") you pay ``1 - bestBid``. ``outcomePrices`` is often a stale
last trade and is never used as a price. When the CLOB order book is
reachable, each outcome token's own best ask replaces those numbers.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .config import CLOB_BASE, GAMMA_BASE, Config
from .httpclient import ApiError, JsonClient
from .models import PMMarket, PMOutcome, Selection, norm_player
from .teams import canonical_abbr, search_terms, team_from_text

# --------------------------------------------------------------------------
# Field parsing
# --------------------------------------------------------------------------


def parse_json_list(value: Any) -> list:
    """`outcomes`, `outcomePrices`, `clobTokenIds` are JSON-encoded strings."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [v.strip() for v in value.split(",") if v.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_fee_schedule(m: dict) -> tuple[bool | None, float | None, float]:
    """(feesEnabled, rate, exponent) from a Gamma market, as far as it says."""
    enabled = m.get("feesEnabled")
    sched = m.get("feeSchedule") or m.get("fee_schedule")
    if isinstance(sched, str):
        try:
            sched = json.loads(sched)
        except json.JSONDecodeError:
            sched = None
    rate = exponent = None
    if isinstance(sched, dict):
        rate = to_float(sched.get("rate"))
        exponent = to_float(sched.get("exponent"))
    return (enabled if isinstance(enabled, bool) else None, rate, exponent or 1.0)


# --------------------------------------------------------------------------
# Slugs
# --------------------------------------------------------------------------


def slug_dates(local_date: date, kickoff: time | None = None,
               tz: str = "America/New_York") -> list[date]:
    """Dates to try in the event slug, most likely first.

    Polymarket slugs appear to use the UTC date, so an evening US kickoff
    (Monday 8:15 pm ET = Tuesday 00:15 UTC) lands on the next day. With a
    known kickoff the UTC date goes first; the other date is still tried.
    """
    nxt = local_date + timedelta(days=1)
    if kickoff is None:
        return [local_date, nxt]
    utc_day = datetime.combine(local_date, kickoff, ZoneInfo(tz)).astimezone(timezone.utc).date()
    return [utc_day] + [d for d in (local_date, nxt) if d != utc_day]


def game_slug(league: str, away: str, home: str, d: date) -> str:
    return f"{league.lower()}-{away.lower()}-{home.lower()}-{d.isoformat()}"


# --------------------------------------------------------------------------
# Question parsing -> Selection
# --------------------------------------------------------------------------

# Longer phrases first: "rushing + receiving yards" before "rushing yards".
STAT_PATTERNS: list[tuple[str, str]] = [
    (r"rush(?:ing)?\s*(?:\+|and|&)\s*rec(?:eiving)?\s*yards|rush\s*\+\s*rec\s*yds", "rush_reception_yds"),
    (r"pass(?:ing)?\s*(?:\+|and|&)\s*rush(?:ing)?\s*yards", "pass_rush_yds"),
    (r"pass(?:ing)?\s*(?:yards|yds)", "pass_yds"),
    (r"pass(?:ing)?\s*(?:touchdowns|tds)", "pass_tds"),
    (r"pass(?:ing)?\s*completions|completions", "pass_completions"),
    (r"pass(?:ing)?\s*attempts", "pass_attempts"),
    (r"interceptions(?:\s*thrown)?|ints", "pass_interceptions"),
    (r"rush(?:ing)?\s*(?:touchdowns|tds)", "rush_tds"),
    (r"rec(?:eiving)?\s*(?:touchdowns|tds)", "reception_tds"),
    (r"rush(?:ing)?\s*(?:yards|yds)", "rush_yds"),
    (r"rush(?:ing)?\s*attempts|carries", "rush_attempts"),
    (r"rec(?:eiving)?\s*(?:yards|yds)", "reception_yds"),
    (r"receptions|catches", "receptions"),
    (r"sacks", "sacks"),
    (r"(?:anytime\s*)?(?:touchdown|td)s?", "anytime_td"),
]


def parse_stat(text: str) -> str | None:
    t = text.lower()
    for pattern, stat in STAT_PATTERNS:
        if re.search(rf"\b(?:{pattern})\b", t):
            return stat
    return None


def parse_period(text: str) -> str:
    t = text.lower().replace("_", " ")
    if re.search(r"\b(1h|first half|1st half|1st-half|first-half)\b", t):
        return "h1"
    if re.search(r"\b(2h|second half|2nd half)\b", t):
        return "h2"
    for n, words in ((1, "1st|first"), (2, "2nd|second"), (3, "3rd|third"), (4, "4th|fourth")):
        if re.search(rf"\b(q{n}|(?:{words})[ -]quarter)\b", t):
            return f"q{n}"
    return "full"


_NUM = r"\d+(?:\.\d+)?"
_VERBS = r"(?:have|record|get|throw for|throw|rush for|catch|gain|total|tally|score)"

PROP_OU = re.compile(rf"^(?P<player>[^:?]+?)(?::|\s[-–])\s*(?P<stat>.+?)\s+(?:o/u|over/under)\s+(?P<line>{_NUM})", re.I)
PROP_OVER_Q = re.compile(rf"^will\s+(?P<player>.+?)\s+{_VERBS}\s+(?:over|more than)\s+(?P<line>{_NUM})\s+(?P<stat>.+?)\??$", re.I)
PROP_MILESTONE_Q = re.compile(rf"^will\s+(?P<player>.+?)\s+{_VERBS}\s+(?P<n>\d+)\+\s*(?P<stat>.+?)\??$", re.I)
PROP_MILESTONE_COLON = re.compile(r"^(?P<player>[^:?]+?)\s*:\s*(?P<n>\d+)\+\s*(?P<stat>.+?)\??$", re.I)
PROP_TD_Q = re.compile(r"^will\s+(?P<player>.+?)\s+score\s+(?:a|an|any|1\+)\s*(?:anytime\s+)?(?:touchdown|td)", re.I)
PROP_TD_COLON = re.compile(r"^(?P<player>[^:?]+?)\s*:\s*(?:anytime\s+)?(?:touchdown|td)(?:\s*scorer)?\??$", re.I)
SPREAD_TEAM = re.compile(rf"(?P<team>[A-Za-z0-9 .'&]+?)\s*\((?P<line>[+-]?{_NUM})\)")
TOTAL_LINE = re.compile(rf"(?:o/u|over/under|total)\s*(?P<line>{_NUM})", re.I)


def _side_name(name: str) -> str:
    n = name.strip().lower()
    return {"over": "over", "under": "under", "yes": "yes", "no": "no"}.get(n, n)


def classify(m: dict, home: str, away: str) -> tuple[str, list[Selection | None], str]:
    """Return (market_type, per-outcome selections, note) for a Gamma market."""
    question = (m.get("question") or "").strip()
    smt = (m.get("sportsMarketType") or "").lower()
    outcomes = [str(o) for o in parse_json_list(m.get("outcomes"))]
    sides = [_side_name(o) for o in outcomes]
    period = parse_period(f"{question} {m.get('slug', '')} {smt}")
    line_field = to_float(m.get("line"))
    q = question.lower()
    none = [None] * len(outcomes)
    if len(outcomes) != 2:
        return "other", none, f"{len(outcomes)} outcomes; only two-way markets are priced"

    teams = [team_from_text(o) for o in outcomes]
    is_two_teams = all(teams) and teams[0] != teams[1] and set(teams) <= {home, away}

    # ---- player props --------------------------------------------------
    sel = _parse_prop(question, sides, period)
    if sel is not None:
        return "player_prop", sel, ""
    if "player" in smt or "prop" in smt:
        return "player_prop", none, "player prop question not recognised"

    # ---- quarter/half novelty markets (no sportsbook twin) --------------
    if "quarter" in q and not is_two_teams and set(sides) <= {"yes", "no"}:
        return "quarter", none, "no sportsbook equivalent"
    if "highest scoring" in q or "highest-scoring" in q:
        return "quarter", none, "no sportsbook equivalent"

    # ---- spread ----------------------------------------------------------
    if "spread" in smt or q.startswith("spread") or (is_two_teams and SPREAD_TEAM.search(question)):
        if not is_two_teams:
            return "spread", none, "spread outcomes are not the two teams"
        mt = SPREAD_TEAM.search(question)
        team = team_from_text(mt.group("team")) if mt else None
        line = to_float(mt.group("line")) if mt else None
        note = ""
        if team is None or line is None:
            if line_field is None:
                return "spread", none, "could not read spread line"
            team, line = teams[0], line_field
            note = "line sign assumed to apply to the first outcome"
        sels = [Selection("spread", period, line=line if t == team else -line, side=t) for t in teams]
        return "spread", sels, note

    # ---- totals (game or team) -------------------------------------------
    if set(sides) == {"over", "under"} or "total" in smt or "o/u" in q:
        mt = TOTAL_LINE.search(question)
        line = to_float(mt.group("line")) if mt else line_field
        if line is None:
            return "total", none, "could not read total line"
        if set(sides) != {"over", "under"}:
            return "total", none, "total outcomes are not Over/Under"
        team = None
        if "team total" in q or "team_total" in smt:
            team = team_from_text(re.split(r"team total", question, flags=re.I)[0])
            if team is None:
                return "total", none, "team total: team not recognised"
        sels = [Selection("total", period, player=team or "", stat="team_total" if team else "",
                          line=line, side=s) for s in sides]
        return "total", sels, ""

    # ---- moneyline ---------------------------------------------------------
    if is_two_teams and ("moneyline" in smt or not smt or "winner" in q or " vs" in q):
        return "moneyline", [Selection("moneyline", period, side=t) for t in teams], ""

    return "other", none, "market type not recognised"


def _parse_prop(question: str, sides: list[str], period: str) -> list[Selection] | None:
    def mk(player, stat, line, yes_side):
        p = norm_player(player)
        if set(sides) == {"over", "under"}:
            return [Selection("player_prop", period, p, stat, line, s) for s in sides]
        if set(sides) == {"yes", "no"}:
            other = {"over": "under", "under": "over", "yes": "no"}[yes_side]
            return [Selection("player_prop", period, p, stat, line, yes_side if s == "yes" else other)
                    for s in sides]
        return None

    for rx in (PROP_TD_Q, PROP_TD_COLON):
        m = rx.match(question)
        if m:
            return mk(m.group("player"), "anytime_td", None, "yes")
    m = PROP_OU.match(question)
    if m and (stat := parse_stat(m.group("stat"))) and stat != "anytime_td":
        return mk(m.group("player"), stat, float(m.group("line")), "over")
    m = PROP_OVER_Q.match(question)
    if m and (stat := parse_stat(m.group("stat"))):
        return mk(m.group("player"), stat, float(m.group("line")), "over")
    for rx in (PROP_MILESTONE_Q, PROP_MILESTONE_COLON):
        m = rx.match(question)
        if m and (stat := parse_stat(m.group("stat"))):
            # "250+ yards" is the same wager as Over 249.5
            if stat == "anytime_td":
                stat = "tds_over"   # "2+ touchdowns"
            return mk(m.group("player"), stat, int(m.group("n")) - 0.5, "over")
    return None


def normalize_market(m: dict, event_slug: str, game_id: str, home: str, away: str,
                     fetched_at: datetime | None = None) -> PMMarket:
    outcomes = [str(o) for o in parse_json_list(m.get("outcomes"))]
    prices = [to_float(p) for p in parse_json_list(m.get("outcomePrices"))]
    tokens = [str(t) for t in parse_json_list(m.get("clobTokenIds"))]
    market_type, sels, note = classify(m, home, away)
    bid, ask = to_float(m.get("bestBid")), to_float(m.get("bestAsk"))
    fees_enabled, fee_rate, fee_exp = parse_fee_schedule(m)

    pm_outcomes = []
    for i, name in enumerate(outcomes):
        o = PMOutcome(index=i, name=name, token_id=tokens[i] if i < len(tokens) else None,
                      selection=sels[i] if i < len(sels) else None)
        # Outcome 0 trades at the market's quoted bid/ask; outcome 1 is its mirror.
        if i == 0:
            o.bid, o.ask = bid, ask
        elif i == 1 and len(outcomes) == 2:
            o.bid = None if ask is None else round(1 - ask, 6)
            o.ask = None if bid is None else round(1 - bid, 6)
        pm_outcomes.append(o)

    return PMMarket(
        game_id=game_id, event_slug=event_slug,
        market_id=str(m.get("id", "")), condition_id=str(m.get("conditionId", "")),
        slug=m.get("slug", ""), question=m.get("question", ""),
        market_type=market_type, sports_market_type=m.get("sportsMarketType"),
        line=to_float(m.get("line")), outcomes=pm_outcomes,
        outcome_prices=[p for p in prices if p is not None],
        best_bid=bid, best_ask=ask, last_trade_price=to_float(m.get("lastTradePrice")),
        liquidity=to_float(m.get("liquidityNum")) or to_float(m.get("liquidity")) or 0.0,
        volume=to_float(m.get("volumeNum")) or to_float(m.get("volume")) or 0.0,
        game_start_time=m.get("gameStartTime"), end_date=m.get("endDate"),
        active=bool(m.get("active", True)), closed=bool(m.get("closed", False)),
        fees_enabled=fees_enabled, fee_rate=fee_rate, fee_exponent=fee_exp,
        parse_note=note, fetched_at=fetched_at,
    )


def is_stale(m: PMMarket, cfg: Config) -> list[str]:
    reasons = []
    if m.mid is not None and m.last_trade_price is not None \
            and abs(m.last_trade_price - m.mid) > cfg.stale_trade_gap:
        reasons.append(f"last {m.last_trade_price:.3f} vs mid {m.mid:.3f}")
    if m.spread is not None and m.spread > cfg.stale_spread:
        reasons.append(f"wide {m.spread:.3f}")
    if m.best_bid is None or m.best_ask is None:
        reasons.append("no quote")
    return reasons


def fee_per_share(price: float, rate: float, exponent: float = 1.0) -> float:
    """Taker fee in USDC per share bought at `price`: rate * (p * (1-p))**exponent."""
    if rate <= 0 or not 0 < price < 1:
        return 0.0
    return rate * (price * (1 - price)) ** exponent


def effective_fee_rate(m: PMMarket, cfg: Config) -> tuple[float, str]:
    if cfg.fee_rate_override is not None:
        return cfg.fee_rate_override, "override"
    if m.fee_rate is not None:
        return (m.fee_rate if m.fees_enabled is not False else 0.0), "market feeSchedule"
    if m.fees_enabled is False:
        return 0.0, "feesEnabled=false"
    return cfg.default_fee_rate, "default (unverified)"


# --------------------------------------------------------------------------
# Order book
# --------------------------------------------------------------------------


def parse_book(book: dict) -> dict:
    """Best bid/ask from a CLOB /book response. Prices arrive as strings and
    the arrays' sort order is not guaranteed, so sort here."""
    def levels(key):
        out = []
        for lvl in book.get(key) or []:
            p, s = to_float(lvl.get("price")), to_float(lvl.get("size"))
            if p is not None and s is not None and s > 0:
                out.append((p, s))
        return out
    bids = sorted(levels("bids"), reverse=True)
    asks = sorted(levels("asks"))
    return {
        "bid": bids[0][0] if bids else None,
        "bid_size": sum(s for p, s in bids if p == bids[0][0]) if bids else None,
        "ask": asks[0][0] if asks else None,
        "ask_size": sum(s for p, s in asks if p == asks[0][0]) if asks else None,
        "asks": asks,
        "tick_size": to_float(book.get("tick_size")),
    }


def apply_book(o: PMOutcome, parsed: dict) -> None:
    o.bid, o.ask = parsed["bid"], parsed["ask"]
    o.ask_size, o.ask_levels = parsed["ask_size"], parsed["asks"]
    o.price_source = "clob"


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class PolymarketClient:
    def __init__(self, http: JsonClient, cfg: Config):
        self.http = http
        self.cfg = cfg
        self.warnings: list[str] = []

    # Gamma ---------------------------------------------------------------
    def event_by_slug(self, slug: str) -> dict | None:
        body = self.http.get(f"{GAMMA_BASE}/events", {"slug": slug}).body
        if isinstance(body, list):
            return body[0] if body else None
        if isinstance(body, dict) and body.get("slug") == slug:
            return body
        return None

    def search_event_slugs(self, query: str) -> list[str]:
        body = self.http.get(f"{GAMMA_BASE}/public-search",
                             {"q": query, "limit_per_type": 50}).body
        events = body.get("events", []) if isinstance(body, dict) else []
        return [e["slug"] for e in events if isinstance(e, dict) and e.get("slug")]

    def discover(self, league: str, away: str, home: str, local_date: date,
                 kickoff: time | None = None, tz: str = "America/New_York",
                 hint_suffixes: Iterable[str] = ("player-props",)) -> tuple[str | None, list[dict]]:
        """Find the game event and every sibling event (props, quarters, ...).

        Returns (base slug, events). Siblings are collected from search
        results whose slug starts with the game slug, so new suffixes show up
        without code changes. `hint_suffixes` are probed directly as a backup
        in case search misses one.
        """
        away, home = canonical_abbr(away), canonical_abbr(home)
        found: dict[str, dict] = {}
        bases = [game_slug(league, away, home, d) for d in slug_dates(local_date, kickoff, tz)]
        base_hit = None
        for base in bases:
            ev = self._try_event(base)
            if ev is not None:
                found[base] = ev
                base_hit = base_hit or base

        prefixes = [base_hit] if base_hit else bases
        queries = [f"{search_terms(away)[0]} {search_terms(home)[0]}",
                   *search_terms(away)[1:], *search_terms(home)[1:]]
        sibling_slugs: set[str] = set()
        for q in queries:
            try:
                for slug in self.search_event_slugs(q):
                    if any(slug == p or slug.startswith(p + "-") for p in prefixes):
                        sibling_slugs.add(slug)
            except ApiError as exc:
                self.warnings.append(f"search {q!r} failed: {exc}")
        for p in prefixes:
            sibling_slugs.update(f"{p}-{s}" for s in hint_suffixes)

        for slug in sorted(sibling_slugs - set(found)):
            ev = self._try_event(slug)
            if ev is not None:
                found[slug] = ev
                base_hit = base_hit or next((p for p in prefixes if slug.startswith(p)), None)
        return base_hit, [found[s] for s in sorted(found)]

    def _try_event(self, slug: str) -> dict | None:
        try:
            return self.event_by_slug(slug)
        except ApiError as exc:
            self.warnings.append(f"event {slug}: {exc}")
            return None

    # CLOB ----------------------------------------------------------------
    def book(self, token_id: str) -> dict:
        return self.http.get(f"{CLOB_BASE}/book", {"token_id": token_id}).body

    def price_history(self, token_id: str, interval: str = "1d", fidelity: int = 5) -> list[tuple[int, float]]:
        """[(unix_ts, price)] for one outcome token. `fidelity` is the bucket in minutes."""
        body = self.http.get(f"{CLOB_BASE}/prices-history",
                             {"market": token_id, "interval": interval, "fidelity": fidelity}).body
        hist = body.get("history", []) if isinstance(body, dict) else []
        out = []
        for pt in hist:
            t, p = pt.get("t"), to_float(pt.get("p"))
            if t is not None and p is not None:
                out.append((int(t), p))
        return sorted(out)

    def attach_books(self, markets: list[PMMarket]) -> int:
        """Replace Gamma quotes with each token's live best ask/bid. Returns the
        number of outcomes priced from the CLOB."""
        jobs = [(m, o) for m in markets for o in m.outcomes if o.token_id]

        def fetch(job):
            m, o = job
            try:
                return job, parse_book(self.book(o.token_id))
            except ApiError as exc:
                return job, exc

        n = 0
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            for (m, o), res in pool.map(fetch, jobs):
                if isinstance(res, Exception):
                    failed.append(f"{m.question} [{o.name}]: {res}")
                    continue
                apply_book(o, res)
                n += 1
        if failed:
            self.warnings.append(f"{len(failed)} outcome(s) have no CLOB book and use Gamma quotes "
                                 f"(first: {failed[0]})")
        for m in markets:
            o0 = m.outcomes[0] if m.outcomes else None
            if o0 is not None and o0.price_source == "clob":
                m.best_bid, m.best_ask = o0.bid, o0.ask
        return n

    def histories(self, markets: list[PMMarket]) -> dict[str, list[tuple[int, float]]]:
        """Price history of outcome 0 for each market, keyed by market_id."""
        jobs = [m for m in markets if m.outcomes and m.outcomes[0].token_id]

        def fetch(m):
            try:
                return m, self.price_history(m.outcomes[0].token_id)
            except ApiError as exc:
                return m, exc

        out = {}
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            for m, res in pool.map(fetch, jobs):
                if isinstance(res, Exception):
                    failed.append(f"{m.question}: {res}")
                else:
                    out[m.market_id] = res
        if failed:
            self.warnings.append(f"price history unavailable for {len(failed)} market(s) "
                                 f"(first: {failed[0]})")
        return out


def event_markets(events: list[dict], game_id: str, home: str, away: str,
                  fetched_at: datetime | None = None) -> list[PMMarket]:
    seen: set[str] = set()
    out = []
    for ev in events:
        for m in ev.get("markets") or []:
            key = str(m.get("id") or m.get("conditionId") or m.get("slug"))
            if key in seen:
                continue
            seen.add(key)
            out.append(normalize_market(m, ev.get("slug", ""), game_id, home, away, fetched_at))
    return out

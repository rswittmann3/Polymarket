"""The Odds API client and normalisation of its odds into BookRows.

Cost reminder: an odds request costs (markets returned) x (regions) credits;
with `bookmakers=` each group of up to 10 books counts as one region.
The events list is free. Remaining credits come back in the
``x-requests-remaining`` header and are reported after every run.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable

from .config import ODDS_BASE, Config
from .fair import american_to_prob
from .httpclient import ApiError, JsonClient, Response
from .models import BookRow, PMMarket, Selection, norm_player
from .teams import full_name, team_from_text

PERIOD_SUFFIX = re.compile(r"_(h1|h2|q1|q2|q3|q4)$")

# NFL player-prop keys listed in The Odds API's market docs. A request with an
# unknown key can be rejected outright, so only these are ever asked for.
KNOWN_PROP_STATS = {
    "pass_attempts", "pass_completions", "pass_interceptions", "pass_tds", "pass_yds",
    "pass_rush_yds", "receptions", "reception_yds", "reception_tds", "rush_attempts",
    "rush_reception_yds", "rush_tds", "rush_yds", "sacks", "anytime_td", "tds_over",
    "reception_longest",
}
NO_ALTERNATE = {"anytime_td", "tds_over", "reception_longest"}


def suffix(period: str) -> str:
    return "" if period == "full" else f"_{period}"


def market_keys_for(sel: Selection, alternates: bool = False) -> list[str]:
    """Odds API market keys that can price this selection."""
    sfx = suffix(sel.period)
    if sel.market_type == "moneyline":
        return [f"h2h{sfx}"]
    if sel.market_type == "spread":
        return [f"alternate_spreads{sfx}"] if alternates else [f"spreads{sfx}"]
    if sel.market_type == "total":
        base = "team_totals" if sel.stat == "team_total" else "totals"
        return [f"alternate_{base}{sfx}"] if alternates else [f"{base}{sfx}"]
    if sel.market_type == "player_prop" and sel.stat in KNOWN_PROP_STATS:
        if alternates:
            return [] if sel.stat in NO_ALTERNATE else [f"player_{sel.stat}_alternate"]
        return [f"player_{sel.stat}"]
    return []


def parse_market_key(key: str) -> tuple[str, str, str] | None:
    """'alternate_spreads_h1' -> ('spread', 'h1', ''); 'player_pass_yds' -> ('player_prop', 'full', 'pass_yds')."""
    period = "full"
    m = PERIOD_SUFFIX.search(key)
    base = key
    if m and not key.startswith("player_"):
        period = m.group(1)
        base = key[: m.start()]
    base = base.removeprefix("alternate_")
    if base == "h2h":
        return "moneyline", period, ""
    if base == "spreads":
        return "spread", period, ""
    if base == "totals":
        return "total", period, ""
    if base == "team_totals":
        return "total", period, "team_total"
    if base.startswith("player_"):
        return "player_prop", period, base.removeprefix("player_").removesuffix("_alternate")
    return None


def normalize_event_odds(body: dict, game_id: str, fetched_at: datetime) -> list[BookRow]:
    rows: list[BookRow] = []
    for bm in body.get("bookmakers") or []:
        book = bm.get("key", "")
        for mk in bm.get("markets") or []:
            key = mk.get("key", "")
            parsed = parse_market_key(key)
            if parsed is None:
                continue
            market_type, period, stat = parsed
            for oc in mk.get("outcomes") or []:
                sel = _outcome_selection(market_type, period, stat, oc)
                price = oc.get("price")
                if sel is None or price is None:
                    continue
                try:
                    prob = american_to_prob(float(price))
                except ValueError:
                    continue
                rows.append(BookRow(game_id=game_id, book=book, market_key=key, selection=sel,
                                    price_american=float(price), prob_implied=prob,
                                    fetched_at=fetched_at))
    return rows


def _outcome_selection(market_type: str, period: str, stat: str, oc: dict) -> Selection | None:
    name = str(oc.get("name", ""))
    point = oc.get("point")
    line = float(point) if point is not None else None
    lname = name.strip().lower()
    if market_type == "moneyline":
        side = "draw" if lname in ("draw", "tie") else team_from_text(name)
        return Selection("moneyline", period, side=side) if side else None
    if market_type == "spread":
        team = team_from_text(name)
        return Selection("spread", period, line=line, side=team) if team and line is not None else None
    if market_type == "total":
        if lname not in ("over", "under") or line is None:
            return None
        player = ""
        if stat == "team_total":
            player = team_from_text(str(oc.get("description", ""))) or ""
            if not player:
                return None
        return Selection("total", period, player=player, stat=stat, line=line, side=lname)
    if market_type == "player_prop":
        player = norm_player(str(oc.get("description", "")))
        if not player or lname not in ("over", "under", "yes", "no"):
            return None
        return Selection("player_prop", period, player=player, stat=stat, line=line, side=lname)
    return None


class OddsApiClient:
    def __init__(self, http: JsonClient, cfg: Config, sport: str):
        self.http = http
        self.cfg = cfg
        self.sport = sport
        self.last_headers: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []   # (what, credits used)

    def _get(self, path: str, params: dict) -> Response:
        if not self.http.offline and not self.cfg.odds_api_key:
            raise ApiError("ODDS_API_KEY is not set (put it in .env)")
        resp = self.http.get(f"{ODDS_BASE}{path}", {**params, "apiKey": self.cfg.odds_api_key or ""})
        if resp.headers:
            self.last_headers = resp.headers
        self.calls.append((path, resp.headers.get("x-requests-last", "?")))
        return resp

    def find_event(self, home: str, away: str, local_date: date) -> dict:
        resp = self._get(f"/sports/{self.sport}/events", {"dateFormat": "iso"})
        events = resp.body if isinstance(resp.body, list) else []
        want_home, want_away = full_name(home), full_name(away)
        cands = [e for e in events if e.get("home_team") == want_home and e.get("away_team") == want_away]
        if not cands:
            raise ApiError(f"Odds API has no event {want_away} @ {want_home}")

        def distance(e):
            try:
                d = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).date()
            except (KeyError, ValueError):
                return 99
            return abs((d - local_date).days)
        best = min(cands, key=distance)
        if distance(best) > 1:
            raise ApiError(f"Odds API event {want_away} @ {want_home} is on {best.get('commence_time')}, "
                           f"not around {local_date}")
        return best

    def event_odds(self, event_id: str, markets: Iterable[str]) -> Response:
        where = {"bookmakers": self.cfg.bookmakers} if self.cfg.bookmakers else {"regions": self.cfg.regions}
        return self._get(f"/sports/{self.sport}/events/{event_id}/odds", {
            **where,
            "markets": ",".join(sorted(set(markets))),
            "oddsFormat": "american",
            "dateFormat": "iso",
        })

    @property
    def remaining(self) -> str | None:
        return self.last_headers.get("x-requests-remaining")


def needed_market_keys(markets: list[PMMarket]) -> set[str]:
    keys: set[str] = {"h2h", "spreads", "totals"}
    for m in markets:
        for o in m.outcomes:
            if o.selection is not None:
                keys.update(market_keys_for(o.selection))
    return keys


def missing_lines(markets: list[PMMarket], rows: list[BookRow]) -> set[str]:
    """Alternate-market keys worth requesting: PM selections with a line that
    no book quoted in the main markets."""
    have = {(r.selection.group_key(), r.selection.line, r.selection.side) for r in rows}
    out: set[str] = set()
    for m in markets:
        for o in m.outcomes:
            s = o.selection
            if s is None or s.line is None:
                continue
            if (s.group_key(), s.line, s.side) not in have:
                out.update(market_keys_for(s, alternates=True))
    return out

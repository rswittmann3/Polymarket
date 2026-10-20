"""Upcoming games with their current and opening lines, from free sources.

NBA / NHL: ESPN's scoreboard lists the day's games; ESPN's odds endpoint gives DraftKings' current and
opening line for each. NFL: the nflverse schedule (current lines only). No Odds API credits are used.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable

import httpx

from .history import _nfl_frame, _nflverse, _num
from .teams import league as _league

SPORT = {"nba": "basketball/nba", "nhl": "hockey/nhl"}
UA = {"User-Agent": "Mozilla/5.0 (research; pm_edges)"}
PROVIDERS = ["DraftKings", "ESPN BET", "Caesars Sportsbook", "Caesars", "MGM", "Bet365"]


def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = httpx.get(url, params=params, headers=UA, timeout=30)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def _american(node) -> float | None:
    return _num((node or {}).get("american")) if isinstance(node, dict) else None


def parse_odds(items: list[dict]) -> dict | None:
    """One book's current and opening lines, home side. Prefers DraftKings."""
    usable = [i for i in items if i.get("overUnder") is not None and "live" not in str((i.get("provider") or {}).get("name", "")).lower()]
    if not usable:
        return None
    pick = next((i for p in PROVIDERS for i in usable if (i.get("provider") or {}).get("name") == p), usable[0])
    home, away = pick.get("homeTeamOdds") or {}, pick.get("awayTeamOdds") or {}
    spread = pick.get("spread")
    if spread is not None:
        spread = -abs(float(spread)) if home.get("favorite") else abs(float(spread))
    cur = _american((home.get("current") or {}).get("pointSpread"))
    out = {"provider": (pick.get("provider") or {}).get("name"), "total": float(pick["overUnder"]),
           "total_open": _american((pick.get("open") or {}).get("total")),
           "home_spread": cur if cur is not None else spread,
           "home_spread_open": _american((home.get("open") or {}).get("pointSpread")),
           "over_odds": _num(pick.get("overOdds")), "under_odds": _num(pick.get("underOdds")),
           "home_ml": _num(home.get("moneyLine")), "away_ml": _num(away.get("moneyLine"))}
    return out


def espn_slate(lg: str, start: date, days: int = 2, get: Callable = _get) -> list[dict]:
    teams = _league(lg)
    out = []
    for k in range(days):
        day = start + timedelta(days=k)
        board = get(f"https://site.api.espn.com/apis/site/v2/sports/{SPORT[lg]}/scoreboard", {"dates": f"{day:%Y%m%d}"})
        for e in (board or {}).get("events", []):
            comp = e["competitions"][0]
            if comp["status"]["type"]["name"] != "STATUS_SCHEDULED":
                continue
            side = {c["homeAway"]: c["team"].get("abbreviation") for c in comp["competitors"]}
            try:
                home, away = teams.canonical_abbr(side["home"]), teams.canonical_abbr(side["away"])
            except KeyError:
                continue
            when = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(timezone.utc)
            sport = SPORT[lg].replace("/", "/leagues/")
            odds = get(f"https://sports.core.api.espn.com/v2/sports/{sport}/events/{e['id']}/competitions/{e['id']}/odds")
            line = parse_odds((odds or {}).get("items", []))
            out.append({"league": lg, "espn_id": e["id"], "away": away, "home": home, "kickoff": when.isoformat(),
                        "date": (when - timedelta(hours=8)).date().isoformat(), **(line or {})})
    return out


def nfl_slate(start: date, days: int = 8) -> list[dict]:
    g = _nfl_frame(_nflverse())
    g = g[g["home_pts"].isna() & (g["date"] >= str(start)) & (g["date"] < str(start + timedelta(days=days)))]
    out = []
    for r in g.itertuples():
        row = {"league": "nfl", "away": r.away, "home": r.home, "date": r.date.date().isoformat(), "provider": "nflverse",
               "total": None if r.total_close != r.total_close else float(r.total_close),
               "home_spread": None if r.home_spread_close != r.home_spread_close else float(r.home_spread_close),
               "home_ml": None if r.home_ml != r.home_ml else float(r.home_ml), "away_ml": None if r.away_ml != r.away_ml else float(r.away_ml),
               "over_odds": None if r.over_odds != r.over_odds else float(r.over_odds),
               "under_odds": None if r.under_odds != r.under_odds else float(r.under_odds),
               "week": int(r.week), "weekday": r.weekday, "hour": None if r.hour != r.hour else float(r.hour),
               "roof": r.roof, "div_game": None if r.div_game != r.div_game else int(r.div_game)}
        out.append(row)
    return out


def slate(lg: str, start: date, days: int = 2, get: Callable = _get) -> list[dict]:
    return nfl_slate(start, max(days, 8)) if lg == "nfl" else espn_slate(lg, start, days, get)

"""Injury reports and probable NHL goalies from ESPN (free, current only: there is no free history).

`report(lg)` is every team's current injury list. `player_value(lg, athlete_id)` reads the athlete's latest
season line (NBA minutes and points per game; NHL points and goalie save percentage) so a report can say whether
the absent player matters. `game_context(...)` joins both onto an upcoming game for the dashboard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import httpx

from .teams import league as _league

SPORT = {"nba": "basketball/nba", "nhl": "hockey/nhl", "nfl": "football/nfl"}
UA = {"User-Agent": "Mozilla/5.0 (research; pm_edges)"}
CACHE = Path("data/lag/players.json")
ACTIVE_STATUSES = {"Out", "Doubtful", "Questionable", "Day-To-Day", "Injured Reserve", "Suspension"}


def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = httpx.get(url, params=params, headers=UA, timeout=30)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def report(lg: str, get: Callable = _get) -> list[dict]:
    """Current injuries, one row per player."""
    teams = _league(lg)
    body = get(f"https://site.api.espn.com/apis/site/v2/sports/{SPORT[lg]}/injuries") or {}
    rows = []
    for t in body.get("injuries", []):
        abbr = teams.team_from_text(t.get("displayName", ""))
        if abbr is None:
            continue
        for i in t.get("injuries", []):
            a = i.get("athlete") or {}
            href = next((l["href"] for l in a.get("links", []) if "/id/" in l.get("href", "")), "")
            m = re.search(r"/id/(\d+)", href)
            rows.append({"league": lg, "team": abbr, "athlete_id": m.group(1) if m else None, "name": a.get("displayName"),
                         "pos": (a.get("position") or {}).get("abbreviation"), "status": i.get("status"), "date": i.get("date"),
                         "note": i.get("shortComment")})
    return rows


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def player_value(lg: str, athlete_id: str, get: Callable = _get, cache: dict | None = None) -> dict:
    """Latest-season per-game line from ESPN's athlete overview, cached on disk."""
    store = cache if cache is not None else _load()
    key = f"{lg}:{athlete_id}"
    if key in store:
        return store[key]
    body = get(f"https://site.web.api.espn.com/apis/common/v3/sports/{SPORT[lg]}/athletes/{athlete_id}/overview") or {}
    stats = body.get("statistics") or {}
    names = stats.get("names", [])
    splits = {sp.get("displayName"): dict(zip(names, sp.get("stats", []))) for sp in stats.get("splits", [])}
    season, career = splits.get("Regular Season") or {}, splits.get("Career") or {}
    # early in a season a hockey player has a handful of games: use the career rate until there is a sample
    row = season if (_num(season.get("games") or season.get("gamesPlayed")) or 0) >= 20 or not career else career
    games = _num(row.get("games") or row.get("gamesPlayed"))
    if lg == "nhl":
        points = (_num(row.get("points")) or 0) / games if games else None
        minutes = _toi(row.get("timeOnIcePerGame"))
    else:
        points, minutes = _num(row.get("avgPoints")), _num(row.get("avgMinutes"))
    out = {"games": games, "minutes": minutes, "points": points, "save_pct": _num(row.get("savePct")) or _num(row.get("savePercentage")),
           "gaa": _num(row.get("avgGoalsAgainst")), "goals": _num(row.get("goals")), "assists": _num(row.get("assists"))}
    store[key] = out
    if cache is None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(store))
    return out


def _toi(x) -> float | None:
    m = re.fullmatch(r"(\d+):(\d\d)", str(x or ""))
    return int(m.group(1)) + int(m.group(2)) / 60 if m else None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_key(lg: str, status: str, pos: str | None, val: dict) -> bool:
    """A rough 'does this absence matter' flag: rotation players in basketball, goalies and top scorers in hockey."""
    if status not in ("Out", "Doubtful", "Questionable", "Day-To-Day"):
        return False
    if lg == "nba":
        return (val.get("minutes") or 0) >= 24 or (val.get("points") or 0) >= 15
    if lg == "nhl":
        return pos == "G" or (val.get("points") or 0) >= 0.75 or (val.get("minutes") or 0) >= 19
    return pos in ("QB",)


def probable_goalies(day: str, get: Callable = _get) -> dict[tuple[str, str], dict]:
    """{(team abbr, espn event id): goalie} for NHL games on a yyyymmdd date."""
    teams = _league("nhl")
    body = get("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard", {"dates": day}) or {}
    out = {}
    for e in body.get("events", []):
        for c in e["competitions"][0]["competitors"]:
            abbr = teams.team_from_text(c["team"].get("abbreviation", "")) or teams.team_from_text(c["team"].get("displayName", ""))
            for p in c.get("probables", []) or []:
                if p.get("name") == "probableStartingGoalie":
                    a = p.get("athlete") or {}
                    out[(abbr, e["id"])] = {"name": a.get("displayName"), "athlete_id": a.get("id"),
                                           "record": p.get("summary") or p.get("displayValue")}
    return out


def game_context(lg: str, away: str, home: str, rows: list[dict], values: dict[str, dict] | None = None) -> list[dict]:
    """Injury rows for a game's two teams, active statuses only, key players first."""
    out = []
    for r in rows:
        if r["team"] not in (away, home) or r["status"] not in ACTIVE_STATUSES:
            continue
        val = (values or {}).get(r["athlete_id"], {})
        out.append({**r, "side": "home" if r["team"] == home else "away", "key": is_key(lg, r["status"], r["pos"], val),
                    "minutes": val.get("minutes"), "points": val.get("points")})
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    return sorted(out, key=lambda r: (not r["key"], order.get(r["status"], 3), -(r.get("minutes") or 0)))

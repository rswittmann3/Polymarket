"""Record how sportsbook lines move before tip-off, so Polymarket's reaction can be measured afterwards.

Three sources are polled in a loop and appended to data/lag/timeline.jsonl:
  * ESPN's DraftKings current and opening line for every scheduled NBA/NHL game (free, every --every minutes)
  * ESPN injury reports (all teams, written when a player's status changes) and NHL probable starting goalies
  * Polymarket order books (best bid/ask and the cost of a $50 fill) for the two busiest spread and total markets per NBA game
  * Pinnacle's current spread and total for NBA from The Odds API (2 credits per poll, every --pinnacle-every minutes,
    stopped when the remaining credits fall below --keep)

Polymarket's own price history is pulled after the games from the CLOB, so it needs no polling here.

    nohup python tools/line_poller.py --hours 36 > data/lag/poller.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from pm_edges import injuries as I
from pm_edges import slate as S
from pm_edges.polymarket import parse_book, walk_book
from pm_edges.teams import league

OUT = Path("data/lag/timeline.jsonl")
NBA = league("nba")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def poll_espn(lg: str, f) -> int:
    today = datetime.now(timezone(timedelta(hours=-5))).date()
    n = 0
    for g in S.espn_slate(lg, today, 3):
        if not g.get("total"):
            continue
        f.write(json.dumps({"ts": now(), "src": "espn_dk", **g}) + "\n")
        n += 1
    return n


def poll_news(seen: set, f, first: bool) -> int:
    """Injury status changes (NBA, NHL, NFL) and NHL probable-goalie changes, each written once when first seen."""
    n = 0
    for lg in ("nba", "nhl", "nfl"):
        for r in I.report(lg):
            key = (lg, r["athlete_id"], r["status"], r["date"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({"ts": now(), "src": "injury", "baseline": first, **r}) + "\n")
            n += 1
    today = datetime.now(timezone(timedelta(hours=-5)))
    for k in range(3):
        day = (today + timedelta(days=k)).strftime("%Y%m%d")
        for (team, eid), g in I.probable_goalies(day).items():
            key = ("goalie", team, eid, g["name"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({"ts": now(), "src": "goalie", "baseline": first, "team": team, "espn_id": eid, **g}) + "\n")
            n += 1
    return n


_events: dict[tuple, tuple[float, dict | None]] = {}


def _get(url: str, params: dict) -> object | None:
    try:
        r = httpx.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 (research; pm_edges)"}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def pm_event(away: str, home: str, tip: datetime) -> dict | None:
    """The Polymarket NBA event for a game, refreshed hourly so new lines appear."""
    key = (away, home, tip.date())
    cached = _events.get(key)
    if cached and time.time() - cached[0] < 3600:
        return cached[1]
    local = tip.astimezone(timezone(timedelta(hours=-4)))
    ev = None
    for h in {home.lower(), {"GSW": "gs", "NOP": "no", "SAS": "sa", "NYK": "ny"}.get(home, home).lower()}:
        for day in (local.date(), local.date() + timedelta(days=1)):
            d = _get("https://gamma-api.polymarket.com/events", {"slug": f"nba-{away.lower()}-{h}-{day:%Y-%m-%d}"})
            if d:
                ev = d[0]
                break
        if ev:
            break
    _events[key] = (time.time(), ev)
    return ev


def poll_pm_books(f, tip_by_game: dict, dollars: float = 50.0) -> int:
    """Best bid/ask and the cost of a $50 fill for the two highest-volume spread and total markets of each NBA game."""
    n = 0
    for (away, home), tip in tip_by_game.items():
        ev = pm_event(away, home, tip)
        if ev is None:
            continue
        for kind in ("spreads", "totals"):
            ms = sorted((m for m in ev["markets"] if m.get("sportsMarketType") == kind and m.get("line") is not None
                         and m.get("acceptingOrders", True) is not False), key=lambda m: -(m.get("volumeNum") or 0))[:2]
            for m in ms:
                token = json.loads(m["clobTokenIds"])[0]
                body = _get("https://clob.polymarket.com/book", {"token_id": token})
                if not isinstance(body, dict):
                    continue
                b = parse_book(body)
                buy, sell = walk_book(b["asks"], dollars), walk_book(b["bids"], dollars)
                f.write(json.dumps({"ts": now(), "src": "pm_book", "away": away, "home": home, "kind": kind, "line": m["line"],
                                    "question": m["question"], "market_id": m["id"], "volume": m.get("volumeNum"),
                                    "bid": b["bid"], "ask": b["ask"], "bid_size": b["bid_size"], "ask_size": b["ask_size"],
                                    "buy_avg": buy["avg"], "buy_filled": buy["filled"], "sell_avg": sell["avg"], "sell_filled": sell["filled"]}) + "\n")
                n += 1
    return n


def poll_pinnacle(key: str, f) -> str | None:
    r = httpx.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                  params={"apiKey": key, "markets": "spreads,totals", "bookmakers": "pinnacle", "oddsFormat": "american"}, timeout=45)
    if r.status_code != 200:
        print("pinnacle poll failed", r.status_code, r.text[:120], flush=True)
        return None
    ts = now()
    for e in r.json():
        book = next((b for b in e["bookmakers"] if b["key"] == "pinnacle"), None)
        if not book:
            continue
        row = {"ts": ts, "src": "pinnacle", "commence": e["commence_time"], "home": NBA.team_from_text(e["home_team"]),
               "away": NBA.team_from_text(e["away_team"])}
        for m in book["markets"]:
            row[m["key"]] = [{"name": o["name"], "price": o["price"], "point": o.get("point")} for o in m["outcomes"]]
        f.write(json.dumps(row) + "\n")
    return r.headers.get("x-requests-remaining")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=36)
    ap.add_argument("--every", type=float, default=5, help="minutes between ESPN polls")
    ap.add_argument("--pinnacle-every", type=float, default=30)
    ap.add_argument("--keep", type=int, default=150, help="stop Pinnacle polling below this many credits")
    a = ap.parse_args()
    key = os.environ.get("ODDS_API_KEY")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    end = time.time() + a.hours * 3600
    next_pin, remaining, seen, first = 0.0, None, set(), True
    while time.time() < end:
        t0 = time.time()
        with OUT.open("a") as f:
            counts = {lg: poll_espn(lg, f) for lg in ("nba", "nhl")}
            tips = {(g["away"], g["home"]): datetime.fromisoformat(g["kickoff"]) for g in S.espn_slate("nba", datetime.now(timezone(timedelta(hours=-5))).date(), 3)
                    if g.get("kickoff")}
            counts["pm_books"] = poll_pm_books(f, tips)
            counts["news"] = poll_news(seen, f, first)
            first = False
            if key and t0 >= next_pin and (remaining is None or int(remaining) > a.keep):
                remaining = poll_pinnacle(key, f) or remaining
                next_pin = t0 + a.pinnacle_every * 60
        print(now(), counts, "credits", remaining, flush=True)
        time.sleep(max(5, a.every * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()

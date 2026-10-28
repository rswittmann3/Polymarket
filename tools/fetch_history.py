"""Download NBA game results and closing lines into data/history/ (raw JSON cached on disk).

Sources:
  * Sportsbook Reviews Online season pages: NBA 2007-08 .. 2022-23, NHL 2007-08 .. 2022-23 (not 2020-21).
  * ESPN's public scoreboard + odds endpoints: NBA from 2022-23, NHL from 2020-21 (scoreboards for
    every NHL season, because they label shootout games, which sportsbooks leave out of totals).
  * nflverse's games.csv: NFL scores, closing lines, weather, rest, roof, 1999 on.

    python tools/fetch_history.py espn --league nba --seasons 2023 2024 2025 2026   # ESPN "season" is the end year
    python tools/fetch_history.py espn --league nhl --seasons 2008 ... 2026 --odds-from 2021
    python tools/fetch_history.py sbr [--league nhl]
    python tools/fetch_history.py nfl

Everything fetched is cached, so re-running only fetches what is missing.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import httpx

BASE = Path("data/history/raw")
SPORT = {"nba": "basketball/nba", "nhl": "hockey/nhl"}
NFLVERSE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def root(league: str) -> Path:
    """NBA keeps its original folder; other leagues get their own."""
    return BASE if league == "nba" else BASE / league


def score_url(league: str) -> str:
    return f"https://site.api.espn.com/apis/site/v2/sports/{SPORT[league]}/scoreboard"


def odds_url(league: str) -> str:
    return "https://sports.core.api.espn.com/v2/sports/{sport}/events/{{id}}/competitions/{{id}}/odds".format(
        sport=SPORT[league].replace("/", "/leagues/"))


SBR = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/{lg}-odds-{a}-{b}"
UA = {"User-Agent": "Mozilla/5.0 (research; pm_edges)"}


def get(client: httpx.Client, url: str, params: dict | None = None, tries: int = 4):
    for i in range(tries):
        try:
            r = client.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 400):
                return None
        except httpx.TransportError:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def season_days(end_year: int, league: str = "nba"):
    # 2019-20 was finished in bubbles that ran into October 2020; the 2020-21 NHL season ended in July
    last = date(end_year, 6, 30)
    if end_year == 2020:
        last = date(2020, 10, 15)
    elif league == "nhl" and end_year == 2021:
        last = date(2021, 7, 31)
    d = date(end_year - 1, 10, 1)
    while d <= last:
        yield d
        d += timedelta(days=1)


def fetch_espn(league: str, seasons: list[int], workers: int, odds_from: int = 0) -> None:
    ROOT = root(league)
    (ROOT / "espn_scoreboard").mkdir(parents=True, exist_ok=True)
    (ROOT / "espn_odds").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(http2=False)

    def day(d: date):
        path = ROOT / "espn_scoreboard" / f"{d:%Y%m%d}.json"
        if path.exists() and d < date.today() - timedelta(days=3):         # recent days were cached before their games finished
            return json.loads(path.read_text())
        r = get(client, score_url(league), {"dates": f"{d:%Y%m%d}", "limit": 100})
        if r is None:
            return None
        body = r.json()
        slim = {"date": f"{d:%Y%m%d}", "events": [{
            "id": e["id"], "date": e["date"], "season": e.get("season"), "name": e.get("name"),
            "status": e["competitions"][0]["status"]["type"]["name"],
            "detail": e["competitions"][0]["status"]["type"].get("detail"),
            "neutral": e["competitions"][0].get("neutralSite"),
            "notes": [n.get("headline") for n in e["competitions"][0].get("notes", [])],
            "teams": [{"homeAway": c["homeAway"], "abbr": c["team"].get("abbreviation") or c["team"].get("shortDisplayName"),
                       "name": c["team"].get("displayName"), "score": c.get("score"), "winner": c.get("winner")}
                      for c in e["competitions"][0]["competitors"]],
        } for e in body.get("events", [])]}
        path.write_text(json.dumps(slim))
        return slim

    days = [d for y in seasons for d in season_days(y, league) if d <= date.today()]
    print(f"scoreboard: {len(days)} days", flush=True)
    events = []
    with ThreadPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(day, days)):
            if res:
                events += [e for e in res["events"] if e["status"] == "STATUS_FINAL"]
            if i % 100 == 0:
                print(f"  days {i}/{len(days)}  final games so far {len(events)}", flush=True)
    print(f"{len(events)} final games; fetching odds", flush=True)

    def odds(e):
        path = ROOT / "espn_odds" / f"{e['id']}.json"
        if path.exists():
            return True
        r = get(client, odds_url(league).format(id=e["id"]))
        if r is None:
            path.write_text(json.dumps({"items": []}))
            return False
        body = r.json()
        slim = {"items": [{
            "provider": it.get("provider", {}).get("name"),
            "spread": it.get("spread"), "overUnder": it.get("overUnder"),
            "overOdds": it.get("overOdds"), "underOdds": it.get("underOdds"),
            "open_total": (it.get("open") or {}).get("total", {}).get("american"),
            "close_total": (it.get("close") or {}).get("total", {}).get("american"),
            "home_open_spread": ((it.get("homeTeamOdds") or {}).get("open") or {}).get("pointSpread", {}).get("american"),
            "home_close_spread": ((it.get("homeTeamOdds") or {}).get("close") or {}).get("pointSpread", {}).get("american"),
            "home_fav": (it.get("homeTeamOdds") or {}).get("favorite"),
            "home_ml": (it.get("homeTeamOdds") or {}).get("moneyLine"),
            "away_ml": (it.get("awayTeamOdds") or {}).get("moneyLine"),
        } for it in body.get("items", [])]}
        path.write_text(json.dumps(slim))
        return True

    ok = 0
    if odds_from:
        events = [e for e in events if (e.get("season") or {}).get("year", 0) >= odds_from]
        print(f"{len(events)} games from season {odds_from} on get odds", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(odds, events)):
            ok += bool(res)
            if i % 250 == 0:
                print(f"  odds {i}/{len(events)}  with data {ok}", flush=True)
    print("done", flush=True)


def fetch_sbr(league: str = "nba") -> None:
    folder = root(league) / ("sbr" if league == "nba" else "sbr")
    folder.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(follow_redirects=True)
    for y in range(2007, 2023):
        a, b = f"{y}", f"{(y + 1) % 100:02d}"
        path = folder / f"{a}-{b}.htm"
        if path.exists():
            continue
        r = get(client, SBR.format(lg=league, a=a, b=b))
        if r is None:
            print("missing", a, b)
            continue
        path.write_text(r.text)
        print("saved", path, len(r.text), flush=True)
        time.sleep(1)


def fetch_nfl() -> None:
    folder = root("nfl")
    folder.mkdir(parents=True, exist_ok=True)
    r = get(httpx.Client(follow_redirects=True), NFLVERSE)
    if r is None:
        raise SystemExit("could not download nflverse games.csv")
    (folder / "games.csv").write_text(r.text)
    print("saved", folder / "games.csv", len(r.text), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("espn")
    e.add_argument("--league", choices=["nba", "nhl"], default="nba")
    e.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--odds-from", type=int, default=0, help="only fetch odds for seasons (end year) from this one on")
    s_ = sub.add_parser("sbr")
    s_.add_argument("--league", choices=["nba", "nhl"], default="nba")
    sub.add_parser("nfl")
    args = ap.parse_args()
    if args.cmd == "espn":
        fetch_espn(args.league, args.seasons, args.workers, args.odds_from)
    elif args.cmd == "sbr":
        fetch_sbr(args.league)
    else:
        fetch_nfl()

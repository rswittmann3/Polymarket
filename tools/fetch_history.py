"""Download NBA game results and closing lines into data/history/ (raw JSON cached on disk).

Two sources:
  * Sportsbook Reviews Online season pages (2007-08 .. 2022-23): scores, open/close spread and total.
  * ESPN's public scoreboard + odds endpoints (2022-23 .. now): scores, season type, pre-game open/close.

    python tools/fetch_history.py espn --seasons 2023 2024 2025 2026     # ESPN "season" is the end year
    python tools/fetch_history.py sbr

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

ROOT = Path("data/history/raw")
SCORE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{id}/competitions/{id}/odds"
SBR = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{a}-{b}"
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


def season_days(end_year: int):
    # 2019-20 was finished in the Orlando bubble, which ran into October 2020
    d, last = date(end_year - 1, 10, 1), (date(2020, 10, 15) if end_year == 2020 else date(end_year, 6, 30))
    while d <= last:
        yield d
        d += timedelta(days=1)


def fetch_espn(seasons: list[int], workers: int) -> None:
    (ROOT / "espn_scoreboard").mkdir(parents=True, exist_ok=True)
    (ROOT / "espn_odds").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(http2=False)

    def day(d: date):
        path = ROOT / "espn_scoreboard" / f"{d:%Y%m%d}.json"
        if path.exists():
            return json.loads(path.read_text())
        r = get(client, SCORE, {"dates": f"{d:%Y%m%d}", "limit": 100})
        if r is None:
            return None
        body = r.json()
        slim = {"date": f"{d:%Y%m%d}", "events": [{
            "id": e["id"], "date": e["date"], "season": e.get("season"), "name": e.get("name"),
            "status": e["competitions"][0]["status"]["type"]["name"],
            "neutral": e["competitions"][0].get("neutralSite"),
            "notes": [n.get("headline") for n in e["competitions"][0].get("notes", [])],
            "teams": [{"homeAway": c["homeAway"], "abbr": c["team"].get("abbreviation") or c["team"].get("shortDisplayName"),
                       "name": c["team"].get("displayName"), "score": c.get("score"), "winner": c.get("winner")}
                      for c in e["competitions"][0]["competitors"]],
        } for e in body.get("events", [])]}
        path.write_text(json.dumps(slim))
        return slim

    days = [d for y in seasons for d in season_days(y) if d <= date.today()]
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
        r = get(client, ODDS.format(id=e["id"]))
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
    with ThreadPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(odds, events)):
            ok += bool(res)
            if i % 250 == 0:
                print(f"  odds {i}/{len(events)}  with data {ok}", flush=True)
    print("done", flush=True)


def fetch_sbr() -> None:
    (ROOT / "sbr").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(follow_redirects=True)
    for y in range(2007, 2023):
        a, b = f"{y}", f"{(y + 1) % 100:02d}"
        path = ROOT / "sbr" / f"{a}-{b}.htm"
        if path.exists():
            continue
        r = get(client, SBR.format(a=a, b=b))
        if r is None:
            print("missing", a, b)
            continue
        path.write_text(r.text)
        print("saved", path, len(r.text), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("espn")
    e.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    e.add_argument("--workers", type=int, default=6)
    sub.add_parser("sbr")
    args = ap.parse_args()
    fetch_espn(args.seasons, args.workers) if args.cmd == "espn" else fetch_sbr()

"""Pull ESPN box scores (team totals and every player's line) for finished NBA games, 2019-20 onward.

Game ids come from the scoreboard days already cached by fetch_history (data/history/raw/espn_scoreboard/*.json). Each game is parsed
into a small JSON file under data/history/box/raw/<id>.json, so an interrupted run resumes where it stopped. `--combine` (also run at
the end) writes data/history/box/team_box.pkl and player_box.pkl.

    python tools/fetch_box.py                 # fetch everything missing, then combine
    python tools/fetch_box.py --limit 50      # try a handful first
    python tools/fetch_box.py --combine       # only rebuild the two tables from the per-game files
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

SCORES = Path("data/history/raw/espn_scoreboard")
OUT = Path("data/history/box")
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def game_ids() -> list[dict]:
    out = {}
    for f in sorted(glob.glob(str(SCORES / "*.json"))):
        for e in json.load(open(f)).get("events", []):
            if e.get("status") != "STATUS_FINAL" or (e.get("season") or {}).get("type") not in (2, 3, 4, 5):
                continue
            out[e["id"]] = {"id": e["id"], "date": e["date"][:10], "season": e["season"]["year"], "stype": e["season"]["type"],
                            "teams": e.get("teams", [])}
    return list(out.values())


def gap_ids(start: str = "2020-10-16", end: str = "2022-09-30") -> list[dict]:
    """Scoreboard days missing from the fetch_history cache (the 2020-21 and 2021-22 seasons), read straight from ESPN and cached."""
    cache = OUT / "gap_ids.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from datetime import date, timedelta
    d, stop, days = date.fromisoformat(start), date.fromisoformat(end), []
    while d <= stop:
        days.append(d)
        d += timedelta(days=1)

    def one(day):
        for _ in range(3):
            try:
                r = httpx.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", params={"dates": f"{day:%Y%m%d}", "limit": 100}, timeout=40)
                if r.status_code == 200:
                    return r.json().get("events", [])
            except httpx.HTTPError:
                time.sleep(1)
        return []
    out = {}
    with ThreadPoolExecutor(10) as ex:
        for evs in ex.map(one, days):
            for e in evs:
                comp = e["competitions"][0]
                if comp["status"]["type"]["name"] != "STATUS_FINAL" or e["season"]["type"] not in (2, 3, 4, 5):
                    continue
                out[e["id"]] = {"id": e["id"], "date": e["date"][:10], "season": e["season"]["year"], "stype": e["season"]["type"],
                                "teams": [{"homeAway": c["homeAway"], "abbr": c["team"]["abbreviation"]} for c in comp["competitors"]]}
    cache.write_text(json.dumps(list(out.values())))
    return list(out.values())


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(g: dict, s: dict) -> dict | None:
    bs = s.get("boxscore") or {}
    if len(bs.get("teams", [])) != 2 or not bs.get("players"):
        return None
    home_abbr = next((t["abbr"] for t in g["teams"] if t.get("homeAway") == "home"), None)
    teams, players = [], []
    for t in bs["teams"]:
        st = {x["name"]: x["displayValue"] for x in t["statistics"]}
        fgm, fga = (st.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0").split("-")
        tpm, tpa = (st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or "0-0").split("-")
        ftm, fta = (st.get("freeThrowsMade-freeThrowsAttempted") or "0-0").split("-")
        teams.append({"team": t["team"]["abbreviation"], "fgm": _num(fgm), "fga": _num(fga), "tpm": _num(tpm), "tpa": _num(tpa), "ftm": _num(ftm),
                      "fta": _num(fta), "oreb": _num(st.get("offensiveRebounds")), "dreb": _num(st.get("defensiveRebounds")), "ast": _num(st.get("assists")),
                      "stl": _num(st.get("steals")), "blk": _num(st.get("blocks")), "tov": _num(st.get("totalTurnovers") or st.get("turnovers")),
                      "pf": _num(st.get("fouls")), "paint": _num(st.get("pointsInPaint")), "fast": _num(st.get("fastBreakPoints")),
                      "off_tov_pts": _num(st.get("turnoverPoints")), "home": t["team"]["abbreviation"] == home_abbr})
    for tp in bs["players"]:
        abbr = tp["team"]["abbreviation"]
        blk = tp["statistics"][0] if tp.get("statistics") else None
        if not blk:
            continue
        labels = blk["labels"]
        for a in blk["athletes"]:
            row = {"team": abbr, "player_id": a["athlete"].get("id"), "player": a["athlete"].get("displayName") or a["athlete"].get("shortName"), "starter": bool(a.get("starter")),
                   "dnp": bool(a.get("didNotPlay")), "reason": a.get("reason") or ""}
            if a.get("stats"):
                d = dict(zip(labels, a["stats"]))
                row["min"] = _num(d.get("MIN"))
                for k, lab in (("pts", "PTS"), ("reb", "REB"), ("ast", "AST"), ("tov", "TO"), ("stl", "STL"), ("blk", "BLK"), ("oreb", "OREB"),
                               ("dreb", "DREB"), ("pf", "PF")):
                    row[k] = _num(d.get(lab))
                row["pm"] = _num((d.get("+/-") or "").replace("+", "")) if d.get("+/-") else None
                for k, lab in (("fg", "FG"), ("tp", "3PT"), ("ft", "FT")):
                    m, _, att = (d.get(lab) or "0-0").partition("-")
                    row[k + "m"], row[k + "a"] = _num(m), _num(att)
            players.append(row)
    officials = [o.get("displayName") or o.get("fullName") for o in (s.get("gameInfo", {}).get("officials") or [])]
    return {"id": g["id"], "date": g["date"], "season": g["season"], "stype": g["stype"], "teams": teams, "players": players, "officials": officials}


def fetch_one(g: dict, client: httpx.Client) -> bool:
    path = OUT / "raw" / f"{g['id']}.json"
    if path.exists():
        return True
    for attempt in range(4):
        try:
            r = client.get(URL, params={"event": g["id"]}, timeout=40)
            if r.status_code == 200:
                try:
                    p = parse(g, r.json())
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    print(f"skip {g['id']}: {type(e).__name__} {e}", flush=True)
                    return False
                if p:
                    path.write_text(json.dumps(p))
                    return True
                return False
        except httpx.HTTPError:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


def combine() -> None:
    teams, players, games = [], [], []
    for f in glob.glob(str(OUT / "raw" / "*.json")):
        p = json.load(open(f))
        games.append({"game_id": p["id"], "date": p["date"], "season": p["season"], "stype": p["stype"], "officials": "|".join(x or "" for x in p["officials"])})
        for t in p["teams"]:
            teams.append({"game_id": p["id"], "date": p["date"], "season": p["season"], "stype": p["stype"], **t})
        for q in p["players"]:
            players.append({"game_id": p["id"], "date": p["date"], "season": p["season"], "stype": p["stype"], **q})
    pd.DataFrame(teams).to_pickle(OUT / "team_box.pkl")
    pd.DataFrame(players).to_pickle(OUT / "player_box.pkl")
    pd.DataFrame(games).to_pickle(OUT / "game_info.pkl")
    print(f"combined {len(games)} games: {len(teams)} team rows, {len(players)} player rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if a.combine:
        combine()
        return
    have = {g["id"] for g in game_ids()}
    games = [g for g in game_ids() + [x for x in gap_ids() if x["id"] not in have] if not (OUT / "raw" / f"{g['id']}.json").exists()]
    if a.limit:
        games = games[: a.limit]
    print(f"{len(games)} games to fetch", flush=True)
    done = 0
    t0 = time.time()
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (research; pm_edges)"}) as client, ThreadPoolExecutor(a.workers) as ex:
        for ok in ex.map(lambda g: fetch_one(g, client), games):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(games)} ({time.time() - t0:.0f}s)", flush=True)
    combine()


if __name__ == "__main__":
    main()

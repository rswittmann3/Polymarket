"""Historical NBA results with closing lines, from two cached sources.

* Sportsbook Reviews Online season pages, 2007-08 .. 2022-23: scores, open and
  close spread and total (one line per team; the favourite's row carries the spread
  and the other row the total).
* ESPN scoreboard + odds endpoints, 2022-23 on: scores, season type, pre-game
  open/close lines from ESPN BET.

`build()` returns one DataFrame, one row per game, with the lines from the
**home** team's point of view (negative home_spread = home favoured). Fetch the raw
files first with tools/fetch_history.py.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .teams import league

RAW = Path("data/history/raw")
NBA = league("nba")

# SBR writes team names without spaces, and uses franchise names of the day.
SBR_TEAMS = {
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "newjersey": "BKN", "charlotte": "CHA",
    "chicago": "CHI", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN", "detroit": "DET",
    "goldenstate": "GSW", "houston": "HOU", "indiana": "IND", "laclippers": "LAC", "laclippers ": "LAC",
    "lalakers": "LAL", "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN",
    "neworleans": "NOP", "newyork": "NYK", "oklahomacity": "OKC", "seattle": "OKC", "orlando": "ORL",
    "philadelphia": "PHI", "phoenix": "PHX", "portland": "POR", "sacramento": "SAC",
    "sanantonio": "SAS", "toronto": "TOR", "utah": "UTA", "washington": "WAS",
    "neworleansoklahomacity": "NOP", "nooklahomacity": "NOP", "noklahomacity": "NOP",
}
KINDS = {1: "pre", 2: "reg", 3: "post", 5: "playin"}


def _num(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("pk", "pick", "p"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def season_types() -> dict[int, dict]:
    raw = json.loads((RAW / "season_types.json").read_text())
    return {int(y): {int(t): v for t, v in types.items()} for y, types in raw.items()}


def kind_for(d: date, season_start: int, types: dict[int, dict]) -> str:
    """Season type of a date, from ESPN's boundaries for that season."""
    t = types.get(season_start + 1, {})
    def start(i): return datetime.fromisoformat(t[i]["start"].replace("Z", "+00:00")).date() if i in t else None
    if start(3) and d >= start(3):
        return "post"
    if start(5) and d >= start(5):
        return "playin"
    if start(2) and d >= start(2):
        return "reg"
    return "pre"


def parse_sbr(path: Path, season_start: int, types: dict[int, dict]) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(path.read_text(errors="ignore")))
    df = max(tables, key=len)
    if str(df.iloc[0, 0]).strip() == "Date":                 # the header arrives as the first row
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:]
    df = df.rename(columns=lambda c: str(c).strip())
    cols = ["Date", "Rot", "VH", "Team", "Final", "Open", "Close"]
    df = df[cols].dropna(subset=["VH", "Team"]).reset_index(drop=True)
    rows = []
    for i in range(0, len(df) - 1, 2):
        v, h = df.iloc[i], df.iloc[i + 1]
        if str(v["VH"]).strip() != "V" or str(h["VH"]).strip() != "H":
            continue                                    # a neutral-site "N" pair or a broken row
        mmdd = int(v["Date"])
        month, day = mmdd // 100, mmdd % 100
        d = date(season_start if month >= 8 else season_start + 1, month, day)
        a_team = SBR_TEAMS.get(re.sub(r"[^a-z]", "", str(v["Team"]).lower()))
        h_team = SBR_TEAMS.get(re.sub(r"[^a-z]", "", str(h["Team"]).lower()))
        if not a_team or not h_team:
            raise ValueError(f"unknown SBR team in {path.name}: {v['Team']!r} / {h['Team']!r}")
        out = {"source": "sbr", "season": season_start, "date": d, "away": a_team, "home": h_team,
               "away_pts": _num(v["Final"]), "home_pts": _num(h["Final"])}
        for tag, col in (("open", "Open"), ("close", "Close")):
            a, b = _num(v[col]), _num(h[col])
            if a is None or b is None:
                out[f"total_{tag}"] = out[f"home_spread_{tag}"] = None
                continue
            total, spread = max(a, b), min(a, b)
            out[f"total_{tag}"] = total
            home_has_spread = b < a                     # the favourite's row carries the spread
            out[f"home_spread_{tag}"] = -spread if home_has_spread else spread
            if spread == 0:
                out[f"home_spread_{tag}"] = 0.0
        out["kind"] = kind_for(d, season_start, types)
        rows.append(out)
    return pd.DataFrame(rows)


def parse_espn(types: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "espn_scoreboard").glob("*.json")):
        day = json.loads(path.read_text())
        for e in day["events"]:
            if e["status"] != "STATUS_FINAL" or len(e["teams"]) != 2:
                continue
            home = next(t for t in e["teams"] if t["homeAway"] == "home")
            away = next(t for t in e["teams"] if t["homeAway"] == "away")
            try:
                h_abbr, a_abbr = NBA.canonical_abbr(home["abbr"]), NBA.canonical_abbr(away["abbr"])
            except KeyError:
                continue                                # all-star games, exhibitions
            kind = KINDS.get((e.get("season") or {}).get("type"))
            if kind is None:
                continue
            # ESPN dates are UTC; shift to the US evening the game was played
            d = (datetime.fromisoformat(e["date"].replace("Z", "+00:00")) - timedelta(hours=8)).date()
            season_year = (e.get("season") or {}).get("year")
            odds_path = RAW / "espn_odds" / f"{e['id']}.json"
            odds = json.loads(odds_path.read_text())["items"] if odds_path.exists() else []
            pre = next((it for it in odds if it.get("provider") and "live" not in it["provider"].lower()
                        and it.get("overUnder") is not None), None)
            row = {"source": "espn", "espn_id": e["id"], "season": (season_year or d.year) - 1, "date": d,
                   "away": a_abbr, "home": h_abbr, "away_pts": _num(away["score"]), "home_pts": _num(home["score"]),
                   "kind": kind, "provider": pre["provider"] if pre else None}
            if pre:
                row["total_close"] = _num(pre["close_total"]) or float(pre["overUnder"])
                row["total_open"] = _num(pre["open_total"])
                row["home_spread_close"] = _num(pre["home_close_spread"]) if pre.get("home_close_spread") not in (None, "") else pre.get("spread")
                row["home_spread_open"] = _num(pre["home_open_spread"])
            rows.append(row)
    return pd.DataFrame(rows)


def build(use_espn_from: int = 2023) -> pd.DataFrame:
    """All games: SBR for seasons before `use_espn_from`, ESPN from then on."""
    types = season_types()
    parts = []
    for path in sorted((RAW / "sbr").glob("*.htm")):
        start = int(path.stem[:4])
        if start < use_espn_from - 1 + 0:
            parts.append(parse_sbr(path, start, types))
    sbr = pd.concat(parts, ignore_index=True)
    espn_all = parse_espn(types)
    espn = espn_all[espn_all["season"] >= use_espn_from - 1]
    # SBR's 2019-20 file stops when the season was suspended: take the rest from ESPN
    have = set(zip(sbr["date"], sbr["away"], sbr["home"]))
    near = lambda r: any((r["date"] + timedelta(days=k), r["away"], r["home"]) in have for k in (-1, 0, 1))
    fill = espn_all[(espn_all["season"] == 2019) & ~espn_all.apply(near, axis=1)]
    games = pd.concat([sbr, fill, espn], ignore_index=True)
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "home"]).reset_index(drop=True)
    games["margin"] = games["home_pts"] - games["away_pts"]
    games["points"] = games["home_pts"] + games["away_pts"]
    return games


def validate_overlap() -> pd.DataFrame:
    """Compare SBR and ESPN lines on the one season both cover (2022-23)."""
    types = season_types()
    sbr = parse_sbr(RAW / "sbr" / "2022-23.htm", 2022, types)
    espn = parse_espn(types)
    espn = espn[(espn["season"] == 2022) & (espn["kind"].isin(["reg", "post"]))]
    key = ["date", "away", "home"]
    m = sbr.merge(espn, on=key, suffixes=("_sbr", "_espn"))
    return m

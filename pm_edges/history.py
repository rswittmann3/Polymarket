"""Historical results with closing lines, for NBA, NHL and NFL, from cached raw files.

NBA
    * Sportsbook Reviews Online (SBR) season pages 2007-08 .. 2021-22 (scores, open/close spread
      and total; the favourite's row carries the spread and the other row the total).
    * ESPN scoreboard + odds endpoints from 2022-23 (and to fill the suspended 2019-20 season).
NHL
    * SBR pages 2007-08 .. 2021-22 (moneyline, puck line, total, with prices; none for 2020-21).
    * ESPN scoreboards for every season (they label overtime and shootout games) and ESPN odds
      from 2020-21 on. Sportsbooks settle totals without the shootout goal, so for a shootout
      game the total is home + away - 1.
NFL
    * nflverse games.csv, 1999 on: scores, closing spread/total/moneyline, weather, roof, rest.

`build_league()` returns one DataFrame, one row per game, with lines from the **home** team's
point of view (negative home_spread = home favoured). Fetch the raw files first with
tools/fetch_history.py.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .teams import league

RAW = Path("data/history/raw")          # NBA keeps this folder; other leagues use RAW/<league>
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
SBR_NHL_TEAMS = {
    "anaheim": "ANA", "arizona": "UTA", "arizonas": "UTA", "phoenix": "UTA", "atlanta": "WPG", "boston": "BOS",
    "buffalo": "BUF", "calgary": "CGY", "carolina": "CAR", "chicago": "CHI", "colorado": "COL", "columbus": "CBJ",
    "dallas": "DAL", "detroit": "DET", "edmonton": "EDM", "florida": "FLA", "losangeles": "LAK",
    "minnesota": "MIN", "montreal": "MTL", "nashville": "NSH", "newjersey": "NJD", "nyislanders": "NYI",
    "nyrangers": "NYR", "ottawa": "OTT", "philadelphia": "PHI", "pittsburgh": "PIT", "sanjose": "SJS",
    "seattle": "SEA", "seattlekraken": "SEA", "stlouis": "STL", "tampa": "TBL", "tampabay": "TBL",
    "toronto": "TOR", "vancouver": "VAN", "vegas": "VGK", "washington": "WSH", "winnipeg": "WPG",
    "winnipegjets": "WPG",
}
KINDS = {1: "pre", 2: "reg", 3: "post", 5: "playin"}
# ESPN's list of sportsbooks differs by season; take the first of these that posted a total
NHL_PROVIDERS = ["DraftKings", "ESPN BET", "Caesars Sportsbook", "Caesars", "MGM", "Bet365"]
NBA_PROVIDERS: list[str] = []
# a total outside these bounds is a bad feed value (ESPN sometimes returns a moneyline in the total field)
TOTAL_RANGE = {"nba": (150.0, 300.0), "nhl": (3.5, 9.5)}


def raw_root(lg: str = "nba") -> Path:
    return RAW if lg == "nba" else RAW / lg


def _num(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("pk", "pick", "p"):
        return 0.0
    try:
        v = float(s)
    except ValueError:
        return None
    return None if np.isnan(v) else v


def season_types(lg: str = "nba") -> dict[int, dict]:
    raw = json.loads((raw_root(lg) / "season_types.json").read_text())
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


def _sbr_table(path: Path) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(io.StringIO(path.read_text(errors="ignore")))
    except (ValueError, ImportError):               # the page is not a table (a missing season)
        return None
    df = max(tables, key=len)
    if str(df.iloc[0, 0]).strip() == "Date":        # the header arrives as the first row
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:]
    return df.rename(columns=lambda c: str(c).strip())


def parse_sbr(path: Path, season_start: int, types: dict[int, dict]) -> pd.DataFrame:
    """One NBA season page."""
    df = _sbr_table(path)
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


def parse_sbr_nhl(path: Path, season_start: int, types: dict[int, dict]) -> pd.DataFrame:
    """One NHL season page. The page has 14 columns (2007-14) or 16 (puck line added), by position:
    moneyline open/close at 8/9, then (puck line and its price), total/price open, total/price close.
    Neutral-site pairs are skipped. The over price is on the visitor's row and the under price on
    the home row."""
    df = _sbr_table(path)
    if df is None:
        return pd.DataFrame()
    wide = df.shape[1] >= 16
    c_open, c_close = (12, 14) if wide else (10, 12)
    df = df.dropna(subset=[df.columns[2], df.columns[3]]).reset_index(drop=True)
    rows = []
    for i in range(0, len(df) - 1, 2):
        v, h = df.iloc[i], df.iloc[i + 1]
        if str(v.iloc[2]).strip() != "V" or str(h.iloc[2]).strip() != "H":
            continue
        mmdd = _num(v.iloc[0])
        if mmdd is None:
            continue
        month, day = int(mmdd) // 100, int(mmdd) % 100
        year = season_start if month >= 8 else season_start + 1
        if season_start == 2019 and month in (8, 9):    # the 2019-20 playoffs resumed in August 2020
            year = 2020
        d = date(year, month, day)
        a_team = SBR_NHL_TEAMS.get(re.sub(r"[^a-z]", "", str(v.iloc[3]).lower()))
        h_team = SBR_NHL_TEAMS.get(re.sub(r"[^a-z]", "", str(h.iloc[3]).lower()))
        if not a_team or not h_team:
            raise ValueError(f"unknown SBR NHL team in {path.name}: {v.iloc[3]!r} / {h.iloc[3]!r}")
        out = {"source": "sbr", "season": season_start, "date": d, "away": a_team, "home": h_team,
               "away_pts": _num(v.iloc[7]), "home_pts": _num(h.iloc[7]),
               "away_ml_open": _num(v.iloc[8]), "home_ml_open": _num(h.iloc[8]),
               "away_ml": _num(v.iloc[9]), "home_ml": _num(h.iloc[9]),
               "total_open": _num(v.iloc[c_open]), "total_close": _num(v.iloc[c_close]),
               "over_odds": _num(v.iloc[c_close + 1]), "under_odds": _num(h.iloc[c_close + 1])}
        out["kind"] = kind_for(d, season_start, types)
        rows.append(out)
    return pd.DataFrame(rows)


def parse_espn(types: dict[int, dict], lg: str = "nba") -> pd.DataFrame:
    """Final games from the cached ESPN scoreboards, with the pre-game line where odds were fetched."""
    teams = league(lg)
    root = raw_root(lg)
    priority = NHL_PROVIDERS if lg == "nhl" else NBA_PROVIDERS
    rows = []
    for path in sorted((root / "espn_scoreboard").glob("*.json")):
        day = json.loads(path.read_text())
        for e in day["events"]:
            if e["status"] != "STATUS_FINAL" or len(e["teams"]) != 2:
                continue
            home = next(t for t in e["teams"] if t["homeAway"] == "home")
            away = next(t for t in e["teams"] if t["homeAway"] == "away")
            try:
                h_abbr, a_abbr = teams.canonical_abbr(home["abbr"]), teams.canonical_abbr(away["abbr"])
            except KeyError:
                continue                                # all-star games, exhibitions
            kind = KINDS.get((e.get("season") or {}).get("type"))
            if kind is None:
                continue
            # ESPN dates are UTC; shift to the US evening the game was played
            d = (datetime.fromisoformat(e["date"].replace("Z", "+00:00")) - timedelta(hours=8)).date()
            season_year = (e.get("season") or {}).get("year")
            odds_path = root / "espn_odds" / f"{e['id']}.json"
            odds = json.loads(odds_path.read_text())["items"] if odds_path.exists() else []
            lo, hi = TOTAL_RANGE.get(lg, (0.0, 1e9))
            usable = [it for it in odds if it.get("provider") and "live" not in it["provider"].lower()
                      and it.get("overUnder") is not None and lo <= float(it["overUnder"]) <= hi]
            pre = next((it for p in priority for it in usable if it["provider"] == p), usable[0] if usable else None)
            detail = e.get("detail") or ""
            ot = "SO" if detail.endswith("/SO") else "OT" if "/" in detail else "REG"
            row = {"source": "espn", "espn_id": e["id"], "season": (season_year or d.year) - 1, "date": d,
                   "away": a_abbr, "home": h_abbr, "away_pts": _num(away["score"]), "home_pts": _num(home["score"]),
                   "kind": kind, "provider": pre["provider"] if pre else None, "ot": ot,
                   "neutral": int(bool(e.get("neutral")))}
            if pre:
                in_range = lambda v: v is not None and lo <= v <= hi
                close_total, open_total = _num(pre.get("close_total")), _num(pre.get("open_total"))
                row["total_close"] = close_total if in_range(close_total) else float(pre["overUnder"])
                row["total_open"] = open_total if in_range(open_total) else None
                row["home_spread_close"] = (_num(pre.get("home_close_spread")) if pre.get("home_close_spread") not in (None, "")
                                            else pre.get("spread"))
                row["home_spread_open"] = _num(pre.get("home_open_spread"))
                over, under = _num(pre.get("overOdds")), _num(pre.get("underOdds"))
                if _book_ok(over, under, 1.0, 1.15):
                    row["over_odds"], row["under_odds"] = over, under
                home_ml, away_ml = _num(pre.get("home_ml")), _num(pre.get("away_ml"))
                if _book_ok(home_ml, away_ml, 1.0, 1.12):       # some seasons' "moneyline" is really a puck-line price
                    row["home_ml"], row["away_ml"] = home_ml, away_ml
            rows.append(row)
    return pd.DataFrame(rows)


def _implied(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _book_ok(a: float | None, b: float | None, lo: float, hi: float) -> bool:
    """Two American prices on opposite sides of one market that add up to a plausible overround."""
    if a is None or b is None or abs(a) < 100 or abs(b) < 100:
        return False
    return lo <= _implied(a) + _implied(b) <= hi


def _finish(games: pd.DataFrame) -> pd.DataFrame:
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "home"]).reset_index(drop=True)
    games["margin"] = games["home_pts"] - games["away_pts"]
    if "points" not in games:
        games["points"] = games["home_pts"] + games["away_pts"]
    return games


def _near(frame: pd.DataFrame) -> "callable":
    have = set(zip(frame["date"], frame["away"], frame["home"]))
    return lambda r: any((r["date"] + timedelta(days=k), r["away"], r["home"]) in have for k in (-1, 0, 1))


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------

def build(use_espn_from: int = 2023) -> pd.DataFrame:
    """NBA: SBR for seasons before `use_espn_from`, ESPN from then on."""
    return build_nba(use_espn_from)


def build_nba(use_espn_from: int = 2023) -> pd.DataFrame:
    types = season_types("nba")
    parts = []
    for path in sorted((RAW / "sbr").glob("*.htm")):
        start = int(path.stem[:4])
        if start < use_espn_from - 1 + 0:
            parts.append(parse_sbr(path, start, types))
    sbr = pd.concat(parts, ignore_index=True)
    espn_all = parse_espn(types, "nba")
    espn = espn_all[espn_all["season"] >= use_espn_from - 1]
    # SBR's 2019-20 file stops when the season was suspended: take the rest from ESPN
    fill = espn_all[(espn_all["season"] == 2019) & ~espn_all.apply(_near(sbr), axis=1)]
    return _finish(pd.concat([sbr, fill, espn], ignore_index=True))


# ---------------------------------------------------------------------------
# NHL
# ---------------------------------------------------------------------------

NHL_ESPN_ONLY = {2020}       # SBR has no page for 2020-21
NHL_SBR_LAST = 2021          # SBR's 2022-23 page stops in October


def build_nhl() -> pd.DataFrame:
    """SBR (with prices) up to 2021-22, ESPN after; overtime and shootout labels come from ESPN.
    `points` is the total settled by sportsbooks: a shootout game's winning goal is not counted."""
    types = season_types("nhl")
    root = raw_root("nhl")
    espn_all = parse_espn(types, "nhl")
    parts = []
    for path in sorted((root / "sbr").glob("*.htm")):
        start = int(path.stem[:4])
        if start > NHL_SBR_LAST or start in NHL_ESPN_ONLY:
            continue
        p = parse_sbr_nhl(path, start, types)
        if len(p):
            parts.append(p)
    sbr = pd.concat(parts, ignore_index=True)
    # label SBR games with ESPN's overtime/shootout flag
    lab = {(r.date, r.away, r.home): (r.ot, r.away_pts, r.home_pts) for r in espn_all.itertuples()}
    def label(r):
        for k in (0, -1, 1):
            hit = lab.get((r["date"] + timedelta(days=k), r["away"], r["home"]))
            if hit:
                return hit
        return (None, None, None)
    got = sbr.apply(label, axis=1, result_type="expand")
    sbr["ot"] = got[0]
    sbr["espn_away_pts"], sbr["espn_home_pts"] = got[1], got[2]
    espn = espn_all[(espn_all["season"] > NHL_SBR_LAST) | espn_all["season"].isin(NHL_ESPN_ONLY)]
    games = pd.concat([sbr, espn], ignore_index=True)
    games["ot"] = games["ot"].fillna("REG")
    games["points"] = games["home_pts"] + games["away_pts"] - (games["ot"] == "SO").astype(int)
    return _finish(games)


# ---------------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------------

NFL_POST = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}


def _nflverse() -> pd.DataFrame:
    return pd.read_csv(raw_root("nfl") / "games.csv")


def _nfl_frame(raw: pd.DataFrame) -> pd.DataFrame:
    nfl = league("nfl")
    g = pd.DataFrame({
        "source": "nflverse", "season": raw["season"], "date": pd.to_datetime(raw["gameday"]),
        "week": raw["week"], "kind": np.where(raw["game_type"] == "REG", "reg", "post"),
        "playoff_round": raw["game_type"].map(NFL_POST),
        "away": raw["away_team"].map(nfl.canonical_abbr), "home": raw["home_team"].map(nfl.canonical_abbr),
        "away_pts": raw["away_score"], "home_pts": raw["home_score"],
        "home_spread_close": -raw["spread_line"], "total_close": raw["total_line"],
        "home_ml": raw["home_moneyline"], "away_ml": raw["away_moneyline"],
        "over_odds": raw["over_odds"], "under_odds": raw["under_odds"],
        "home_spread_odds": raw["home_spread_odds"], "away_spread_odds": raw["away_spread_odds"],
        "ot": np.where(raw["overtime"] == 1, "OT", "REG"), "neutral": (raw["location"] != "Home").astype(int),
        "roof": raw["roof"], "temp": raw["temp"], "wind": raw["wind"], "div_game": raw["div_game"],
        "weekday": raw["weekday"], "gametime": raw["gametime"],
        "home_rest": raw["home_rest"], "away_rest": raw["away_rest"],
        "home_qb": raw["home_qb_name"], "away_qb": raw["away_qb_name"],
    })
    g["hour"] = pd.to_numeric(g["gametime"].astype(str).str.slice(0, 2), errors="coerce")
    return g


def build_nfl() -> pd.DataFrame:
    """Played NFL games (those with a score)."""
    g = _nfl_frame(_nflverse())
    return _finish(g.dropna(subset=["home_pts", "away_pts"]).reset_index(drop=True))


def upcoming_nfl(away: str, home: str, when: date) -> dict | None:
    """The scheduled game from nflverse (with its lines), or None."""
    g = _nfl_frame(_nflverse())
    g = g[g["home_pts"].isna() & (g["away"] == away) & (g["home"] == home)]
    g = g[(g["date"] - pd.Timestamp(when)).abs() <= pd.Timedelta(days=1)]
    return g.iloc[0].to_dict() if len(g) else None


# ---------------------------------------------------------------------------

def build_league(lg: str) -> pd.DataFrame:
    if lg == "nba":
        return build_nba()
    if lg == "nhl":
        return build_nhl()
    if lg == "nfl":
        return build_nfl()
    raise KeyError(f"no history for league {lg!r}")


def validate_overlap(lg: str = "nba") -> pd.DataFrame:
    """Compare SBR and ESPN lines on the season both cover."""
    types = season_types(lg)
    if lg == "nhl":
        sbr = parse_sbr_nhl(raw_root(lg) / "sbr" / "2021-22.htm", 2021, types)
        season = 2021
    else:
        sbr = parse_sbr(RAW / "sbr" / "2022-23.htm", 2022, types)
        season = 2022
    espn = parse_espn(types, lg)
    espn = espn[(espn["season"] == season) & (espn["kind"].isin(["reg", "post"]))]
    return sbr.merge(espn, on=["date", "away", "home"], suffixes=("_sbr", "_espn"))

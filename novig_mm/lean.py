"""Does the NBA open-to-close lean pay in Novig's own prices?

Your frozen predictions (snapshots/frozen/nba_*.json) say where the sportsbook line will close. Here we test the same
call against Novig, using only what the recorder saw:

  1. match each frozen game to its Novig event (team names) and pick the Novig market at the line the prediction was made on
     (total: the strike equal to `total`; spread: the market holding `home_spread` for the home team);
  2. the lean: predicted close above the line -> buy Over (spread: predicted close below the home line -> buy home at that
     line; otherwise buy the away side);
  3. entry = Novig's state when the prediction was saved; exit = its last state `exit_buffer_min` before the scheduled
     start (positions must be closed before the start: going live cancels resting orders and live takers pay a fee).

Three results per bet, in cents per contract (+ = made money):
    gross   mid at exit - mid at entry                    (the move itself, no trading cost)
    taker   exit bid - entry ask                          (cross the spread both ways)
    maker   exit bid - entry bid                          (rest a bid and get filled at the entry bid; optimistic: ignores
                                                           whether that fill would have happened and adverse selection)
Standard errors are across games. A market's state persists until its next recorded row, so entry/exit use the last row
at or before the target time and are skipped if that row is older than `max_age_min` (recorder was down).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from pm_edges.teams import league

from . import analyze

_NAME = re.compile(r"^(.+?)\s+([+-]?\d+(?:\.\d+)?)$")


def load_meta(paths: Iterable[Path]) -> tuple[dict[str, dict], dict[str, dict]]:
    events: dict[str, dict] = {}
    markets: dict[str, dict] = {}
    for r in analyze.read_rows(paths):
        (events if r["k"] == "ev" else markets)[r["e"] if r["k"] == "ev" else r["m"]] = r
    return events, markets


def event_teams(description: str, lg: str = "nba") -> tuple[str, str] | None:
    """'Miami Heat @ Los Angeles Clippers' -> ('MIA', 'LAC') as (away, home)."""
    if " @ " not in (description or ""):
        return None
    away, home = description.split(" @ ", 1)
    L = league(lg)
    a, h = L.team_from_text(away), L.team_from_text(home)
    return (a, h) if a and h else None


def match_event(game: dict, events: dict[str, dict], tol_hours: float = 6.0) -> dict | None:
    L = league("nba")
    want = (L.canonical_abbr(game["away"]), L.canonical_abbr(game["home"]))
    kick = datetime.fromisoformat(game["kickoff"]).timestamp() * 1000
    for e in events.values():
        if event_teams(e["d"]) == want and abs(e["st"] - kick) <= tol_hours * 3600_000:
            return e
    return None


def find_total_market(event_id: str, line: D, markets: dict[str, dict]) -> tuple[str, int] | None:
    """(marketId, index of the Over outcome)."""
    for m in markets.values():
        if m["e"] == event_id and m["mt"] == "TOTAL" and m.get("s") is not None and D(str(m["s"])) == line:
            names = [str(n).lower() for n in m["n"]]
            if names[0].startswith("over") and names[1].startswith("under"):
                return m["m"], 0
            if names[1].startswith("over") and names[0].startswith("under"):
                return m["m"], 1
    return None


def find_spread_market(event_id: str, home: str, line: D, markets: dict[str, dict]) -> tuple[str, int] | None:
    """(marketId, index of the outcome 'HOME <line>'), where the other outcome is the away team at -line."""
    L = league("nba")
    for m in markets.values():
        if m["e"] != event_id or m["mt"] != "SPREAD":
            continue
        for i, n in enumerate(m["n"]):
            mt = _NAME.match(str(n))
            if mt and L.team_from_text(mt.group(1)) == home and D(mt.group(2)) == line:
                return m["m"], i
    return None


@dataclass
class Plan:
    game: str
    kind: str
    side: str
    line: D
    pred: float
    saved_ms: int
    start_ms: int
    market_id: str
    idx: int


def plan_bets(frozen_paths: Iterable[Path], events: dict, markets: dict) -> tuple[list[Plan], list[dict]]:
    """One plan per game and kind from the earliest prediction of each game; plus reasons for games that cannot be tested."""
    seen: dict[tuple, tuple[int, dict]] = {}
    for p in sorted(frozen_paths):
        d = json.loads(Path(p).read_text())
        saved = int(datetime.fromisoformat(d["saved_at"]).timestamp() * 1000)
        for g in d["games"]:
            seen.setdefault((g["away"], g["home"], g["date"]), (saved, g))
    plans, skips = [], []
    for (away, home, date), (saved, g) in seen.items():
        name = f"{away}@{home} {date}"
        ev = match_event(g, events)
        if ev is None:
            skips.append({"game": name, "why": "no matching Novig event recorded"})
            continue
        if saved >= ev["st"]:
            skips.append({"game": name, "why": "prediction saved after the start"})
            continue
        hc = league("nba").canonical_abbr(home)
        for kind in ("total", "spread"):
            raw_line = g.get("total") if kind == "total" else g.get("home_spread")
            pred = g.get("pred_close_total") if kind == "total" else g.get("pred_close_spread")
            if raw_line is None or pred is None or float(pred) == float(raw_line):
                skips.append({"game": f"{name} {kind}", "why": "no predicted move"})
                continue
            line = D(str(raw_line))
            hit, side, idx = None, "", 0
            if kind == "total":
                hit = find_total_market(ev["e"], line, markets)
                over = float(pred) > float(line)
                if hit:
                    idx = hit[1] if over else 1 - hit[1]
                    side = f"{'Over' if over else 'Under'} {line}"
            else:
                hit = find_spread_market(ev["e"], hc, line, markets)
                home_side = float(pred) < float(line)          # line expected to move toward a bigger home favourite
                if hit:
                    idx = hit[1] if home_side else 1 - hit[1]
                    side = f"{hc} {line:+}" if home_side else f"{league('nba').canonical_abbr(away)} {-line:+}"
            if not hit:
                skips.append({"game": f"{name} {kind}", "why": f"no Novig {kind} market at {line}"})
                continue
            plans.append(Plan(name, kind, side, line, float(pred), saved, ev["st"], hit[0], idx))
    return plans, skips


def load_series(touch_paths: Iterable[Path], wanted: set[str]) -> dict[str, list[tuple]]:
    """{marketId: [(ts_ms, bid0, bid1), ...] sorted}; bids are Decimal or None."""
    out: dict[str, list[tuple]] = {m: [] for m in wanted}
    for r in analyze.read_rows(touch_paths):
        if r["m"] in out:
            b0, b1 = r["b"]
            out[r["m"]].append((r["ts"], None if b0 is None else D(b0), None if b1 is None else D(b1)))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def state_at(series: list[tuple], t_ms: int, max_age_ms: int) -> tuple | None:
    """Last row at or before t_ms that has both sides quoted and is no older than max_age_ms."""
    best = None
    for row in series:
        if row[0] > t_ms:
            break
        best = row
    if best is None or t_ms - best[0] > max_age_ms or best[1] is None or best[2] is None:
        return None
    return best


def evaluate(frozen_paths: Iterable[Path], touch_paths: Iterable[Path], meta_paths: Iterable[Path],
             exit_buffer_min: float = 5.0, max_age_min: float = 10.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events, markets = load_meta(list(meta_paths))
    plans, skips = plan_bets(list(frozen_paths), events, markets)
    series = load_series(list(touch_paths), {p.market_id for p in plans})
    age, buf = int(max_age_min * 60_000), int(exit_buffer_min * 60_000)
    rows = []
    for p in plans:
        s = series[p.market_id]
        i, j = p.idx, 1 - p.idx
        e_row = state_at(s, p.saved_ms, age)
        x_row = state_at(s, p.start_ms - buf, age)
        if e_row is None or x_row is None or x_row[0] < p.saved_ms:
            skips.append({"game": f"{p.game} {p.kind}", "why": "no usable entry/exit state (recorder gap or market closed early)"})
            continue
        bid1, other1 = e_row[1 + i], e_row[1 + j]
        bid2, other2 = x_row[1 + i], x_row[1 + j]
        ask1 = 1 - other1
        mid1, mid2 = (1 + bid1 - other1) / 2, (1 + bid2 - other2) / 2
        rows.append({"game": p.game, "kind": p.kind, "side": p.side, "entry_ts": e_row[0], "exit_ts": x_row[0],
                     "entry_bid": float(bid1), "entry_ask": float(ask1), "exit_bid": float(bid2),
                     "gross_c": float(100 * (mid2 - mid1)), "taker_c": float(100 * (bid2 - ask1)),
                     "maker_c": float(100 * (bid2 - bid1)),
                     "spread_in_c": float(100 * (ask1 - bid1)), "spread_out_c": float(100 * ((1 - other2) - bid2))})
    return rows, skips


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for kind in ("total", "spread", "all"):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not rs:
            continue
        row: dict[str, Any] = {"kind": kind, "bets": len(rs),
                               "moved_right_%": round(100 * sum(r["gross_c"] > 0 for r in rs) / max(1, sum(r["gross_c"] != 0 for r in rs)), 1),
                               "spread_in_c": round(mean(r["spread_in_c"] for r in rs), 2)}
        for col in ("gross_c", "taker_c", "maker_c"):
            v = [r[col] for r in rs]
            row[col] = round(mean(v), 3)
            row[col + "_se"] = round(stdev(v) / math.sqrt(len(v)), 3) if len(v) > 1 else None
        out.append(row)
    return out

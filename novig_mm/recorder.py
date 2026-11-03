"""Poll Novig's public catalog and record the top of book for upcoming games.

One catalog listing returns every market plus each outcome's best bid, so a full league costs a handful of
requests per poll (the CDN caches responses ~15 s, so polling faster than that only repeats data).

Output: data/novig/touch_<UTC date>.jsonl.gz, one JSON object per market when its touch changes (or every
`heartbeat_s`, default 120 s; a market's state persists until its next row):
    {"ts": ms, "m": marketId, "e": eventId, "lg": league, "mt": marketType, "k": strike|null, "st": startsTs,
     "ch": "WHEN_LIVE"|"ALWAYS", "vol": int, "oi": int, "b": [bid0|null, bid1|null], "q": [qty0, qty1], "o": [outcomeId0, outcomeId1]}
Absence of a market from a poll means its event left OPEN_PREGAME (went live, closed or settled).

Alongside it, meta_<UTC date>.jsonl.gz holds names, written once per market per run and refreshed for events every
10 minutes, so touches can be tied to games and outcomes:
    {"k": "ev", "e": eventId, "lg": league, "d": "Miami Heat @ Los Angeles Clippers", "st": startsTs}
    {"k": "mk", "m": marketId, "e": eventId, "d": "LAC -1.5", "mt": type, "s": strike|null, "n": [outcome0 name, outcome1 name]}
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .book import touch_from_market
from .client import NovigClient

DEFAULT_TYPES = "MONEY,SPREAD,TOTAL"
EVENTS_EVERY_S = 600.0


def rows_from_markets(markets: Iterable[dict[str, Any]], ts_ms: int, league: str) -> list[dict[str, Any]]:
    out = []
    for m in markets:
        if m.get("status") != "OPEN":
            continue
        t = touch_from_market(m)
        if t is None:
            continue
        out.append({
            "ts": ts_ms, "m": m["marketId"], "e": m["eventId"], "lg": league, "mt": m["marketType"],
            "k": m.get("strike"), "st": m["startsTs"], "ch": (m.get("fee") or {}).get("charged"),
            "vol": m.get("volume", 0), "oi": m.get("openInterest", 0),
            "b": [None if p is None else str(p) for p in t.bid], "q": list(t.bid_qty), "o": list(t.outcome_ids),
        })
    return out


def out_path(out_dir: Path, ts_ms: int, prefix: str = "touch") -> Path:
    day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return out_dir / f"{prefix}_{day}.jsonl.gz"


def market_meta_rows(markets: Iterable[dict[str, Any]], seen: set[str]) -> list[dict[str, Any]]:
    """One meta row per market not yet in `seen` (which is updated). Only two-outcome markets, like the touches."""
    out = []
    for m in markets:
        if m["marketId"] in seen or len(m.get("outcomes") or []) != 2:
            continue
        seen.add(m["marketId"])
        out.append({"k": "mk", "m": m["marketId"], "e": m["eventId"], "d": m.get("description"), "mt": m["marketType"],
                    "s": m.get("strike"), "n": [o.get("name") for o in m["outcomes"]]})
    return out


def event_meta_rows(events: Iterable[dict[str, Any]], league: str) -> list[dict[str, Any]]:
    return [{"k": "ev", "e": e["eventId"], "lg": league, "d": e.get("description"), "st": e["startsTs"]} for e in events]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:       # appended gzip members read back as one stream
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def changed_rows(rows: list[dict[str, Any]], last: dict[str, tuple], ts_ms: int, heartbeat_ms: int) -> list[dict[str, Any]]:
    """Keep a row only if the touch changed since the last one written for that market, or the last write is at least
    `heartbeat_ms` old. A market's state persists until its next row, so nothing is lost; heartbeat 0 keeps every row."""
    out = []
    for r in rows:
        state = (tuple(r["b"]), tuple(r["q"]))
        prev = last.get(r["m"])
        if prev is None or prev[0] != state or ts_ms - prev[1] >= heartbeat_ms:
            out.append(r)
            last[r["m"]] = (state, ts_ms)
    return out


def poll_once(client: NovigClient, leagues: list[str], market_types: str, window_hours: float,
              out_dir: Path, now_ms: int | None = None, seen: set[str] | None = None, with_events: bool = False,
              last_written: dict[str, tuple] | None = None, heartbeat_s: float = 0.0) -> int:
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    seen = set() if seen is None else seen
    last_written = {} if last_written is None else last_written
    before = ts + int(window_hours * 3600_000)
    n = 0
    for lg in leagues:
        markets = client.markets(lg, market_types=market_types, starts_before_ms=before)
        rows = changed_rows(rows_from_markets(markets, ts, lg), last_written, ts, int(heartbeat_s * 1000))
        append_rows(out_path(out_dir, ts), rows)
        meta = market_meta_rows(markets, seen)
        if with_events:
            meta += event_meta_rows(client.events(lg, starts_before_ms=before), lg)
        if meta:
            append_rows(out_path(out_dir, ts, "meta"), meta)
        n += len(rows)
    return n


def run(client: NovigClient, leagues: list[str], market_types: str = DEFAULT_TYPES, interval_s: float = 30.0,
        duration_s: float | None = None, window_hours: float = 24.0, out_dir: Path = Path("data/novig"),
        heartbeat_s: float = 120.0,
        log: Callable[[str], None] = print, sleep=time.sleep, clock=time.time) -> None:
    started = clock()
    seen: set[str] = set()
    last_written: dict[str, tuple] = {}
    last_events = float("-inf")
    while True:
        t0 = clock()
        try:
            with_events = t0 - last_events >= EVENTS_EVERY_S
            n = poll_once(client, leagues, market_types, window_hours, out_dir, seen=seen, with_events=with_events,
                          last_written=last_written, heartbeat_s=heartbeat_s)
            if with_events:
                last_events = t0
            log(f"{datetime.fromtimestamp(t0, tz=timezone.utc):%H:%M:%S}Z recorded {n} market touches")
        except Exception as e:                                # keep recording through transient failures
            log(f"poll failed: {e}")
        if duration_s is not None and clock() - started + interval_s > duration_s:
            return
        sleep(max(0.0, interval_s - (clock() - t0)))

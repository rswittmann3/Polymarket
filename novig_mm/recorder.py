"""Poll Novig's public catalog and record the top of book for upcoming games.

One catalog listing returns every market plus each outcome's best bid, so a full league costs a handful of
requests per poll (the CDN caches responses ~15 s, so polling faster than that only repeats data).

Output: data/novig/touch_<UTC date>.jsonl.gz, one JSON object per market per poll:
    {"ts": ms, "m": marketId, "e": eventId, "lg": league, "mt": marketType, "k": strike|null, "st": startsTs,
     "ch": "WHEN_LIVE"|"ALWAYS", "vol": int, "oi": int, "b": [bid0|null, bid1|null], "q": [qty0, qty1], "o": [outcomeId0, outcomeId1]}
Absence of a market from a poll means its event left OPEN_PREGAME (went live, closed or settled).
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


def out_path(out_dir: Path, ts_ms: int) -> Path:
    day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return out_dir / f"touch_{day}.jsonl.gz"


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:       # appended gzip members read back as one stream
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def poll_once(client: NovigClient, leagues: list[str], market_types: str, window_hours: float,
              out_dir: Path, now_ms: int | None = None) -> int:
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    n = 0
    for lg in leagues:
        markets = client.markets(lg, market_types=market_types,
                                 starts_before_ms=ts + int(window_hours * 3600_000))
        rows = rows_from_markets(markets, ts, lg)
        append_rows(out_path(out_dir, ts), rows)
        n += len(rows)
    return n


def run(client: NovigClient, leagues: list[str], market_types: str = DEFAULT_TYPES, interval_s: float = 30.0,
        duration_s: float | None = None, window_hours: float = 24.0, out_dir: Path = Path("data/novig"),
        log: Callable[[str], None] = print, sleep=time.sleep, clock=time.time) -> None:
    started = clock()
    while True:
        t0 = clock()
        try:
            n = poll_once(client, leagues, market_types, window_hours, out_dir)
            log(f"{datetime.fromtimestamp(t0, tz=timezone.utc):%H:%M:%S}Z recorded {n} market touches")
        except Exception as e:                                # keep recording through transient failures
            log(f"poll failed: {e}")
        if duration_s is not None and clock() - started + interval_s > duration_s:
            return
        sleep(max(0.0, interval_s - (clock() - t0)))

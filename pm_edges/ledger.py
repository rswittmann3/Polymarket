"""A paper-trade ledger that scores flagged edges by closing-line value (CLV), not just by results.

Each flagged edge is recorded once with the price you could have bought at. Every later `edges --ledger` run
before kickoff refreshes the sportsbook fair value for that same outcome, so the last refresh stands in for the
closing line. After the game `settle` reads the Polymarket resolution.

    clv    = fair_at_close - ask             (probability points per share; positive = you beat the close)
    pnl    = shares * (result - ask - fee)   (dollars)

CLV converges in dozens of bets; win/loss needs thousands, so CLV is the headline number. Nothing here places a
bet: stakes are the configured paper stake, capped at `max_stake`.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    outcome_index INTEGER NOT NULL,
    question TEXT, outcome TEXT, market_type TEXT,
    entered_at TEXT NOT NULL,
    kickoff TEXT,
    ask REAL NOT NULL, fee REAL NOT NULL, stake REAL NOT NULL, shares REAL NOT NULL,
    fair_entry REAL, pin_entry REAL, edge_entry REAL,
    fair_last REAL, pin_last REAL, last_seen TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    result REAL, settled_at TEXT, clv REAL, pnl REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_ledger ON ledger(game_id, market_id, outcome_index);
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _kickoff(raw: str | None) -> str | None:
    """Polymarket writes "2026-04-05 19:30:00+00"; normalise to an ISO UTC string."""
    if not raw:
        return None
    text = raw.strip().replace(" ", "T").replace("Z", "+00:00")
    if text.endswith("+00"):
        text += ":00"
    try:
        return _iso(datetime.fromisoformat(text))
    except ValueError:
        return None


@dataclass
class Summary:
    entries: int
    settled: int
    open: int
    mean_clv: float | None
    clv_se: float | None
    clv_t: float | None
    beat_close: float | None        # share of settled bets whose close was better than the ask
    roi: float | None               # pnl / dollars staked
    pnl: float
    staked: float
    hit_rate: float | None


class Ledger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)

    # -- recording -------------------------------------------------------------------------------------

    def record(self, game_id: str, edges: list, cfg, at: datetime) -> tuple[int, int]:
        """Add newly flagged edges; refresh the fair value of every open entry in this scan. Returns (added, refreshed)."""
        added = refreshed = 0
        stake = min(cfg.default_stake, cfg.max_stake)
        with self.conn:
            for e in edges:
                key = (game_id, e.market.market_id, e.outcome.index)
                row = self.conn.execute("SELECT * FROM ledger WHERE game_id=? AND market_id=? AND outcome_index=?", key).fetchone()
                if row is None:
                    if not e.flagged or e.ask is None or e.fair.prob is None:
                        continue
                    cost = e.ask + (e.fee or 0.0)
                    shares = stake / cost
                    if e.size_with_edge:
                        shares = min(shares, e.size_with_edge)
                    pin = e.fair.book_prob("pinnacle") if e.fair.source == "books" else None
                    self.conn.execute(
                        "INSERT INTO ledger (game_id, market_id, outcome_index, question, outcome, market_type, entered_at, kickoff, ask, fee,"
                        " stake, shares, fair_entry, pin_entry, edge_entry, fair_last, pin_last, last_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (*key, e.market.question, e.outcome.name, e.market.market_type, _iso(at), _kickoff(e.market.game_start_time), e.ask,
                         e.fee or 0.0, shares * cost, shares, e.fair.prob, pin, e.edge, e.fair.prob, pin, _iso(at)))
                    added += 1
                elif row["status"] == "open" and e.fair.prob is not None and (row["kickoff"] is None or _iso(at) < row["kickoff"]):
                    pin = e.fair.book_prob("pinnacle") if e.fair.source == "books" else row["pin_last"]
                    self.conn.execute("UPDATE ledger SET fair_last=?, pin_last=?, last_seen=? WHERE id=?", (e.fair.prob, pin, _iso(at), row["id"]))
                    refreshed += 1
        return added, refreshed

    def add_manual(self, game_id: str, market_id: str, outcome_index: int, question: str, outcome: str, market_type: str,
                   kickoff: str | None, ask: float, fee: float, stake: float, fair: float | None, at: datetime) -> bool:
        """Record a paper trade chosen by hand (a tail, a hunch) so it is scored like any flagged edge.
        `fair` is the sportsbook no-vig price at entry when known; it stands in for the close until a scan refreshes it."""
        shares = stake / (ask + fee)
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO ledger (game_id, market_id, outcome_index, question, outcome, market_type, entered_at, kickoff, ask, fee,"
                " stake, shares, fair_entry, edge_entry, fair_last, last_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (game_id, market_id, outcome_index, question, outcome, market_type, _iso(at), _kickoff(kickoff), ask, fee, stake, shares,
                 fair, None if fair is None else fair - ask - fee, fair, _iso(at)))
        return cur.rowcount == 1

    # -- settling --------------------------------------------------------------------------------------

    def settle_entry(self, entry_id: int, result: float, at: datetime) -> None:
        row = self.conn.execute("SELECT * FROM ledger WHERE id=?", (entry_id,)).fetchone()
        clv = None if row["fair_last"] is None else row["fair_last"] - row["ask"]
        pnl = row["shares"] * (result - row["ask"] - row["fee"])
        with self.conn:
            self.conn.execute("UPDATE ledger SET status='settled', result=?, settled_at=?, clv=?, pnl=? WHERE id=?",
                              (result, _iso(at), clv, pnl, entry_id))

    def settle_open(self, now: datetime, get: Callable | None = None) -> int:
        """Settle entries whose Polymarket market has resolved. Returns how many were settled."""
        get = get or _gamma
        n = 0
        for row in self.conn.execute("SELECT * FROM ledger WHERE status='open'").fetchall():
            if row["kickoff"] and _iso(now) < row["kickoff"]:
                continue
            result = resolution(get(f"https://gamma-api.polymarket.com/markets/{row['market_id']}"), row["outcome_index"])
            if result is not None:
                self.settle_entry(row["id"], result, now)
                n += 1
        return n

    # -- reporting -------------------------------------------------------------------------------------

    def rows(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM ledger ORDER BY entered_at")]

    def summary(self) -> Summary:
        rows = self.rows()
        done = [r for r in rows if r["status"] == "settled"]
        clvs = [r["clv"] for r in done if r["clv"] is not None]
        mean = sum(clvs) / len(clvs) if clvs else None
        se = None
        if len(clvs) > 1:
            sd = math.sqrt(sum((c - mean) ** 2 for c in clvs) / (len(clvs) - 1))
            se = sd / math.sqrt(len(clvs))
        staked = sum(r["shares"] * (r["ask"] + r["fee"]) for r in done)
        pnl = sum(r["pnl"] for r in done)
        return Summary(len(rows), len(done), len(rows) - len(done), mean, se, mean / se if se else None,
                       sum(c > 0 for c in clvs) / len(clvs) if clvs else None, pnl / staked if staked else None, pnl, staked,
                       sum(r["result"] > 0.5 for r in done) / len(done) if done else None)


def resolution(market: dict | None, outcome_index: int) -> float | None:
    """The settled payout (0, 1 or a refund share) of an outcome, or None while the market is unresolved."""
    if not market or not market.get("closed"):
        return None
    try:
        prices = [float(p) for p in json.loads(market["outcomePrices"])]
    except (KeyError, TypeError, ValueError):
        return None
    if len(prices) <= outcome_index or abs(sum(prices) - 1.0) > 0.02:
        return None
    if any(0.02 < p < 0.98 for p in prices) and not all(abs(p - 0.5) < 0.02 for p in prices):
        return None                                  # still trading near a real price: not a final resolution
    return round(prices[outcome_index], 4)


def _gamma(url: str) -> dict | None:
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 (research; pm_edges)"}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None

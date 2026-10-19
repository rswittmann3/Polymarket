"""SQLite snapshots of both sources. The first snapshot of a game is its "open"."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import BookRow, PMMarket, Selection

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS book_rows (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    game_id TEXT NOT NULL,
    source TEXT NOT NULL,
    book TEXT NOT NULL,
    market_key TEXT NOT NULL,
    market_type TEXT NOT NULL,
    period TEXT NOT NULL,
    player TEXT NOT NULL,
    stat TEXT NOT NULL,
    line REAL,
    side TEXT NOT NULL,
    price_american REAL NOT NULL,
    prob_implied REAL NOT NULL,
    liquidity REAL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pm_rows (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    game_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    slug TEXT,
    question TEXT,
    market_type TEXT,
    outcome_index INTEGER NOT NULL,
    outcome TEXT,
    period TEXT, player TEXT, stat TEXT, line REAL, side TEXT,
    bid REAL, ask REAL, ask_size REAL,
    last_trade REAL,
    liquidity REAL,
    price_source TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_book_game ON book_rows(game_id, snapshot_id);
CREATE INDEX IF NOT EXISTS ix_pm_game ON pm_rows(game_id, snapshot_id);
"""


@dataclass
class Snapshot:
    id: int
    taken_at: datetime
    book_rows: list[BookRow]
    pm_rows: list[dict]


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def write(self, game_id: str, taken_at: datetime, book_rows: list[BookRow],
              markets: list[PMMarket], note: str = "") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO snapshots (game_id, taken_at, note) VALUES (?, ?, ?)",
                (game_id, taken_at.isoformat(), note))
            sid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO book_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(sid, r.game_id, r.source, r.book, r.market_key, r.selection.market_type,
                  r.selection.period, r.selection.player, r.selection.stat, r.selection.line,
                  r.selection.side, r.price_american, r.prob_implied, r.liquidity,
                  r.fetched_at.isoformat()) for r in book_rows])
            pm = []
            for m in markets:
                for o in m.outcomes:
                    s = o.selection
                    pm.append((sid, m.game_id, m.market_id, m.slug, m.question, m.market_type,
                               o.index, o.name,
                               s.period if s else None, s.player if s else None,
                               s.stat if s else None, s.line if s else None, s.side if s else None,
                               o.bid, o.ask, o.ask_size, m.last_trade_price, m.liquidity,
                               o.price_source, (m.fetched_at or taken_at).isoformat()))
            self.conn.executemany(
                "INSERT INTO pm_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", pm)
        return sid

    def snapshots(self, game_id: str) -> list[Snapshot]:
        snaps = self.conn.execute(
            "SELECT id, taken_at FROM snapshots WHERE game_id = ? ORDER BY taken_at, id",
            (game_id,)).fetchall()
        out = []
        for s in snaps:
            books = [
                BookRow(
                    game_id=r["game_id"], book=r["book"], market_key=r["market_key"],
                    selection=Selection(r["market_type"], r["period"], r["player"], r["stat"],
                                        r["line"], r["side"]),
                    price_american=r["price_american"], prob_implied=r["prob_implied"],
                    fetched_at=datetime.fromisoformat(r["fetched_at"]), source=r["source"],
                    liquidity=r["liquidity"])
                for r in self.conn.execute("SELECT * FROM book_rows WHERE snapshot_id = ?", (s["id"],))
            ]
            pm = [dict(r) for r in self.conn.execute(
                "SELECT * FROM pm_rows WHERE snapshot_id = ?", (s["id"],))]
            out.append(Snapshot(s["id"], datetime.fromisoformat(s["taken_at"]), books, pm))
        return out

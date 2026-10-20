"""`pm_edges trends`: which historical situational angles apply to a game, and how much to believe them.

The library holds broad angles written down before looking at results (season openers,
home openers, openers after a playoff meeting, ...). For each one the report shows the
record both ways, whether it is active for this game, and a plain verdict. Your own
TrendsCenter-style filter can be added with --angle; it gets a leave-one-out table so
you can see which conditions carry the result.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import angles as A
from .teams import league as _league

# (name, query, note). Written down in advance; not tuned on results.
LIBRARY: list[tuple[str, str]] = [
    ("Any team's season opener", "game_num == 1"),
    ("Home team's opener", "game_num == 1 and at_home == 1"),
    ("Opener after a playoff meeting (either team home)", "game_num == 1 and prev_vs_opp_post == 1"),
    ("Home opener after a playoff meeting", "game_num == 1 and at_home == 1 and prev_vs_opp_post == 1"),
    ("Home opener, home spread between -7 and +2", "game_num == 1 and at_home == 1 and spread > -7 and spread < 2"),
    ("Home opener after playoff meeting, spread -7 to +2",
     "game_num == 1 and at_home == 1 and prev_vs_opp_post == 1 and spread > -7 and spread < 2"),
    ("Home team, first five games", "game_num <= 5 and at_home == 1"),
    ("Total at least 3 above both teams' scoring last season", "at_home == 1 and total_vs_avg >= 3"),
    ("Total at least 3 below both teams' scoring last season", "at_home == 1 and total_vs_avg <= -3"),
]


def history_frames(data_dir: Path = Path("data")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, team-game features), built from the cached raw files on first use."""
    from . import history
    hist = data_dir / "history"
    games_path, team_path = hist / "games.pkl", hist / "team_games.pkl"
    if not games_path.exists():
        history.RAW = hist / "raw"
        games = history.build()
        games.to_pickle(games_path)
    games = pd.read_pickle(games_path)
    if not team_path.exists():
        A.team_games(games).to_pickle(team_path)
    return games, pd.read_pickle(team_path)


def verdict(r: A.Result, bet: str) -> str:
    d = r.wins + r.losses
    if d < 20:
        return "too few games to say anything"
    if r.p_vs_coin < 0.01:
        return "unusual at this sample size, but see the multiple-testing note"
    if r.win_pct < 0.5:
        return "leans the other way"
    return "indistinguishable from a coin flip"


def run(console: Console, league: str, away: str, home: str, when: date, spread: float, total: float,
        data_dir: Path, angle: str | None, bet: str, leave_one_out: bool) -> None:
    lg = _league(league)
    away, home = lg.canonical_abbr(away), lg.canonical_abbr(home)
    _, t = history_frames(data_dir)
    if "total_vs_avg" not in t.columns or "prev_total" not in t.columns:
        raise SystemExit("history is out of date; delete data/history/*.pkl and rerun")
    season = when.year if when.month >= 8 else when.year - 1
    day = pd.Timestamp(when)
    row = A.upcoming(t, season, home, away, True, day, spread, total)
    prev_avg = t[(t["team"] == home) & (t["season"] == season - 1) & (t["kind"] == "reg")]["points"].mean()
    opp_avg = t[(t["team"] == away) & (t["season"] == season - 1) & (t["kind"] == "reg")]["points"].mean()
    row["total_vs_avg"] = total - (prev_avg + opp_avg) / 2

    games = len(t[t["kind"] != "pre"].drop_duplicates("game_id"))
    console.print(f"[bold]{away} @ {home}[/] {when}  home spread {spread:+g}, total {total:g}   "
                  f"(history: {games:,} games, {int(t['season'].min())}-{int(t['season'].max())} seasons, closing lines)")
    last = "none" if pd.isna(row.get("prev_vs_opp_days")) else \
        f"{'playoffs' if row['prev_vs_opp_post'] else 'regular season'}, {int(row['prev_vs_opp_days'])} days ago, total {row['prev_total']:g}"
    console.print(f"home game #{int(row['game_num'])} of the season · {home} won {int(row['prev_po_wins'])} playoff games last season · "
                  f"last meeting: {last} · total vs last meeting {row['total_vs_prev']:+.1f} · "
                  f"total vs both teams' scoring last season {row['total_vs_avg']:+.1f}")

    table = Table(title="Situational angles (record at flat -110; Under and Over)", title_justify="left")
    for c in ("angle", "active", "Under", "Over", "games", "verdict"):
        table.add_column(c, justify="right" if c in ("Under", "Over", "games") else "left", overflow="fold")
    for name, q in LIBRARY:
        u, o = A.evaluate(t, q, "under"), A.evaluate(t, q, "over")
        table.add_row(name, "[green]yes[/]" if A.is_active(row, q) else "no",
                      f"{u.record()} ({u.win_pct:.0%})", f"{o.record()} ({o.win_pct:.0%})", str(u.n),
                      verdict(u if u.win_pct >= o.win_pct else o, bet))
    console.print(table)
    console.print("[dim]p-values here are for one pre-written angle each. If you pick the best of many, "
                  "expect to find something: in a shuffled-outcome test the best of ~140 random angles "
                  "with 12-15 games reached 79-92%.[/]")

    if angle:
        conds = [c.strip() for c in angle.split(" and ") if c.strip()]
        r = A.evaluate(t, angle, bet)
        console.print(f"\n[bold]Your angle ({bet})[/]: {angle}")
        console.print(f"  record {r.record()} ({r.win_pct:.0%}), {r.n} games over {r.seasons} seasons, ROI {r.roi:.0%} at -110, "
                      f"last 10 {r.last10[0]}-{r.last10[1]}, P(this good by chance) = {r.p_vs_coin:.3f}")
        console.print(f"  active for this game: {'[green]yes[/]' if A.is_active(row, angle) else 'no'}")
        if leave_one_out and len(conds) > 1:
            lo = Table(title="Drop one condition at a time", title_justify="left")
            for c in ("without", "record", "win %", "games", "extra games that lost"):
                lo.add_column(c, justify="right" if c != "without" else "left")
            for c, rr in A.leave_one_out(t, conds, bet):
                lo.add_row(c, rr.record(), f"{rr.win_pct:.0%}", str(rr.n), str(rr.losses - r.losses))
            console.print(lo)
            console.print("[dim]If dropping a condition barely changes the record, it did nothing. If dropping one adds many "
                          "games and the win rate falls to ~50%, that condition is doing all the work on a few games.[/]")

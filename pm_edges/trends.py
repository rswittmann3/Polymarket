"""`pm_edges trends`: which historical situational angles apply to a game, and how much to believe them.

Each league has a library of broad angles written down before looking at results (season openers,
dome games, back-to-backs, home underdogs, ...). For each one the report shows the record, whether it
is active for this game, and a plain verdict. Your own TrendsCenter-style filter can be added with
--angle; it gets a leave-one-out table so you can see which conditions carry the result.

Records are at the posted price where the history has it (hockey and football moneylines, hockey
totals, football spreads from 2009 on) and at flat -110 otherwise.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import angles as A
from .teams import league as _league

# (name, query, bet). Written down in advance; not tuned on results. bet "total" shows Under and Over.
LIBRARY: dict[str, list[tuple[str, str, str]]] = {
    "nba": [
        ("Any team's season opener", "game_num == 1", "total"),
        ("Home team's opener", "game_num == 1 and at_home == 1", "total"),
        ("Opener after a playoff meeting (either team home)", "game_num == 1 and prev_vs_opp_post == 1", "total"),
        ("Home opener after a playoff meeting", "game_num == 1 and at_home == 1 and prev_vs_opp_post == 1", "total"),
        ("Home opener, home spread between -7 and +2", "game_num == 1 and at_home == 1 and spread > -7 and spread < 2", "total"),
        ("Home opener after playoff meeting, spread -7 to +2",
         "game_num == 1 and at_home == 1 and prev_vs_opp_post == 1 and spread > -7 and spread < 2", "total"),
        ("Home team, first five games", "game_num <= 5 and at_home == 1", "total"),
        ("Total at least 3 above both teams' scoring last season", "at_home == 1 and total_vs_avg >= 3", "total"),
        ("Total at least 3 below both teams' scoring last season", "at_home == 1 and total_vs_avg <= -3", "total"),
        ("Home team on a back-to-back, opponent rested", "kind == 'reg' and at_home == 1 and b2b == 1 and opp_b2b == 0", "ats"),
        ("Rest edge of 2+ days", "kind == 'reg' and rest_edge >= 2 and rest_days <= 10", "ats"),
    ],
    "nfl": [
        ("Dome or closed roof", "dome == 1", "total"),
        ("Wind 15+ mph outdoors", "wind >= 15 and dome == 0", "total"),
        ("32 degrees or colder outdoors", "temp <= 32 and dome == 0", "total"),
        ("Total 50 or higher", "total >= 50", "total"),
        ("Total 38 or lower", "total <= 38", "total"),
        ("Division games", "div_game == 1", "total"),
        ("Thursday games", "thu == 1", "total"),
        ("Playoff games", "kind == 'post'", "total"),
        ("Both teams off a bye", "kind == 'reg' and game_num > 1 and rest_days >= 13 and opp_rest_days >= 13", "total"),
        ("Total 3+ above both teams' scoring last season", "at_home == 1 and total_vs_avg >= 3", "total"),
        ("Total 3+ below both teams' scoring last season", "at_home == 1 and total_vs_avg <= -3", "total"),
        ("Home underdogs", "kind == 'reg' and at_home == 1 and spread > 0", "ats"),
        ("Road underdogs of 7+", "kind == 'reg' and at_home == 0 and spread >= 7", "ats"),
        ("Favorites of 10+", "kind == 'reg' and spread <= -10", "ats"),
        ("Team off a bye", "kind == 'reg' and game_num > 1 and rest_days >= 13", "ats"),
        ("Short week against a rested opponent (rest edge -3 or worse)", "kind == 'reg' and rest_edge <= -3 and rest_days <= 8", "ats"),
        ("Division underdogs", "kind == 'reg' and div_game == 1 and spread > 0", "ats"),
        ("Off a loss by 14+", "kind == 'reg' and prev_margin <= -14", "ats"),
        ("Home underdogs of 3 to 7 outright (moneyline)", "kind == 'reg' and at_home == 1 and spread >= 3 and spread <= 7", "ml"),
    ],
    "nhl": [
        ("Total 5.5 or lower", "total <= 5.5", "total"),
        ("Total 6.5 or higher", "total >= 6.5", "total"),
        ("Both teams on a back-to-back", "kind == 'reg' and b2b == 1 and opp_b2b == 1", "total"),
        ("Season openers", "game_num == 1", "total"),
        ("Playoff games", "kind == 'post'", "total"),
        ("Rematch within 3 days", "prev_vs_opp_days <= 3", "total"),
        ("Total 1+ above the last meeting's total", "at_home == 1 and total_vs_prev >= 1", "total"),
        ("Total 1+ below the last meeting's total", "at_home == 1 and total_vs_prev <= -1", "total"),
        ("Total 0.5+ above both teams' scoring last season", "at_home == 1 and total_vs_avg >= 0.5", "total"),
        ("Total 0.5+ below both teams' scoring last season", "at_home == 1 and total_vs_avg <= -0.5", "total"),
        ("Rest edge of 2+ days (moneyline)", "kind == 'reg' and rest_edge >= 2 and rest_days <= 10", "ml"),
        ("Home team on a back-to-back, opponent rested (moneyline)", "kind == 'reg' and at_home == 1 and b2b == 1 and opp_b2b == 0", "ml"),
        ("Home underdogs (moneyline)", "at_home == 1 and win_prob < 0.5", "ml"),
        ("Road favorites (moneyline)", "at_home == 0 and win_prob > 0.5", "ml"),
        ("Heavy favorites, 70%+ no-vig (moneyline)", "win_prob >= 0.7", "ml"),
        ("Playoff favorites, 60%+ no-vig (moneyline)", "kind == 'post' and win_prob >= 0.6", "ml"),
    ],
}

# columns of the nflverse schedule row worth passing on when the game is not played yet
NFL_UPCOMING = ["week", "weekday", "hour", "roof", "temp", "wind", "div_game", "neutral", "playoff_round",
                "home_ml", "away_ml", "over_odds", "under_odds", "home_spread_odds", "away_spread_odds"]
CACHE_VERSION = "v2"


def history_frames(data_dir: Path = Path("data"), lg: str = "nba") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, team-game features), built from the cached raw files on first use."""
    from . import history
    hist = data_dir / "history"
    games_path = hist / f"games_{lg}.pkl"
    team_path = hist / f"team_games_{lg}_{CACHE_VERSION}.pkl"
    history.RAW = hist / "raw"
    root = history.raw_root(lg)
    newest = max((p.stat().st_mtime for p in (root / "espn_scoreboard", root / "games.csv") if p.exists()), default=0)
    for path in (games_path, team_path):                # raw files fetched since the cache was built: rebuild
        if path.exists() and path.stat().st_mtime < newest:
            path.unlink()
    if not games_path.exists():
        history.build_league(lg).to_pickle(games_path)
    games = pd.read_pickle(games_path)
    if not team_path.exists():
        A.team_games(games).to_pickle(team_path)
    return games, pd.read_pickle(team_path)


def verdict(r: A.Result) -> str:
    d = r.wins + r.losses
    if d < 30:
        return "too few games"
    if r.p_vs_breakeven < 0.01:
        return "profitable here; see the multiple-testing note"
    if r.p_vs_breakeven < 0.05:
        return "mildly above break-even (1 in 20 by luck)"
    if r.win_pct < 0.5:
        return "leans the other way"
    return "no edge visible"


def _fmt(r: A.Result) -> str:
    return f"{r.record()} ({r.win_pct:.0%})" if r.wins + r.losses else "-"


def _flip(r: A.Result) -> str:
    """The record of betting the other side of a two-sided angle (ignoring price)."""
    d = r.wins + r.losses
    return f"{r.losses}-{r.wins} ({r.losses / d:.0%})" if d else "-"


def _situation(console: Console, rows: pd.DataFrame, away: str, home: str) -> None:
    def n(x, f="{:.0f}"):
        return "?" if pd.isna(x) else f.format(x)
    for r in (rows[rows["at_home"] == 1].iloc[0], rows[rows["at_home"] == 0].iloc[0]):
        rec = "" if pd.isna(r["season_win_pct"]) else f", {r['season_win_pct']:.0%} wins so far"
        po = f", won {int(r['prev_po_wins'])} of {int(r['prev_po_games'])} playoff games last season" if r["prev_po_games"] else ""
        console.print(f"{r['team']} ({'home' if r['at_home'] else 'away'}): game #{n(r['game_num'])}, rest {n(r['rest_days'])} days"
                      f"{rec}{po}")
    r = rows[rows["at_home"] == 1].iloc[0]
    last = "none" if pd.isna(r.get("prev_vs_opp_days")) else \
        f"{'playoffs' if r['prev_vs_opp_post'] else 'regular season'}, {int(r['prev_vs_opp_days'])} days ago, total {r['prev_total']:g}"
    console.print(f"last meeting: {last} · total vs last meeting {n(r['total_vs_prev'], '{:+.1f}')} · "
                  f"total vs both teams' scoring last season {n(r['total_vs_avg'], '{:+.1f}')}")


def run(console: Console, league: str, away: str, home: str, when: date, spread: float | None, total: float | None,
        data_dir: Path, angle: str | None, bet: str, leave_one_out: bool,
        home_ml: float | None = None, away_ml: float | None = None,
        wind: float | None = None, temp: float | None = None) -> None:
    from . import history
    lg = _league(league)
    away, home = lg.canonical_abbr(away), lg.canonical_abbr(home)
    games, t = history_frames(data_dir, lg.key)
    extra: dict = {}
    if lg.key == "nfl":
        sched = history.upcoming_nfl(away, home, when)
        if sched:
            extra = {k: sched[k] for k in NFL_UPCOMING if k in sched and not pd.isna(sched[k])}
            spread = sched["home_spread_close"] if spread is None and not pd.isna(sched["home_spread_close"]) else spread
            total = sched["total_close"] if total is None and not pd.isna(sched["total_close"]) else total
            console.print(f"[dim]lines and context from the nflverse schedule (week {extra.get('week', '?')})[/]")
    if total is None:
        raise SystemExit("--total is required (no scheduled game with a posted total was found)")
    if home_ml is not None:
        extra["home_ml"] = home_ml
    if away_ml is not None:
        extra["away_ml"] = away_ml
    if wind is not None:
        extra["wind"] = wind
    if temp is not None:
        extra["temp"] = temp
    rows = A.upcoming(games, away, home, when, spread, total, **extra)

    count = len(t[t["kind"] != "pre"].drop_duplicates("game_id"))
    line = f"home spread {spread:+g}, " if spread is not None else ""
    console.print(f"[bold]{away} @ {home}[/] {when}  {line}total {total:g}   "
                  f"(history: {count:,} {lg.key.upper()} games, {int(t['season'].min())}-{int(t['season'].max())} seasons)")
    _situation(console, rows, away, home)

    table = Table(title="Situational angles", title_justify="left")
    for c in ("angle", "bet", "active", "record", "ROI", "games", "other side", "verdict"):
        table.add_column(c, justify="right" if c in ("ROI", "games") else "left", overflow="fold", min_width=12 if c == "angle" else None)
    for name, q, kind in LIBRARY[lg.key]:
        who = A.active_teams(rows, q)
        if kind == "total":
            u, o = A.evaluate(t, q, "under"), A.evaluate(t, q, "over")
            best, label, other = (u, "Under", o) if u.win_pct >= o.win_pct else (o, "Over", u)
            active = "[green]yes[/]" if who else "no"
            table.add_row(name, label, active, _fmt(best), f"{best.roi:+.0%}", str(best.n), _fmt(other), verdict(best))
        else:
            r = A.evaluate(t, q, kind)
            active = f"[green]{', '.join(who)}[/]" if who else "no"
            table.add_row(name, "ATS" if kind == "ats" else "ML", active, _fmt(r), f"{r.roi:+.0%}", str(r.n), _flip(r), verdict(r))
    console.print(table)
    console.print("[dim]Each row is one pre-written angle, but a total row shows whichever side did better, and "
                  f"there are {len(LIBRARY[lg.key])} rows: expect a few to look good by luck. In a shuffled-outcome "
                  "test the best of ~140 random angles with 12-15 games reached 79-92%.[/]")

    if angle:
        conds = [c.strip() for c in angle.split(" and ") if c.strip()]
        r = A.evaluate(t, angle, bet)
        console.print(f"\n[bold]Your angle ({bet})[/]: {angle}")
        console.print(f"  record {r.record()} ({r.win_pct:.0%}), {r.n} games over {r.seasons} seasons, ROI {r.roi:.0%}, "
                      f"last 10 {r.last10[0]}-{r.last10[1]}, P(this good by chance) = {r.p_vs_coin:.3f}")
        who = A.active_teams(rows, angle)
        console.print(f"  active for this game: {'[green]yes (' + ', '.join(who) + ')[/]' if who else 'no'}")
        if leave_one_out and len(conds) > 1:
            lo = Table(title="Drop one condition at a time", title_justify="left")
            for c in ("without", "record", "win %", "games", "extra games that lost"):
                lo.add_column(c, justify="right" if c != "without" else "left")
            for c, rr in A.leave_one_out(t, conds, bet):
                lo.add_row(c, rr.record(), f"{rr.win_pct:.0%}", str(rr.n), str(rr.losses - r.losses))
            console.print(lo)
            console.print("[dim]If dropping a condition barely changes the record, it did nothing. If dropping one adds many "
                          "games and the win rate falls to ~50%, that condition is doing all the work on a few games.[/]")

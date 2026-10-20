"""Does Polymarket lag the sportsbooks? Lead-lag study on NBA spread and total markets.

Input is data/lag/timeline.jsonl from tools/line_poller.py (Pinnacle every ~30 minutes, DraftKings every ~5) and
Polymarket's own price history from the CLOB, fetched here. For each game and market kind we pick the Polymarket line
nearest Pinnacle's last pre-tip line, express Pinnacle's no-vig price at that exact line with the repo's score model,
and compare bar-to-bar changes:

    d_pm[i] = a1 * d_book[i-1] + a0 * d_book[i] + a_lead * d_book[i+1] + e

a1 > 0 means Polymarket was still catching up on the previous bar's sportsbook move (it lags); a_lead > 0 means it
moved before the book did (it leads). The level gap (Polymarket minus book) and how much of it is gone by tip-off
come with it. Confidence intervals resample whole games.

    python tools/lag_study.py            # reads data/lag/timeline.jsonl, writes data/lag/results.json and rows.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from pm_edges.config import Config
from pm_edges.fair import DEVIG, american_to_prob, model_for
from pm_edges.teams import league

NBA = league("nba")
DATA = Path("data/lag")
UA = {"User-Agent": "Mozilla/5.0 (research; pm_edges)"}
ET = ZoneInfo("America/New_York")


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def get(url, params):
    for _ in range(3):
        try:
            r = httpx.get(url, params=params, headers=UA, timeout=45)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
    return None


def pm_event(away: str, home: str, tip: datetime) -> dict | None:
    local = tip.astimezone(ET).date()
    for a in {away.lower()}:
        for h in {home.lower(), {"GSW": "gs", "NOP": "no", "SAS": "sa", "NYK": "ny"}.get(home, home).lower()}:
            for day in (local, local + timedelta(days=1)):
                d = get("https://gamma-api.polymarket.com/events", {"slug": f"nba-{a}-{h}-{day:%Y-%m-%d}"})
                if d:
                    return d[0]
    return None


def pm_history(token: str, start: datetime, end: datetime) -> list[dict]:
    d = get("https://clob.polymarket.com/prices-history", {"market": token, "startTs": int(start.timestamp()), "endTs": int(end.timestamp()), "fidelity": 1})
    return (d or {}).get("history", [])


def pm_price_at(hist: list[dict], ts: float, max_age: float = 1200) -> float | None:
    last = None
    for h in hist:
        if h["t"] <= ts:
            last = h
        else:
            break
    return last["p"] if last and ts - last["t"] <= max_age else None


def fair_at(row: dict, kind: str, home_name: str) -> tuple[float, float] | None:
    """(line, no-vig probability) for the home team's spread or for the over."""
    outs = row.get("spreads" if kind == "spreads" else "totals")
    if not outs or len(outs) != 2:
        return None
    fair = DEVIG["power"]([american_to_prob(o["price"]) for o in outs])
    for o, p in zip(outs, fair):
        if kind == "spreads" and NBA.team_from_text(o["name"]) == home_name:
            return float(o["point"]), p
        if kind == "totals" and o["name"] == "Over":
            return float(o["point"]), p
    return None


def team_is_home(question: str, home: str) -> bool:
    return NBA.team_from_text(question.split(":")[1].split("(")[0]) == home


def build_rows(min_bars: int = 3) -> pd.DataFrame:
    cfg = Config().apply_league("nba")
    models = {"spreads": model_for("spread", cfg), "totals": model_for("total", cfg)}
    pins = [json.loads(l) for l in (DATA / "timeline.jsonl").read_text().splitlines() if '"src": "pinnacle"' in l]
    games: dict[tuple, list[dict]] = {}
    for r in pins:
        if r.get("home") and r.get("away"):
            games.setdefault((r["away"], r["home"], r["commence"]), []).append(r)
    rows = []
    for (away, home, commence), series in games.items():
        tip = utc(commence)
        series = sorted((s for s in series if utc(s["ts"]) < tip - timedelta(minutes=5)), key=lambda s: s["ts"])
        if len(series) < min_bars:
            continue
        ev = pm_event(away, home, tip)
        if ev is None:
            continue
        home_name = NBA.full_name(home)
        t_start, t_end = utc(series[0]["ts"]) - timedelta(hours=1), min(tip, datetime.now(timezone.utc))
        for kind in ("spreads", "totals"):
            ref = next((fair_at(s, kind, home) for s in reversed(series) if fair_at(s, kind, home)), None)
            if ref is None:
                continue
            ms = [m for m in ev["markets"] if m.get("sportsMarketType") == kind and m.get("line") is not None and m.get("acceptingOrders", True) is not False]
            def line_of(m): return (m["line"] if team_is_home(m["question"], home) else -m["line"]) if kind == "spreads" else m["line"]
            ms = [m for m in ms if abs(line_of(m) - ref[0]) <= 1.0]
            if not ms:
                continue
            m = max(ms, key=lambda m: m.get("volumeNum") or 0)
            line = line_of(m)
            flip = kind == "spreads" and not team_is_home(m["question"], home)
            hist = pm_history(json.loads(m["clobTokenIds"])[0], t_start, t_end)
            for s in series:
                f = fair_at(s, kind, home)
                ts = utc(s["ts"])
                pm_p = pm_price_at(hist, ts.timestamp())
                if f is None or pm_p is None:
                    continue
                rows.append({"game": f"{away}@{home}", "kind": kind, "ts": s["ts"], "market_id": str(m["id"]), "hours_to_tip": (tip - ts).total_seconds() / 3600,
                             "pm_line": line, "pin_line": f[0], "volume": m.get("volumeNum"),
                             "book": models[kind].convert(f[0], f[1], line), "pm": (1 - pm_p) if flip else pm_p})
    return pd.DataFrame(rows)


def load_books() -> pd.DataFrame:
    """Polymarket order-book quotes the poller recorded (best bid/ask and the average price of a $50 fill)."""
    path = DATA / "timeline.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if '"src": "pm_book"' in l]
    return pd.DataFrame(rows)


def gap_rule(rows: pd.DataFrame, books: pd.DataFrame, threshold: float = 0.02, window_min: float = 6.0, fee: float = 0.0) -> pd.DataFrame:
    """Buy whichever side of a Polymarket market is cheaper than Pinnacle's no-vig price by at least `threshold`,
    at the average price of a $50 fill, once per game and market kind (the first bar that qualifies). Score each
    entry by closing-line value: Pinnacle's last pre-tip fair value for that side minus what it cost."""
    if rows.empty or books.empty:
        return pd.DataFrame()
    books = books.assign(t=pd.to_datetime(books["ts"], utc=True))
    out = []
    for (game, kind), d in rows.sort_values("ts").groupby(["game", "kind"]):
        d = d.reset_index(drop=True)
        close = d["book"].iloc[-1]
        for _, r in d.iloc[:-1].iterrows():
            t = pd.Timestamp(r["ts"]).tz_convert("UTC") if pd.Timestamp(r["ts"]).tzinfo else pd.Timestamp(r["ts"], tz="UTC")
            q = books[(books["market_id"].astype(str) == r["market_id"]) & ((books["t"] - t).abs() <= pd.Timedelta(minutes=window_min))]
            if q.empty:
                continue
            q = q.iloc[(q["t"] - t).abs().argmin():][:1].iloc[0]
            options = []
            if q["buy_filled"] and q["buy_avg"] is not None and not pd.isna(q["buy_avg"]):
                options.append(("yes", q["buy_avg"] + fee, r["book"] - q["buy_avg"] - fee, close))
            if q["sell_filled"] and q["sell_avg"] is not None and not pd.isna(q["sell_avg"]):
                options.append(("no", 1 - q["sell_avg"] + fee, q["sell_avg"] - r["book"] - fee, 1 - close))
            best = max(options, key=lambda o: o[2], default=None)
            if best and best[2] >= threshold:
                out.append({"game": game, "kind": kind, "ts": r["ts"], "side": best[0], "cost": best[1], "edge_at_entry": best[2],
                            "fair_close": best[3], "clv": best[3] - best[1], "hours_to_tip": r["hours_to_tip"]})
                break
    return pd.DataFrame(out)


def summarize_trades(t: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    if t.empty:
        return {"trades": 0}
    rng = np.random.default_rng(seed)
    games = t["game"].unique()
    by = {g: t[t["game"] == g]["clv"].to_numpy() for g in games}
    means = [np.concatenate([by[g] for g in rng.choice(games, len(games))]).mean() for _ in range(n_boot)]
    return {"trades": int(len(t)), "games": int(len(games)), "mean_edge_at_entry": float(t["edge_at_entry"].mean()),
            "mean_clv": float(t["clv"].mean()), "clv_ci95": np.percentile(means, [2.5, 97.5]).tolist(),
            "share_beating_close": float((t["clv"] > 0).mean()),
            "share_of_edge_that_held": float(t["clv"].mean() / t["edge_at_entry"].mean())}


def ols(y, X):
    return np.linalg.lstsq(np.column_stack(X), y, rcond=None)[0]


def analyze() -> dict:
    df = build_rows()
    if df.empty:
        raise SystemExit("no rows yet: let the poller run through some games first")
    df = df[df["volume"].fillna(0) >= 1000].sort_values(["game", "kind", "ts"]).reset_index(drop=True)      # thin markets carry stale quotes
    if df.empty:
        raise SystemExit("no market with at least $1,000 of volume yet")
    deltas = []
    for (g, k), d in df.groupby(["game", "kind"]):
        d = d.reset_index(drop=True)
        t = pd.to_datetime(d["ts"])
        ok = t.diff().dt.total_seconds().between(60, 45 * 60)                       # consecutive bars only
        d["d_pm"], d["d_book"] = d["pm"].diff().where(ok), d["book"].diff().where(ok)
        d["d_book_prev"], d["d_book_next"] = d["d_book"].shift(1), d["d_book"].shift(-1)
        d["gap"] = d["pm"] - d["book"]
        d["gap_last"] = d["gap"].iloc[-1]
        deltas.append(d)
    p = pd.concat(deltas, ignore_index=True)
    reg = p.dropna(subset=["d_pm", "d_book", "d_book_prev", "d_book_next"])
    out = {"bars": int(len(df)), "games": int(df["game"].nunique()), "regression_rows": int(len(reg)),
           "mean_abs_gap": float(p["gap"].abs().mean()), "mean_abs_gap_last_bar": float(p.groupby(["game", "kind"])["gap"].last().abs().mean()),
           "book_moves": int((p["d_book"].abs() > 0.01).sum())}
    if len(reg) >= 30:
        cols = ["d_book_prev", "d_book", "d_book_next"]
        beta = ols(reg["d_pm"].to_numpy(), [reg[c].to_numpy() for c in cols])
        rng = np.random.default_rng(0)
        names = reg["game"].unique()
        by = {g: reg[reg["game"] == g] for g in names}
        draws = []
        for _ in range(2000):
            s = pd.concat([by[g] for g in rng.choice(names, len(names))])
            draws.append(ols(s["d_pm"].to_numpy(), [s[c].to_numpy() for c in cols]))
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
        out["lead_lag"] = {c: {"coef": float(b), "ci95": ci[:, i].tolist()} for i, (c, b) in enumerate(zip(["lag_1_bar", "same_bar", "lead_1_bar"], beta))}
    big = p[p["d_book"].abs() >= 0.02]
    if len(big):
        out["after_big_book_move"] = {"n": int(len(big)), "pm_follows_same_bar": float((np.sign(big["d_pm"]) == np.sign(big["d_book"])).mean()),
                                      "share_of_move_same_bar": float((big["d_pm"] * np.sign(big["d_book"])).mean() / big["d_book"].abs().mean())}
    books = load_books()
    out["gap_rule"] = {}
    for thr in (0.01, 0.02, 0.03):
        out["gap_rule"][f"{thr:.2f}"] = summarize_trades(gap_rule(df, books, thr))
    p.to_csv(DATA / "rows.csv", index=False)
    (DATA / "results.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(analyze(), indent=1))

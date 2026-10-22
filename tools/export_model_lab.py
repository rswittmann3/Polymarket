"""Flatten data/model/{results,slate}.json into the row files the Model Lab dashboard reads (data/model/lab/*.json)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

NAMES = {"nfl": "NFL", "nba": "NBA", "nhl": "NHL"}


def rows(results: dict, slate: dict) -> dict[str, list[dict]]:
    verdicts, seasons, bets, calib, slopes, imps = [], [], [], [], [], []
    for lg, L in results["leagues"].items():
        for task, ev in L["tasks"].items():
            if not ev.get("available"):
                continue
            m = ev["models"]["avg"]
            base = ev.get("baseline_mse") or ev.get("baseline_loss") or 1.0
            key = f"{lg}|{task}"
            verdicts.append({"id": key, "league": NAMES[lg], "model": ev["label"], "verdict": ev["status"]["label"],
                             "tone": ev["status"]["tone"], "improvement_pct": 100 * m["gain"] / base, "t_stat": m["t"],
                             "seasons_up": m["seasons_positive"], "seasons": m["seasons"], "games": ev["n"],
                             "placebo_t_max": max(ev["placebo_t"] or [0], key=abs), "first_season": ev["seasons"][0], "last_season": ev["seasons"][1]})
            for s, g in m["by_season"].items():
                seasons.append({"id": f"{key}|{s}", "series": key, "label": f"{NAMES[lg]} {ev['label']}", "season": int(s),
                                "improvement_pct": 100 * g / base})
            for b in m["bets"]:
                bets.append({"id": f"{key}|{b['top']}", "series": key, "league": NAMES[lg], "model": ev["label"], "top_share": b["top"],
                             "bets": b["bets"], "win_pct": 100 * b["win_pct"] if b.get("win_pct") is not None else None, "roi_pct": 100 * b["roi"] if b.get("roi") is not None else None,
                             "roi_se_pct": 100 * b["roi_se"] if b.get("roi_se") is not None else None,
                             "clv_points": b.get("clv_pts"), "clv_se": b.get("clv_se"), "toward_pct": 100 * b["toward"] if b.get("toward") is not None else None})
            total = sum(max(i["importance"], 0) for i in ev["importances"]) or 1
            for i in ev["importances"]:
                imps.append({"id": f"{key}|{i['feature']}", "series": key, "league": NAMES[lg], "model": ev["label"],
                             "feature": i["feature"], "share_pct": 100 * max(i["importance"], 0) / total})
        for c in L["calibration"].get("win_calibration", []):
            calib.append({"id": f"{lg}|{c['bin']}", "league": NAMES[lg], "bin": c["bin"], "games": c["n"],
                          "predicted_pct": 100 * c["predicted"], "actual_pct": 100 * c["actual"]})
        for rel, d in L["calibration"].items():
            if isinstance(d, dict) and "slope" in d:
                slopes.append({"id": f"{lg}|{rel}", "league": NAMES[lg], "relation": rel, "slope": d["slope"], "intercept": d["intercept"],
                               "games": d["n"], "sd": d["sd"]})
    from pm_edges.angles import bh_adjust
    qs = bh_adjust([math.erfc(abs(v["t_stat"]) / math.sqrt(2)) for v in verdicts])        # two-sided p from the t-stat
    for v, q in zip(verdicts, qs):
        v["q_value"] = q
    games, injuries, goalies = [], [], []
    for i, g in enumerate(slate["games"]):
        gid = f"{g['league']}|{g['date']}|{g['away']}|{g['home']}"
        for j, r in enumerate(g.get("injuries", [])[:10]):
            injuries.append({"id": f"{gid}|{j}", "league": NAMES[g["league"]], "date": g["date"], "away": g["away"], "home": g["home"],
                             "team": r["team"], "player": r["name"], "pos": r["pos"], "status": r["status"],
                             "minutes": r.get("minutes"), "points": r.get("points"), "key": "yes" if r["key"] else "", "reported": (r.get("date") or "")[:10],
                             "note": (r.get("note") or "")[:160]})
        for gk in g.get("goalies", []):
            goalies.append({"id": f"{gid}|{gk['team']}", "league": NAMES[g["league"]], "date": g["date"], "away": g["away"], "home": g["home"],
                            "team": gk["team"], "goalie": gk["name"], "save_pct": gk.get("save_pct"), "gaa": gk.get("gaa"), "games": gk.get("games")})
        p = g["preds"]
        games.append({"id": f"{g['league']}|{g['date']}|{g['away']}|{g['home']}", "league": NAMES[g["league"]], "date": g["date"],
                      "kickoff": g.get("kickoff"), "away": g["away"], "home": g["home"], "total": g.get("total"), "total_open": g.get("total_open"),
                      "home_spread": g.get("home_spread") if g["league"] != "nhl" else None,
                      "home_spread_open": g.get("home_spread_open") if g["league"] != "nhl" else None,
                      "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml"),
                      "model_total": g.get("model_total"), "model_home_spread": g.get("model_home_spread"),
                      "pred_close_total": g.get("pred_close_total"), "pred_close_spread": g.get("pred_close_spread"),
                      "total_status": (p.get("total") or {}).get("status"), "spread_status": (p.get("side") or {}).get("status"),
                      "move_status": (p.get("move_total") or p.get("move_side") or {}).get("status")})
    return {"injuries": injuries, "goalies": goalies, "verdicts": verdicts, "seasons": seasons, "bets": bets, "calibration": calib, "slopes": slopes, "importances": imps, "slate": games}


def ledger_rows(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Paper trades and a one-row CLV summary from the pm_edges database (empty lists when nothing is recorded)."""
    import sqlite3
    from pm_edges.config import Config
    from pm_edges.ledger import Ledger
    cfg = Config(data_dir=data_dir)
    if not cfg.db_path.exists():
        return [], []
    led = Ledger(sqlite3.connect(cfg.db_path))
    rows = [{"id": str(r["id"]), "game": r["game_id"], "bet": f"{r['question']} / {r['outcome']}", "entered": r["entered_at"][:16].replace("T", " "),
             "ask": r["ask"], "fair_now": r["fair_last"], "clv_points": 100 * (r["clv"] if r["clv"] is not None else (r["fair_last"] - r["ask"]) if r["fair_last"] is not None else 0),
             "status": r["status"], "result": r["result"], "pnl": r["pnl"]} for r in led.rows()]
    if not rows:
        return [], []
    s = led.summary()
    pct = lambda v: None if v is None else 100 * v
    return rows, [{"id": "summary", "entries": s.entries, "settled": s.settled, "mean_clv_points": pct(s.mean_clv), "clv_se_points": pct(s.clv_se),
                   "t_stat": s.clv_t, "beat_close_pct": pct(s.beat_close), "roi_pct": pct(s.roi), "pnl": s.pnl}]


def lag_rows(data_dir: Path) -> list[dict]:
    """A few plain measures from the Polymarket-vs-Pinnacle study (empty when it has not run)."""
    path = data_dir / "lag" / "results.json"
    if not path.exists():
        return []
    r = json.loads(path.read_text())
    g1 = (r.get("gap_rule") or {}).get("0.01", {})
    rows = [("games", "Games compared", r["games"], "Spread and total markets, Polymarket vs Pinnacle"),
            ("bars", "Price comparisons", r["bars"], "Each is one poll of both markets before tip-off"),
            ("book_moves", "Pinnacle line moves observed", r["book_moves"], "Too few to say who moves first"),
            ("gap", "Mean gap, percentage points", 100 * r["mean_abs_gap"], "Polymarket price minus Pinnacle's no-vig price, absolute"),
            ("trades", "Gap-rule trades at 1 point", g1.get("trades", 0), "Buy when Polymarket is at least a point cheaper than Pinnacle")]
    return [{"id": k, "measure": m, "value": v, "note": n} for k, m, v, n in rows]


def forward_rows(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Frozen-prediction scores (tools/score_frozen.py): per-game detail and a summary row per question."""
    path = data_dir / "model" / "forward.json"
    if not path.exists():
        return [], []
    r = json.loads(path.read_text())
    detail, summary = [], []
    for kind, st in r.items():
        for label, key in (("From the opening line", "from_open"), ("From the line when predicted", "from_then")):
            v = st[key]
            summary.append({"id": f"{kind}|{key}", "question": "Total" if kind == "total" else "Spread", "basis": label, "games": st["games"],
                            "mean_clv_points": v["mean_clv_points"], "se": v["se"], "moved": v["moved"], "right": v["direction_right"]})
        for i, d in enumerate(st["detail"]):
            detail.append({"id": f"{kind}|{i}", "date": d["date"], "game": d["game"], "question": "Total" if kind == "total" else "Spread",
                           "open": d["open"], "then": d["then"], "predicted": d["pred"], "closed": d["close"]})
    return detail, summary


WHY = (("model-converted", "Estimated: the books do not quote this exact line"), ("edge below", "Gap is under the minimum edge"),
       ("Pinnacle does not", "Pinnacle does not confirm it"), ("one soft book", "Only one soft book disagrees"),
       ("liquidity", "Market too thin"), ("stale", "Quote looks stale"))


def scan_rows(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(edges, one summary row, per-game rows) from tools/scan_slate.py, all leagues scanned so far."""
    edges, games, scans = [], [], []
    for path in sorted((data_dir / "scan").glob("*.json")):
        r = json.loads(path.read_text())
        scans.append(r)
        for e in r["edges"]:
            why = next((v for k, v in WHY if k in (e.get("why_not") or "")), (e.get("why_not") or "")[:60])
            edges.append({"id": e["id"], "league": e["league"], "game": e["game"], "date": e["date"], "bet": e["bet"], "market": e["market"],
                          "ask": e["ask"], "fair": e["fair"], "edge_pts": e["edge_pts"], "pin_edge_pts": e["pin_edge_pts"], "books": e["books"],
                          "dollars": e["dollars_with_edge"] if e["dollars_with_edge"] is not None else e["dollars_at_ask"],
                          "liquidity": e["liquidity"], "status": e["status"], "confidence": e["confidence"], "why": "" if e["status"] == "flagged" else why})
        games += [{"id": f"{r['league']}|{g['game']}", "league": r["league"].upper(), **{k: g[k] for k in ("game", "markets_checked", "outcomes_priced", "flagged")}}
                  for g in r["games"]]
    if not scans:
        return [], [], []
    if not edges:                      # an empty file would not load; one inert row keeps the dataset readable
        edges = [{"id": "none", "league": "", "game": "", "date": "", "bet": "", "market": "", "ask": None, "fair": None, "edge_pts": 0,
                  "pin_edge_pts": None, "books": 0, "dollars": None, "liquidity": 0, "status": "none", "confidence": "low", "why": ""}]
    flagged = [e for e in edges if e["status"] == "flagged"]
    summary = [{"id": "scan", "scanned_at": min(r["scanned_at"] for r in scans)[:16].replace("T", " "), "games": len(games),
                "markets": sum(g["markets_checked"] for g in games), "flagged": len(flagged),
                "best_flagged": max((e["edge_pts"] for e in flagged), default=None),
                "best_gap": max((e["edge_pts"] for e in edges), default=None),
                "min_edge": scans[0]["min_edge_pts"], "max_stake": scans[0]["max_stake"]}]
    return edges, summary, games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    a = ap.parse_args()
    results = json.loads((a.data_dir / "model" / "results.json").read_text())
    slate = json.loads((a.data_dir / "model" / "slate.json").read_text())
    out = a.data_dir / "model" / "lab"
    out.mkdir(parents=True, exist_ok=True)
    out_rows = rows(results, slate)
    out_rows["ledger"], out_rows["clv_summary"] = ledger_rows(a.data_dir)
    out_rows["lag_summary"] = lag_rows(a.data_dir)
    out_rows["forward"], out_rows["forward_summary"] = forward_rows(a.data_dir)
    out_rows["scan_edges"], out_rows["scan_summary"], out_rows["scan_games"] = scan_rows(a.data_dir)
    for name, r in out_rows.items():
        (out / f"{name}.json").write_text(json.dumps(r))
        print(name, len(r))
    print("generated", results["generated"])


if __name__ == "__main__":
    main()

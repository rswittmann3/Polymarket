"""Flatten data/model/{results,slate}.json into the row files the Model Lab dashboard reads (data/model/lab/*.json)."""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    a = ap.parse_args()
    results = json.loads((a.data_dir / "model" / "results.json").read_text())
    slate = json.loads((a.data_dir / "model" / "slate.json").read_text())
    out = a.data_dir / "model" / "lab"
    out.mkdir(parents=True, exist_ok=True)
    for name, r in rows(results, slate).items():
        (out / f"{name}.json").write_text(json.dumps(r))
        print(name, len(r))
    print("generated", results["generated"])


if __name__ == "__main__":
    main()

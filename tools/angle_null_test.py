"""Is an angle's record better than what a search for angles finds in pure noise?

Runs the same random-angle search on the real over/under results of season openers and on
copies where the outcomes were shuffled between games (so any pattern is chance). If the
shuffled copies find angles as good as the real data's best, the real best proves nothing.

    python tools/angle_null_test.py [--shuffles 3] [--trials 5000]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pm_edges import angles as A


def best_of(frame: pd.DataFrame, trials: int, seed: int) -> tuple[int, float, int, int]:
    out = []
    for bet in ("under", "over"):
        r = A.null_search(frame, bet, conditions=6, trials=trials, n_min=11, n_max=17, seed=seed,
                          base="at_home == 1 and season > 2005")
        out += [(w, l) for w, l, _ in r["records"] if 12 <= w + l <= 15]
    wp = [w / (w + l) for w, l in out]
    return len(out), (max(wp) if wp else float("nan")), sum(x >= 0.8 for x in wp), sum(x >= 0.85 for x in wp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    t = pd.read_pickle(args.data_dir / "history" / "team_games.pkl")
    openers = t[(t["game_num"] == 1) & (t["kind"] == "reg")].copy()
    t0 = time.time()
    n, best, ge80, ge85 = best_of(openers, args.trials, 7)
    print(f"REAL data   : {n:>4} angles with 12-15 games, best {best:.0%}, >=80%: {ge80}, >=85%: {ge85}  ({time.time() - t0:.0f}s)")
    rng = np.random.default_rng(0)
    for k in range(args.shuffles):
        sh = openers.copy()
        g = sh.drop_duplicates("game_id")[["game_id", "over"]].copy()
        g["over"] = rng.permutation(g["over"].values)
        sh = sh.drop(columns="over").merge(g, on="game_id")
        n, best, ge80, ge85 = best_of(sh, args.trials, 8 + k)
        print(f"SHUFFLED #{k + 1}: {n:>4} angles with 12-15 games, best {best:.0%}, >=80%: {ge80}, >=85%: {ge85}")


if __name__ == "__main__":
    main()

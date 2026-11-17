"""Is Novig's EARLY spread stale relative to the line-move model? For every frozen NBA game with Novig coverage:
Novig fair home spread at (a) the earliest logged quote, (b) the freeze time, (c) the last pre-tip quote; vs the model's predicted close and the DK open.
Gain = move toward the model direction, in points, measured from each starting point to Novig's own close."""
import sys, glob, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from datetime import datetime, timezone
from tools import injury_lag as IL

df, ev, mk = IL.load_novig()
rows = []
for f in sorted(glob.glob("snapshots/frozen/nba_*.json")):
    d = json.load(open(f)); ft = datetime.strptime(f.split("_")[-1][:13], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000
    for g in d["games"] if "games" in d else d:
        for e, (away, home, st) in ev.items():
            if (away, home) == (g["away"], g["home"]) and abs(st - pd.Timestamp(g["kickoff"]).timestamp() * 1000) < 3 * 3600e3:
                rows.append(dict(game=f"{away}@{home}", date=g["date"], event=e, st=st, ft=ft, dk_open=g["home_spread_open"], dk_then=g["home_spread"], pred=g["pred_close_spread"]))
G = pd.DataFrame(rows).drop_duplicates(["event", "ft"])
print("frozen games with Novig data:", len(G), "| distinct games:", G.event.nunique())
out = []
for _, r in G.iterrows():
    sub = df[df.m.map(lambda m: mk[m]["event"] == r.event)]
    if sub.empty: continue
    t0 = int(sub.ts.min()); times = [t0 + 600e3, int(r.ft), int(r.st - 60e3)]
    ln = IL.fair_line_series(df, mk, r.event, [int(t) for t in times])
    if None in ln: continue
    out.append({**r.to_dict(), "nv_early": ln[0], "nv_freeze": ln[1], "nv_close": ln[2], "hrs_early_to_tip": (r.st - t0) / 3.6e6})
O = pd.DataFrame(out); O = O[O.hrs_early_to_tip > 6]
if O.empty: raise SystemExit("not enough coverage")
print("games used:", len(O), "| median hours from first Novig quote to tip: %.1f" % O.hrs_early_to_tip.median())
print("Novig early minus DK open (points, home-spread sign): mean %+.2f  sd %.2f" % ((O.nv_early - O.dk_open).mean(), (O.nv_early - O.dk_open).std()))
print("Novig close minus DK close(then->close unknown) skipped")
for lab, start in (("from Novig earliest", "nv_early"), ("from Novig at freeze", "nv_freeze")):
    sig = np.sign(O.pred - O[start]); mv = O.nv_close - O[start]; gain = sig * mv; big = (O.pred - O[start]).abs() >= 2
    m = O[(sig != 0) & (mv != 0)]; g2 = (sig * mv)[m.index]
    print(f"{lab:22s} n={len(O)}  all: mean gain {gain.mean():+.2f} pts (se {gain.std()/np.sqrt(len(gain)):.2f}) | moved games {len(m)} dir right {(g2>0).sum()} | |pred-start|>=2: n={big.sum()} mean {gain[big].mean() if big.sum() else float('nan'):+.2f}")
print(O[["game", "date", "dk_open", "nv_early", "nv_freeze", "nv_close", "pred"]].round(2).to_string(index=False))

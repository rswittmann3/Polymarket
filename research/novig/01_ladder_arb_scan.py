"""Model-free structural arbitrage scan of Novig NBA quotes.

Every two-outcome market outcome is rewritten as an event on a numeric variable: "V > c" or "V < c" (half-point thresholds, so no pushes).
Two outcomes whose events together cover every value of V pay at least $1 for sure; if their asks (ask_i = 1 - bid of the other outcome) sum to < 1, it is an arbitrage.
Covering pairs scanned:  V>c1 with V<c2 (c2 > c1) on the same variable; combo>c1 with component<c2 (c1 <= c2);  game total >c1 with team total / 1H total <c2 (c1 <= c2).
Spreads are mapped to the home margin M = home - away.  Reconstructs the book state every STEP minutes from the touch logs and counts violations.
"""
import glob, gzip, json, re, sys, collections
import numpy as np, pandas as pd

STEP = 600_000
COMBOS = {"POINTS_REBOUNDS_ASSISTS": {"POINTS", "REBOUNDS", "ASSISTS", "POINTS_REBOUNDS", "POINTS_ASSISTS", "REBOUNDS_ASSISTS"},
          "POINTS_REBOUNDS": {"POINTS", "REBOUNDS"}, "POINTS_ASSISTS": {"POINTS", "ASSISTS"}, "REBOUNDS_ASSISTS": {"REBOUNDS", "ASSISTS"}}

def rows(paths):
    for p in paths:
        with gzip.open(p, "rt") as f:
            for l in f: yield json.loads(l)

def load(dirs):
    meta, ev, touches = {}, {}, []
    for d in dirs:
        for r in rows(sorted(glob.glob(f"{d}/meta_*.jsonl.gz"))):
            if r.get("k") == "mk": meta[r["m"]] = r
            elif r.get("k") == "ev": ev[r["e"]] = r
        touches += [r for r in rows(sorted(glob.glob(f"{d}/touch_*.jsonl.gz"))) if r.get("lg") == "NBA"]
    touches.sort(key=lambda r: r["ts"])
    return meta, ev, touches

def teams(meta):
    t = {}
    for r in meta.values():
        if r["mt"] == "TOTAL":
            m = re.match(r"(\S+) @ (\S+) t", r["d"])
            if m: t[r["e"]] = (m.group(1), m.group(2))
    return t

def describe(mk, tm):
    """-> list of two (var, op, c) for outcome 0 and 1, or None. var is a tuple key identifying the numeric variable."""
    mt, e, n = mk["mt"], mk["e"], mk["n"]
    try: s = float(mk["s"])
    except (TypeError, ValueError): return None
    if s == int(s): return None
    if mt == "TOTAL": return [(("TOT", e), ">", s), (("TOT", e), "<", s)]
    if mt == "SPREAD" and e in tm:
        away, home = tm[e]; out = []
        for nm in n:                                                    # "home +h" wins iff M > -h ; "away +h" wins iff M < h (h signed as written)
            team, line = nm.split()[0], float(nm.split()[-1])
            if team not in (home, away): return None
            out.append((("M", e), ">", -line) if team == home else (("M", e), "<", line))
        return out
    if mt in ("POINTS", "REBOUNDS", "ASSISTS", "THREE_POINTERS_MADE", "STEALS", "BLOCKS") or mt in COMBOS:
        player = mk["d"].rsplit(" ", 2)[0]
        return [((mt, e, player), ">", s), ((mt, e, player), "<", s)]
    if mt == "TOTAL_1H": return [((mt, e), ">", s), ((mt, e), "<", s)]
    if mt == "TEAM_TOTAL":                                             # a different quantity for each team
        team = mk["d"].rsplit(" ", 2)[0]
        return [((mt, e, team), ">", s), ((mt, e, team), "<", s)]
    return None

def main(dirs=("data/novig", "data/props"), save=True, quiet=False):
    meta, ev, touches = load(list(dirs))
    tm = teams(meta)
    desc = {m: describe(mk, tm) for m, mk in meta.items()}
    state = {}; i = 0
    t0, t1 = touches[0]["ts"], touches[-1]["ts"]
    found = []; samples = 0
    for t in list(range(t0, t1, STEP)) + [t1]:
        while i < len(touches) and touches[i]["ts"] <= t:
            r = touches[i]; state[r["m"]] = r; i += 1
        live = [(m, r) for m, r in state.items() if desc.get(m) and r["st"] > t]
        if not live: continue
        samples += 1
        legs = collections.defaultdict(list)                          # variable -> legs (op, c, ask, qty, market)
        for m, r in live:
            d = desc[m]
            for k in (0, 1):
                bid_other = r["b"][1 - k]; q_other = r["q"][1 - k]
                if bid_other is None: continue
                legs[d[k][0]].append((d[k][1], d[k][2], 1 - float(bid_other), q_other, m, k))
        def scan(vars_hi, vars_lo, need):
            for a in vars_hi:
                for b in vars_lo:
                    cost = a[2] + b[2]
                    if cost < 1 - 1e-9 and a[4] != b[4]:
                        found.append({"t": t, "cost": cost, "edge": 1 - cost, "qty": min(a[3], b[3]), "a": (a[4], a[1], a[0]), "b": (b[4], b[1], b[0])})
        for var, L in legs.items():
            gt = [x for x in L if x[0] == ">"]; lt = [x for x in L if x[0] == "<"]
            for a in gt:
                for b in lt:
                    if b[1] > a[1]:
                        found.append({"t": t, "kind": "ladder", "var": var[0], "cost": a[2] + b[2], "edge": 1 - a[2] - b[2], "qty": min(a[3], b[3]), "lines": (a[1], b[1]), "markets": (a[4], b[4])})
        # combo > c1 with component < c2 (c1 <= c2), same event and player
        byp = collections.defaultdict(dict)
        for var, L in legs.items():
            if len(var) == 3: byp[(var[1], var[2])][var[0]] = L
        for key, stats in byp.items():
            for combo, comps in COMBOS.items():
                if combo not in stats: continue
                for comp in comps & stats.keys():
                    for a in (x for x in stats[combo] if x[0] == ">"):
                        for b in (x for x in stats[comp] if x[0] == "<" and x[1] >= a[1]):
                            if True:
                                found.append({"t": t, "kind": f"{combo}>/{comp}<", "var": combo, "cost": a[2] + b[2], "edge": 1 - a[2] - b[2], "qty": min(a[3], b[3]), "lines": (a[1], b[1]), "markets": (a[4], b[4]), "player": key[1]})
        # game total > c1 with team/1H total < c2
        bye = collections.defaultdict(dict)
        for var, L in legs.items():
            if var[0] in ("TOT", "TOTAL_1H", "TEAM_TOTAL"): bye[var[1]].setdefault(var[0], []).extend(L)
        for e, d in bye.items():
            for comp in ("TOTAL_1H", "TEAM_TOTAL"):
                for a in (x for x in d.get("TOT", []) if x[0] == ">"):
                    for b in (x for x in d.get(comp, []) if x[0] == "<" and x[1] >= a[1]):
                        if True:
                            found.append({"t": t, "kind": f"TOT>/{comp}<", "var": comp, "cost": a[2] + b[2], "edge": 1 - a[2] - b[2], "qty": min(a[3], b[3]), "lines": (a[1], b[1]), "markets": (a[4], b[4])})
    if not found:
        return pd.DataFrame()
    F = pd.DataFrame(found).drop(columns=[c for c in ("a", "b") if c in pd.DataFrame(found).columns], errors="ignore")
    if not quiet: print(f"snapshots scanned: {samples}  (step {STEP//60000} min, {pd.to_datetime(t0, unit='ms')} -> {pd.to_datetime(t1, unit='ms')})")
    if F.empty:
        print("no covering pairs at all"); return
    F["kind"] = F.kind.fillna("ladder")
    print("covering pairs observed:", len(F), "| crossed (edge > 0):", int((F.edge > 1e-9).sum()))
    print("\nby kind: edge = 1 - (ask1 + ask2); positive means a guaranteed profit per $1 of payout"); print(F.groupby("kind").agg(pairs=("edge", "size"), crossed=("edge", lambda x: int((x > 1e-9).sum())), best_edge=("edge", "max"), median_edge=("edge", "median"), median_qty=("qty", "median")).round(4).to_string())
    print("\nclosest to crossing:"); print(F.sort_values("edge", ascending=False).head(8)[["kind", "edge", "qty", "lines"]].to_string())
    if save: F.to_pickle("data/model/nba/ladder_arbs.pkl")
    return F

if __name__ == "__main__":
    main()

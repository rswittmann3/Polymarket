# What we tested, and what held up (Oct 2026 - Nov 17 2026)

Everything here is read-only research: no orders were placed. Reproduce with the scripts named in each line.

## Verdict
Public-data modelling of NBA spreads, totals and player props does not beat Novig/sportsbook prices. Every apparent edge was an artifact.
Passive market making on Novig NBA earns about zero against the close in the five days measured.

## Game lines (2008-2026 closing lines, ~24k games; research/nba/02,05,06,09)
- Possession-based team ratings (opponent-adjusted offence/defence/pace) vs the closing line: RMSE 14.15 vs 13.58 for the line; gap slope -0.036 (SE 0.044). No information.
- Player-impact ridge and rotation absences: the closing line already prices known absences (t = 1.2 / 0.3); absences explain ~7% of open-to-close move variance. Perfect foresight of who sits would be worth ~+1 pt/game from the open, before ~0.4-0.8 pt of cost.
- Rest, back-to-backs, travel, trailing form, altitude, early season, playoffs, referee crews, shooting luck: nothing survives multiple-testing correction.
- "Home favourites underperform the spread by 0.74 pts (t = -4.4)": artifact. ATS cover rates are 49.8% / 49.1%; the effect is a mean-vs-median skew and vanishes after 2020.
- "Big-spread games go Over": over rate at |spread| >= 6 is 50.5% (z = 1.1); breakeven is 52.4%. The earlier +1.1 pt was another mean-in-points artifact.

## Line-move model (tools/freeze_predictions.py, score_frozen.py; forward test since Oct 22)
- Gains +0.98 pts against the OPENING line (195 games, SE 0.21), but only +0.15 (SE 0.14) from the morning line, and +0.03 (SE 0.32, 26 games) from Novig's own earliest line (research/novig/02). Not tradable as far as measurable.
- Big-move flags (|predicted close - line| >= 2): 1 scored (+1.0). The last four flags were the model reverting to the open after an injury-driven move (it cannot see news).

## Player props (pm_edges/nba_props.py, tools/freeze_props.py, score_props.py; forward test Nov 11-15, 5 slates)
- Walk-forward 2023-25 the model beats a trailing average by 5-11% (Poisson deviance); teammate absences roughly double what form alone gives.
- Against Novig's mid on the same tight two-sided markets the model is WORSE: Brier gain -0.0087 (SE 0.0025, ~3.5 SE). Paper bets at the ask lose at every edge threshold. The forward test is retired.
- Bugs found on the way (worth remembering): role players without a market counted as absent; tree models cannot extrapolate (use ratio-to-baseline targets); raw negative-binomial tails were 6-7pp overconfident (isotonic calibrator fixed it); wide-quote "mids" are meaningless (filter by width).

## Cross-venue and structure
- Polymarket vs Novig on matched prop rungs: +1.1pp (SD 1.2pp) on tight Novig markets; the apparent +4pp gap was wide Novig quotes. Crossing Polymarket's books costs ~8pp at $5. (tools/xv_props.py)
- Model-free ladder / covering-pair arbitrage inside Novig: 11.5 million covering pairs over 6 days, none crossed. (research/novig/01; logged each watchdog pass in data/props/arb_scan.log)
- Injury-lag study: n = 1 usable game (84% of the move done by 15 min). Too thin to say anything.

## Market making (read-only; research/novig/03,04; tools/fetch_novig_trades.py)
Source: Novig's public daily trade files (every fill, anonymized) joined to my recorded quotes for market type, main/alt line and scheduled start. Maker = resting side;
result measured against the CLOSING price (VWAP of the last 10 min of pregame trades). Pregame has no fee for anyone; the maker credit (25% of the taker fee) exists only live.
First pass, Nov 11-15 (5 days, ~30 games):
- All pregame NBA makers combined earn +$1,837/day +/- $5,626 (market-clustered SE) on ~$5.1M/day taker notional: 3.6 bp, indistinguishable from zero. Daily pool ranged -$26.8k to +$17.7k.
- A 5% share would be ~$92/day +/- $281. Nearly every segment is within noise of 0c/contract; a few are positive (alt TOTAL 6-24h before start +0.56c +/- 0.26; 3PM main 1-6h +1.0c +/- 0.2 on the 60s markout). Segment results correlate across disjoint days (0.43, 24 segments), so some structure may be real; many segments were tested.
- Adverse selection roughly equals the half-spread captured. Live: credit pool ~$6k/day across all makers but live markouts are much worse (props -4 to -6c) and need fast in-game models. Resting orders are voided at tip.
- Not tested: Polymarket US maker economics; hedged / two-venue quoting.
Recheck planned ~Dec 1 with 2+ weeks of days (`python tools/fetch_novig_trades.py` runs from the watchdog; then run research/novig/03 and 04).

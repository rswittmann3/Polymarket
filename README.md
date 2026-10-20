# pm_edges — Polymarket vs. sportsbook fair-price checker (NFL)

A read-only command-line tool. For one game it:

1. lists every Polymarket market (moneyline, spreads, totals, player props, quarter/half markets) with the price you can actually buy at;
2. derives a fair probability for each one from sportsbook lines, giving sharp books (Pinnacle first) the most weight;
3. records snapshots so it can show line movement and other signals over time;
4. flags a market only when the fair probability beats the executable price, after Polymarket's taker fee, by a set margin. Otherwise it prints **"No edge found."**

It never places bets, trades or touches a wallet. **Check that using Polymarket and sportsbooks is legal where you live.**

> **Status:** first live run done on 2026-10-19 (WAS @ SF, about 70 minutes before kickoff). It found 494 Polymarket markets across 2 events and checked 341 liquid ones, using 48 Odds API credits. Result: "No edge found." Three markets were hand-checked against the raw responses. Real responses from that run are in `tests/fixtures/live_was_sf_2026-10-19` and are used as regression tests.

## Ground rules (built into the code)

- **Executable price, not `outcomePrices`.** `outcomePrices` is often a stale last trade. To buy outcome 0 ("Yes") you pay `bestAsk`; to buy outcome 1 ("No") you pay `1 - bestBid`. When the CLOB order book is reachable, each outcome token's own best ask is used instead, along with the size available there.
- **Liquidity filter.** Markets with `liquidityNum < 100` are ignored (`--min-liquidity`).
- **Raw JSON first.** Every API response is saved to `data/raw/<game>/<run>/` before parsing. `outcomes`, `outcomePrices` and `clobTokenIds` are JSON strings inside JSON and are decoded with `json.loads`.
- **Trend records are not inputs.** Records like "15-2 as a road dog" never enter the fair probability.
- **No invented vig.** A one-sided book market, such as only "Yes" quoted for an anytime TD, gets no fair value unless some book quotes both sides.
- **Factual language.** Output reads like "edge +3.1 pts vs consensus no-vig". Nothing is a lock.

## Setup

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then put your free key from https://the-odds-api.com in it
pytest                      # runs offline against tests/fixtures/synthetic
```

`.env`:

```
ODDS_API_KEY=your_key_here
```

## Usage

```bash
# 1. What does Polymarket list, and at what executable price?
pm_edges markets  --league nfl --away WAS --home SF --date 2026-10-19 --kickoff 20:15

# 2. Store a snapshot of both sources (run on a schedule on game day)
pm_edges snapshot --league nfl --away WAS --home SF --date 2026-10-19 --kickoff 20:15

# 3. Compare and flag
pm_edges edges    --league nfl --away WAS --home SF --date 2026-10-19 --kickoff 20:15 --min-edge 0.03 --html
```

`--date` is the **local** game date. Polymarket slugs (`nfl-was-sf-YYYY-MM-DD`) appear to use the UTC date, so a Monday-night game is usually listed under Tuesday. Both dates are always tried. `--kickoff` (local time, `--tz` default `America/New_York`) only decides which one is tried first.

Useful flags:

| flag | effect |
|---|---|
| `--offline` | re-run entirely from saved raw JSON (newest copy of each request); `--offline-run <folder>` picks one run |
| `--no-clob` | use Gamma `bestBid`/`bestAsk` only |
| `--all` (markets) | include markets below the liquidity filter |
| `--devig power\|multiplicative\|shin` | de-vig method (default `power`) |
| `--allow-model` | let model-converted lines be flagged (they always show) |
| `--fee-rate 0.05` | override the taker fee rate |
| `--bookmakers a,b,…` | Odds API books to ask for (default: 10 books incl. Pinnacle; 10 named books cost the same as 1 region) |
| `--regions us,eu` | use regions instead of named books (about twice the credits) |
| `--scope full\|halves\|all` | which periods to request odds for; fewer periods cost fewer credits |
| `--no-alternates` | never request alternate lines (saves credits) |
| `--no-history` | skip CLOB price history |
| `--snapshot` (edges) | also store this run as a snapshot |
| `--max-stake 50` | largest order the model considers, in dollars (default 50); caps the "size with edge" column and any Kelly stake |
| `--kelly --bankroll 1000` | show a quarter-Kelly stake, capped at 2% of bankroll and at `--max-stake`; off by default |
| `--html` / `--no-csv` | report formats; CSV is written by default to `data/reports/` |

### Scheduling snapshots

The Odds API free tier has about 500 credits a month. Each odds request costs **(markets returned) × (regions)**, where up to 10 named bookmakers count as one region. Only market keys for Polymarket markets that pass the liquidity filter are requested. Alternates are requested only for lines the main markets don't cover.

Measured on the 2026-10-19 game, with Polymarket listing game, half, quarter, team-total and player-prop markets:

| `--scope` | market keys | credits per run (approx.) |
|---|---|---|
| `all` (default) | 32 main + alternates | 48 (measured) |
| `halves` | 20 main + alternates | ~30 |
| `full` | 13 main + alternates | ~20 |

Credits remaining are printed after every run, read from the `x-requests-remaining` header. For repeated snapshots use `--scope full` or `halves`; at 48 credits a run, 500 credits is about 10 full runs.

Steam detection compares the last two snapshots and only runs when they are ≤10 min apart (`steam_window_min` in `config.py`). Schedule every 10 minutes if you want it.

cron (Linux/macOS), every 10 minutes from 13:00 to 20:50 on 2026-10-19:

```cron
*/10 13-20 19 10 * cd /path/to/Polymarket && .venv/bin/pm_edges snapshot --away WAS --home SF --date 2026-10-19 --kickoff 20:15 >> data/snapshot.log 2>&1
```

Windows Task Scheduler: create a task with a 10-minute repeat that runs
`C:\path\to\Polymarket\.venv\Scripts\pm_edges.exe snapshot --away WAS --home SF --date 2026-10-19 --kickoff 20:15`,
with "Start in" set to the repository folder, so `.env` and `data/` resolve.

The first snapshot of a game is treated as the "open".

## How the numbers are made

**Fair probability**

1. Each book's two-way market is de-vigged. The methods are *multiplicative* (`q/Σq`), *power* (`qᵏ` with k such that the sum is 1; the default, which takes more margin off longshots) and *Shin*.
2. The consensus is a weighted average across books. Pinnacle has weight 3. BetOnline and LowVig (`betonlineag`, `lowvig`) have weight 2. Everyone else has weight 1. Circa and Bookmaker are also weighted 2 in `config.py`, but the Odds API did not list them for NFL in October 2026. Edit the weights in `config.py`.
3. **Line mismatch**, for example Polymarket at 52.5 while books show 51.5:
   - First, an exact alternate line from the books (`alternate_spreads`, `alternate_totals`, `player_*_alternate`) is used. That counts as `fair_source = books`.
   - Otherwise, **full-game** spreads and totals are converted with a discretised normal model: margin σ 13.5 with extra mass on key numbers 3 and 7, total σ 10. These rows are labelled `fair_source = model (low conf.)` and never flag unless `--allow-model` is set.
   - Halves, quarters, team totals and player props are never model-converted, because those σ values only describe full games. They get no fair value, with a reason.

**Edge**

```
fee  = rate × p × (1 − p)        per share, taker only (Polymarket's published formula)
edge = fair − ask − fee
```

A row is flagged when `edge ≥ min_edge` (default 0.03), liquidity ≥ the threshold and the fair value is not model-derived. The fee rate comes from the market's `feeSchedule`. Live NFL markets on 2026-10-19 carried `{"rate": 0.03, "exponent": 1, "takerOnly": true}` with `feeType: sports_fees_nfl_cfb_oct26`. The rate is 0 when `feesEnabled` is false. If a market has no schedule, the tool falls back to `default_fee_rate = 0.05`, which is deliberately conservative. The footer always prints which rate was used.

**Signals** (context only; they never change fair values)

| signal | fires when |
|---|---|
| line move | the consensus main line or no-vig prob changed since the first snapshot (≥0.5 pt or ≥1 pt prob) |
| steam | ≥3 books moved the same direction between two snapshots ≤10 min apart |
| juice | same line, sides priced unevenly at a book (implied gap ≥2 pts); says which side is being paid for |
| divergence | Pinnacle no-vig vs soft-book average differs by >2 pts at the same line |
| PM drift | Polymarket price change over 1h/6h (CLOB `prices-history`, else own snapshots), and whether it moved toward or away from the books' fair value |
| rlm | only with a bet-split provider plugged into `signals.BetSplitsProvider`; none ships, and nothing is scraped |

## Output columns (`edges`, CSV)

| column | meaning |
|---|---|
| `ask` | executable price to buy this outcome now |
| `fee` | taker fee per share at that price |
| `fair` | consensus no-vig probability |
| `edge_pts` | `(fair − ask − fee) × 100` |
| `size_at_ask` | shares at the best ask (CLOB only) |
| `size_with_edge` | shares across all ask levels that still clear `min_edge` |
| `fair_source` | `books` (exact line), `model` (converted line) or `none` |
| `books` | books used; `pinnacle_fair` is Pinnacle's own no-vig number |
| `why_not` | why a row is not flagged |
| `stale` | `|lastTradePrice − mid| > 0.03`, spread > 0.05, or no quote |

The footer always reads `N markets checked, K flagged`. When K = 0 it prints **No edge found.**

## Limitations

- **One-sided props.** Anytime-TD and milestone ("250+") markets are often quoted only on "Yes" or "Over". Those get no fair value rather than a guessed one.
- **Model-converted lines** are approximations: a fixed σ, rough key-number weights, and push handling that assumes books grade integer lines as pushes.
- **Fees.** The rate is read per market. The formula `rate × p × (1 − p)` comes from third-party summaries of Polymarket's fee docs. Maker orders pay no fee, and the tool assumes you take liquidity.
- **Period ties.** A sportsbook's 2-way half or quarter moneyline usually pushes on a tie. Polymarket's 1H/1Q moneyline rules for a tied period may differ, so read the market rules.
- **Stale prices.** Gamma's `bestBid`/`bestAsk` can lag the order book; the CLOB book is preferred when it loads. A wide or stale market is flagged in the output but still evaluated.
- **Question parsing.** Mapping a Polymarket question to a sportsbook market uses `sportsMarketType` plus regex on the question text. Unrecognised formats show up in the "No fair value" table with a reason. They are never silently dropped.
- **Ties and pushes.** Polymarket's resolution rules for ties and pushes may differ from a sportsbook's. Read the market rules for integer lines and moneylines.
- **Free data only.** Historical odds are paid, so line history starts at your first snapshot.

## What the first live run verified (2026-10-19)

- **Slug:** `nfl-was-sf-2026-10-20` is the UTC date. The local-date slug returned nothing.
- **Events:** two, `…-2026-10-20` (327 game, half and quarter markets) and `…-player-props` (167 markets). The props event was found by the suffix probe, not by search.
- **`sportsMarketType` values seen:**
  - Priced: `moneyline`, `spreads`, `totals`, `team_totals`, each also with a `first_half_`/`second_half_`/`q1_`…`q4_` prefix. Props: `passing_yards`, `passing_touchdowns`, `passing_attempts`, `passing_completions`, `rushing_yards`, `receiving_yards`, `receptions`, `longest_reception`, `anytime_touchdowns`, `two_plus_touchdowns`.
  - No sportsbook equivalent: `team_first_touchdowns`, `exact_margin`, `safety`, `team_touchdowns`, `team_offensive_yards`, `two_point_conversions`, `qN_both_teams_to_score_points`.
- **Question formats:**
  - `Spread: 49ers (-6.5)` (outcome 0 is the named team; `line` matches)
  - `1Q Spread: Commanders (-2.5)`
  - `Commanders vs. 49ers: 1H O/U 22.5`
  - `Commanders 1H Team Total: O/U 6.5`
  - `Commanders vs. 49ers: 2H Moneyline`
  - `Brock Purdy: Passing Yards O/U 224.5`
  - `George Kittle: Anytime Touchdown`
  - `Christian McCaffrey: 2+ Touchdowns`
- **Field types:** `outcomes`, `outcomePrices` and `clobTokenIds` are JSON strings. `bestBid` and `liquidityNum` can be null.
- **CLOB `/book`:** prices and sizes are strings, and both sides are listed worst-first.
- **`/prices-history?market=<token_id>&interval=1d&fidelity=5`:** returns 5-minute points, so fidelity is in minutes.
- **Odds API:** every market key used exists (the `/events/{id}/markets` listing costs 1 credit). Team totals carry the team in `description`.
- **Hand-checks** against the raw responses:
  - ML: Pinnacle SF −340/WAS +289 gives 76.17%; Polymarket asks 0.24 / 0.77.
  - Prop: McCaffrey receptions 4.5, Pinnacle −127/+105, gives 53.67%.
  - Total: main line 51.5 (DraftKings) / 52 (Pinnacle), down from 52.5 in the reference notes.

Still worth checking: the Polymarket UI against the CLOB numbers, and Polymarket's tie and push rules for integer lines and period moneylines.

## Layout

```
pm_edges/
  cli.py          argparse entry point (pm_edges markets|snapshot|edges)
  config.py       thresholds, book weights, fee defaults, .env loading
  httpclient.py   retries/backoff, non-JSON detection, raw archiving, offline replay
  polymarket.py   Gamma + CLOB clients, slug discovery, question parsing, executable prices
  sportsbooks.py  Odds API client and normalisation
  match.py        Polymarket outcome -> sportsbook selection -> fair probability
  fair.py         odds conversion, de-vig (3 methods), consensus, line model, Kelly
  signals.py      line move, steam, juice, divergence, PM drift, bet-split hook
  edges.py        edge, fee, depth, flagging
  store.py        SQLite snapshots
  report.py       terminal tables, CSV, HTML
  pipeline.py     fetch -> match -> signals -> edges
tests/            offline tests; fixtures/synthetic is generated by fixtures/build_synthetic.py
data/             raw JSON, snapshots.db, reports (gitignored)
```

## Reference sanity check (2026-10-19 WAS @ SF, from the build plan)

Spread SF −7.5 (−110/−110; opened −5.5). Total 52.5 (DraftKings O −105 / U −115) → no-vig about Over 48.9% / Under 51.1%. ML SF about 77–79%. A prediction-market Under 52.5 at about 53.5¢ against a fair value near 51% is **no edge**. The synthetic fixtures reproduce this case, and `tests/test_e2e.py` asserts it.


## Small stakes and what the tool refuses to flag

- **Stake cap.** Nothing models an order above `max_stake` (default $50). `tools/prop_liquidity.py` computes fills at $5, $25 and $50 and refuses larger stakes unless you pass `--max-stake`.
- **One soft book is not enough.** A row only flags when its fair value comes from Pinnacle or at least two books (`min_books_to_flag`). On 2026-10-19 the tool showed McCaffrey rushing Under 59.5 at +2.2 from FanDuel alone, while Pinnacle, DraftKings and BetMGM all posted 61.5 to 62.5. At 59.5 the Under should be below 50%, so the edge was an artifact.
- **Props resolve differently.** Polymarket resolves a player prop to Under/No if the player is inactive or does not play. Sportsbooks usually refund.
- **Prop brief tooling.** `tools/prop_liquidity.py` pulls fresh books and walks the ask side per stake; `tools/prop_brief_data.py` joins that with the sportsbook fair values.

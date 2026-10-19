# pm_edges — Polymarket vs. sportsbook fair-price checker (NFL)

A read-only command-line tool. For one game it:

1. lists every Polymarket market (moneyline, spreads, totals, player props, quarter/half markets) with the price you can actually buy at;
2. derives a fair probability for each one from sportsbook lines, giving sharp books (Pinnacle first) the most weight;
3. records snapshots so it can show line movement and other signals over time;
4. flags a market only when the fair probability beats the executable price, after Polymarket's taker fee, by a set margin. Otherwise it prints **"No edge found."**

It never places bets, trades or touches a wallet. **Check that using Polymarket and sportsbooks is legal where you live.**

> **Status: not yet run against the live APIs.** It was built in a sandbox with no access to Polymarket or The Odds API. Everything was tested against *synthetic* fixtures shaped like the documented responses. Before you trust any output, do the first-live-run checklist below.

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
| `--regions us,eu` | Odds API regions; `eu` is where Pinnacle is listed |
| `--no-alternates` | never request alternate lines (saves credits) |
| `--no-history` | skip CLOB price history |
| `--snapshot` (edges) | also store this run as a snapshot |
| `--kelly --bankroll 1000` | show a quarter-Kelly stake, capped at 2% of bankroll; off by default |
| `--html` / `--no-csv` | report formats; CSV is written by default to `data/reports/` |

### Scheduling snapshots

The Odds API free tier has about 500 credits a month. Each odds request costs **(markets returned) × (regions)**. A typical run makes one request for the main markets plus props, and sometimes a second for alternate lines. With 2 regions and about 8 market keys, that is roughly 16–30 credits a run. Credits remaining are printed after every run, read from the `x-requests-remaining` header. Plan the schedule around that: 8 snapshots on game day is about 150–250 credits.

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
2. The consensus is a weighted average across books. Pinnacle has weight 3. Circa, BetOnline, Bookmaker and LowVig (Odds API keys `circasports`, `betonlineag`, `bookmaker`, `lowvig`) have weight 2, if present. Everyone else has weight 1. Edit these in `config.py`.
3. **Line mismatch**, for example Polymarket at 52.5 while books show 51.5:
   - First, an exact alternate line from the books (`alternate_spreads`, `alternate_totals`, `player_*_alternate`) is used. That counts as `fair_source = books`.
   - Otherwise, spreads and totals are converted with a discretised normal model: margin σ 13.5 with extra mass on key numbers 3 and 7, total σ 10. These rows are labelled `fair_source = model (low conf.)` and never flag unless `--allow-model` is set.
   - Player props are never model-converted. They get no fair value, with a reason.

**Edge**

```
fee  = rate × p × (1 − p)        per share, taker only (Polymarket's published formula)
edge = fair − ask − fee
```

A row is flagged when `edge ≥ min_edge` (default 0.03), liquidity ≥ the threshold and the fair value is not model-derived. The fee rate comes from the market's `feeSchedule` when Gamma provides one, and is 0 when `feesEnabled` is false. Otherwise it falls back to `default_fee_rate = 0.05`. That fallback is the reported 2026 sports rate, not verified; the footer always prints which rate was used.

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
- **Fees.** The fee formula and the 0.05 fallback rate come from third-party summaries of Polymarket's fee docs; confirm on the first live run. Maker orders pay no fee, and the tool assumes you take liquidity.
- **Stale prices.** Gamma's `bestBid`/`bestAsk` can lag the order book; the CLOB book is preferred when it loads. A wide or stale market is flagged in the output but still evaluated.
- **Question parsing.** Mapping a Polymarket question to a sportsbook market uses `sportsMarketType` plus regex on the question text. Unrecognised formats show up in the "No fair value" table with a reason. They are never silently dropped.
- **Ties and pushes.** Polymarket's resolution rules for ties and pushes may differ from a sportsbook's. Read the market rules for integer lines and moneylines.
- **Free data only.** Historical odds are paid, so line history starts at your first snapshot.

## Not yet verified against live APIs — first-live-run checklist

These were taken from docs and observed conventions. They could not be checked from the build environment.

- Gamma: `/events?slug=`, `/public-search?q=` (response shape `{"events": [...]}`), market fields `sportsMarketType` (values seen: `moneyline`, `spreads`, `totals`), `line`, `feesEnabled`, `feeSchedule.rate`/`exponent`.
- Polymarket question wording for props and spreads. The parsers handle `Player: Stat O/U N.5`, `Will Player throw for 250+ … yards?`, `Will Player score a touchdown?` and `Spread: Team (-7.5)`.
- Spread `line` sign convention when the question has no `(±x.5)`; that case is flagged in the notes.
- CLOB `/book?token_id=` (string prices, unsorted levels) and `/prices-history?market=<token_id>&interval=1d&fidelity=5` (fidelity treated as minutes).
- The Odds API market keys for props (`player_pass_yds`, `player_reception_yds`, `player_anytime_td`, …), `alternate_spreads`/`alternate_totals`, and period suffixes (`spreads_h1`, `h2h_q1`). The full list is in `sportsbooks.KNOWN_PROP_STATS`. If the API rejects a key (HTTP 422), the run retries without player props and says so.
- Bookmaker keys for the sharp tier (`circasports`, `bookmaker`, `lowvig`, `betonlineag`).

On the first live run:

1. Run `pm_edges markets …` and `pm_edges edges … --snapshot`. Look at the "No fair value" table for parsing misses.
2. Copy the run folder from `data/raw/<game>/<run>/` into `tests/fixtures/` and add regression tests against it. `tests/fixtures/build_synthetic.py` shows the shapes the tests expect.
3. Hand-check three markets: one moneyline, one total and one prop. Compare the Polymarket UI with the tool's bid/ask/size, and a sportsbook with the tool's prices and no-vig numbers.

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

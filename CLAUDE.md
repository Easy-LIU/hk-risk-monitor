# HK Equity Risk Attribution Monitor — Project Background

## Language rule

All project artifacts must be written in English: code comments, docstrings,
variable/function names, commit messages, README, design.md, and any other
file that ends up in this repository. Discussion with the user during a
session may happen in Chinese, but anything written to a file must be in
English. This rule applies to every future session on this project.

## One-line description

A web tool that answers: "Right now, how much of HSI's (Hang Seng Index) risk
is US (SPX) driven, how much is China (SSEC) driven, and how much is local
idiosyncratic risk?" It also alerts when the transmission structure breaks.

## Why this project exists

This is the engineering extension of the author's published paper (VAR +
Granger causality testing on how US/China monetary policy transmits into HK
equity market stability; core finding SPX→HSI, F=155.65, p<0.01).

- The paper is **static and retrospective**: it fits one VAR model on the
  full 2015-2025 history and reports a single ten-year average transmission
  strength — not useful for "what should I hedge with today."
- This tool is **dynamic and actionable**: it refits a VAR on a rolling
  250-day window every day, producing a daily transmission share and
  detecting structural breaks (regime changes).
- The paper's literature review criticizes static models for failing to
  capture how trade wars reroute transmission channels — this tool is the
  author's own fix for that gap.

## Decision scenario this tool serves

User: someone holding HK equity exposure (fund manager, risk analyst,
cross-market allocator).
Daily decision they need to make: **what should I hedge my HK equity
position with, and at what ratio?**

| Tool reading | Decision implication |
|---|---|
| US-driven 65% | HSI is currently tracking US equities → hedging with S&P futures/US ETFs is effective, hedge ratio can be high |
| US-driven 30%, China-driven 50% | Continuing to hedge with S&P is wasting money → switch to A50 futures or reduce exposure |
| Sudden jump from 35% to 70% | Regime change in transmission structure → existing hedges need re-evaluation |

## Core method: upgrading from Granger to FEVD

- The paper uses Granger causality tests: output is "does A cause B"
  (F-statistic + p-value) — a yes/no answer.
- This tool uses **FEVD** (Forecast Error Variance Decomposition): it splits
  HSI's forecast error variance into US-driven / China-driven / Idiosyncratic
  shares that sum to 100%.
- `statsmodels`'s VAR model ships with `.fevd()`, so no need to implement the
  math from scratch — the work is calling it correctly, rolling it correctly,
  and interpreting the output correctly.

### Rolling window mechanism

Window length is 250 trading days, rolled forward 1 day at a time; each
window is refit as a VAR → FEVD to produce that day's three percentages.
Chaining these percentages over time produces the moving line chart on the
web page.

Roughly 2250 windows total. Measured on the real 2015-2026 dataset, the
full rolling run (2386 windows) takes ~6 seconds — faster than the
~30-45 second estimate made before the engine existed. Regardless of
the actual number, recomputing on every slider drag is bad UX, so the
tool **precomputes offline and caches** (`cache/rolling_results.parquet`);
the frontend only reads the cache and never computes in real time.

## Data layer

### Variable list (5, reduced from 6 in the paper)

| Variable | Source | Ticker |
|---|---|---|
| HSI (Hang Seng Index) | yfinance | ^HSI |
| SPX (S&P 500) | yfinance | ^GSPC |
| SSEC (Shanghai Composite) | yfinance | 000001.SS |
| USD_CNY exchange rate | yfinance | USDCNY=X |
| USD_YIELD (US 10Y Treasury) | FRED API | DGS10 |

CNY_YIELD (China's 10-year government bond yield) was dropped — reason: no
stable free daily-frequency data source exists. Deliberately narrowing scope
and stating the reason is more professional than forcing in a low-quality
data source, and this must be documented in the README.

### Trading calendar alignment (the most important engineering decision in this project)

HK, US, and China A-share markets have completely different holiday
calendars (HK: Lunar New Year, Ching Ming, Dragon Boat, Mid-Autumn, Chung
Yeung, Christmas; US: Thanksgiving, Independence Day, Labor Day, Juneteenth;
China A-shares: 7-day Spring Festival, 7-day National Day).

**Decision: inner join** — keep only the days all three markets are open.
This is consistent with how the author's paper handled its 2296 observations.

The rejected alternative is forward-fill (carrying the prior day's close
forward on holidays) — reason: the constant series produced by forward-fill
creates large amounts of fake "zero return" data, which systematically
suppresses volatility estimates and biases Granger/FEVD tests toward
"detecting no causality." For a tool whose entire purpose is measuring
transmission strength, this is a fatal bias. Losing ~10-15% of the sample is
an acceptable price for keeping every remaining day genuine.

Implementation requirement: `MarketDataLoader.get_alignment_report()` must
explicitly record "N days remaining after alignment, M days lost," and this
must be reported in the README.

### Timezone/timing detail (differentiator worth discussing in the README)

HK market close (16:00 HKT) precedes the US market open (22:30 HKT, same
calendar day). So SPX's lag-1 term in the VAR actually means "last night's
US performance → this morning's HK open reaction," which is economically
sound. The reverse — HSI's lag-1 → SPX's same-day move — is subtler, because
within the same calendar day HSI actually occurs before SPX. The paper
likely never addressed this; discussing it proactively demonstrates deeper
data understanding than a typical student project.

## Architecture principles

### Five modules with clear responsibility boundaries

```
src/
├── data_loader.py   # MarketDataLoader: fetch / align_calendars / to_log_returns / check_stationarity / get_alignment_report
├── var_engine.py     # RollingVAREngine: fit_window / run_rolling / validate_against_paper
├── network.py        # SpilloverNetwork: single-timestamp transmission network, graph-algorithm showcase
├── detector.py        # RegimeDetector: structural break detection (share jumps / edges appearing-disappearing / centrality rank flips)
└── precompute.py      # offline precompute script, produces cache/rolling_results.parquet
```

- **`validate_against_paper` is the soul of this project**: it runs once on
  the full sample (no rolling) and compares the resulting Granger
  F-statistics against the paper's Table 3 (SPX→HSI should be near the
  155.65 magnitude, USD_CNY→HSI should be near 36.61). This is the project's
  biggest differentiator and must be prominently documented in the README.

- **SpilloverNetwork uses an adjacency matrix (numpy 5×5 array), not an
  adjacency list**: because the node count is fixed and small (5-6), the
  graph is dense (almost every pair of nodes has an edge), and the primary
  operation is "frequently diffing two networks" (edge-by-edge comparison)
  — matrix subtraction handles diff in one line and is cache-friendly. Be
  ready to explain this reasoning in an interview.

- **RegimeDetector tracks three signal types**: share jumps (primary),
  edge appearance/disappearance (secondary), and centrality rank flips.
  - **Share jumps** track the three dashboard headline numbers —
    US-driven / China-driven / Idiosyncratic — not individual raw edge
    weights. The test for what belongs in the primary signal: "if this
    number changes, does anyone change what they do?" US-driven moving
    from 35% to 70% means a hedger should re-evaluate their hedge ratio
    — that's a decision-relevant number. A single raw edge like
    SSEC→USD_YIELD jumping does not map to any hedging decision on its
    own; putting every one of the 20 raw edges into the primary signal
    would just reproduce the same problem edge appearance/disappearance
    already ran into (see docs/notes.md's Day 5 threshold check) —
    technically real, not actionable. Individual edge structure is
    already covered by the other two signals (threshold crossings, and
    which of SPX/SSEC is the dominant source), so the primary signal
    doesn't need to duplicate that at the raw-edge level.
  - **Centrality rank flip** detects when the dominant risk source
    switches between SPX and SSEC. Business rationale: hedging
    instrument choice is a discrete decision (US-linked vs. China-linked
    instruments). A gradual shift may never trigger the weight-jump
    threshold on any single day, yet still cross the point where the
    existing hedge becomes mismatched. This signal covers that blind
    spot.

- **RegimeDetector validation standard**: without telling it about any
  historical event, check whether it can automatically flag March 2020
  (COVID), July 2018 (trade war), and March 2022 (rate hikes). It only
  counts as working if it can.

### Scope boundaries (no scope creep)

- No GARCH, no machine learning forecasting, no additional markets
- Numerically unstable windows are caught with try/except; the count of
  failed windows is recorded and honestly reported rather than pretending
  everything succeeded. In practice, the full 2386-window rolling run on
  real 2015-2026 data had 0 failures — the failure-tracking mechanism is
  kept as a safety net regardless, and it already proved its worth once:
  during development, the hand-implemented Generalized FEVD produced
  silent NaNs on a degenerate (zero-variance) synthetic test window
  before an explicit finite-value check was added to turn that into a
  caught, reported failure instead (see docs/notes.md).
- The frontend can be cut if time runs short — by the end of Day 5, having a
  working, tested, validated engine is already enough; engine correctness >
  tests > README > frontend polish

## Git commit conventions

- `feat:` new feature / `fix:` bug fix / `test:` add tests / `docs:` docs
  change / `refactor:` refactoring
- Commit each module separately as it's completed; don't dump everything in
  one commit at the end

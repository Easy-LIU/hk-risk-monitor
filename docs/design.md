# Design Document

## 1. Module Responsibilities

**MarketDataLoader** fetches data via official APIs (yfinance for equity
indices/FX, FRED for Treasury yields) rather than scraping. Its core
responsibilities are trading calendar alignment (inner join across the
HK/US/CN markets), conversion of prices to log returns, and stationarity
testing (ADF).

**RollingVAREngine** fits a VAR model on each rolling window and computes
FEVD (Forecast Error Variance Decomposition) to quantify what percentage
of HSI's forecast error variance is attributable to each other market.
It also includes `validate_against_paper()`, which runs a full-sample
(non-rolling) fit and compares the resulting Granger causality
F-statistics against the published paper's Table 3 results as a
correctness check.

**SpilloverNetwork** represents a single time point's transmission
network as an adjacency matrix. It provides centrality analysis
(identifying the largest risk source), path search (finding indirect
transmission chains via DFS), and network diffing (element-wise
comparison between two time points, used by RegimeDetector).

**RegimeDetector** monitors a time series of SpilloverNetwork objects
for structural breaks. It tracks three signal types:

1. **Weight jumps** on existing edges (e.g., SPX→HSI share jumping from
   30% to 70% within days), signaling a material shift in the
   information environment relevant to HK equity holders.
2. **Edge appearance/disappearance** (rarer, but often corresponding to
   structural events in FX/rates/monetary policy channels).
3. **Centrality rank flip** — detects when the dominant risk source
   switches between SPX and SSEC. Business rationale: hedging
   instrument choice is a discrete decision (US-linked vs. China-linked
   instruments). A gradual shift may never trigger the weight-jump
   threshold on any single day, yet still cross the point where the
   existing hedge becomes mismatched. This signal covers that blind
   spot.

All three signal types are logged as alerts.

**Streamlit frontend** is a read-only visualization layer. It displays
cached rolling results and does not perform live computation (see the
performance/caching discussion in the README).

## 2. Data Flow

```
Raw market data (yfinance / FRED APIs)
        │
        ▼
MarketDataLoader
  - fetch()
  - align_calendars()      [inner join across HK/US/CN]
  - to_log_returns()
  - check_stationarity()   [ADF test]
        │
        ▼  (clean, aligned returns DataFrame)
RollingVAREngine
  - fit_window() for each 250-day rolling window
  - FEVD → attribution percentages per window
  - validate_against_paper() [full-sample sanity check]
        │
        ▼  (a time series of FEVD results, one per day)
SpilloverNetwork  (one instantiated per time point)
  - stores each window's result as an adjacency matrix
  - centrality / path search on demand
  - diff() between consecutive time points
        │
        ▼  (series of networks + pairwise diffs)
RegimeDetector
  - scans the diff series for weight jumps / edge changes / centrality rank flips
  - outputs a list of Alert objects (date, type, magnitude)
        │
        ▼
precompute.py  (offline script, run once, not on every page load)
  - runs the full pipeline above
  - writes results to cache/rolling_results.parquet
        │
        ▼
Streamlit frontend (app.py)
  - reads only from cache/rolling_results.parquet
  - renders KPI cards, time-series chart, network graph, alert log
  - never re-runs the VAR/FEVD computation live
```

Two points worth calling out explicitly:

1. `precompute.py` exists as a standalone step, separate from the
   frontend. This is the solution to the ~30-45 second rolling
   computation cost identified earlier: the full pipeline runs once,
   offline, and writes its output to a cache file. The frontend only
   reads that cache, so dragging the time slider never triggers a
   half-minute recomputation.

2. `pytest` coverage focuses on `SpilloverNetwork` (matrix operations,
   path search, diff logic) and `RegimeDetector` (alert-triggering
   logic), since both are pure logic that's easy to verify with small,
   hand-constructed examples. `MarketDataLoader`'s calendar alignment
   logic should also be tested — e.g. constructing two fake calendars
   and verifying the inner join produces the correct intersection.

## 3. Trading Calendar Alignment Approach

The candidate alternative is forward-fill: on a market's holiday, carry
its previous day's closing price forward. The advantage is that it
preserves every calendar trading day across the combined dataset.

However, this project's core purpose is quantifying the real-time
transmission strength of other markets into HK equities, and the "zero
return" days that forward-fill manufactures create two problems. First,
they systematically suppress volatility estimates, biasing Granger/FEVD
tests toward missing real causal relationships (a flat filled series
carries no information for the model to detect). Second, a market being
closed does not mean the information environment is closed — macro news
or policy announcements can still be moving related markets during that
window, and filling with the prior day's price masks this real,
non-synchronous transmission of information rather than representing it.

For these reasons, the chosen approach is an inner join on the three
markets' trading calendars, keeping only days all three are open. This
sacrifices some sample size, but ensures every observation used in the
analysis reflects an actual market reaction rather than a stale
carried-forward value.

## 4. SpilloverNetwork Data Structure Choice

The transmission network is small (5-6 nodes) and dense — any pair of
macro variables can plausibly transmit risk to any other, so most edges
are populated rather than absent. Given this, an adjacency matrix is used
instead of an adjacency list.

Adjacency lists are typically preferred for sparse graphs because they
save space by only storing edges that exist. Space is not the bottleneck
here: a 5x5 or 6x6 matrix is trivially small regardless of representation.

The core operation this structure needs to support is frequent diffing
between two networks at different points in time (e.g. comparing today's
transmission network to yesterday's, or to a network before/after a
regime change). With a numpy adjacency matrix, this diff is a single
element-wise subtraction. With an adjacency list, the same comparison
requires first aligning node sets between the two graphs and then
matching and subtracting edge weights pair by pair — more code, and more
opportunities for bugs (e.g. mishandling an edge that exists in one
network but not the other).

## 5. FEVD Attribution Grouping

HSI's forecast error variance decomposition produces a share for each of
the five variables (HSI itself, SPX, SSEC, USD_CNY, USD_YIELD). The
dashboard's three headline numbers — US-driven, China-driven,
Idiosyncratic — are these five shares rolled up into three groups. The
grouping is:

- **US-driven = SPX + USD_YIELD + USD_CNY**
- **China-driven = SSEC**
- **Idiosyncratic = HSI** (HSI's own shock)

The non-obvious call here is putting USD_CNY under US-driven rather than
China-driven. USD_CNY is the RMB exchange rate, so grouping it with SSEC
under "China-driven" is the intuitive first guess — but
`validate_against_paper()` shows USD_YIELD → USD_CNY is a highly
significant channel (paper F=16.58; this tool reproduces it at F=29.35,
same order of magnitude). That result says USD_CNY's day-to-day
variation is substantially driven by US monetary policy (the 10-year
Treasury yield) flowing through the exchange rate, not by an independent
China-side shock. If USD_CNY's FEVD share were counted as China-driven,
a USD-originated move that transmits through the RMB channel into HSI
would get recorded as Chinese risk — the tool would systematically
overstate China's share and understate the US's, exactly in the cases
where the US channel is operating indirectly rather than directly
through SPX. Grouping USD_CNY with US-driven keeps the attribution
consistent with the verified causal structure rather than with the
variable's surface-level geography.

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
It uses the **Generalized FEVD** (Pesaran & Shin, 1998), not the
Cholesky-orthogonalized FEVD `statsmodels` provides out of the box,
because Cholesky FEVD's result depends on variable ordering — see
docs/notes.md for the diagnostic that found this the hard way. It also
includes `validate_against_paper()`, which runs a full-sample
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
   frontend. The point of precomputing and caching is zero-latency
   frontend interaction: dragging the time slider should never trigger
   a recomputation, regardless of how long the rolling pipeline
   actually takes. In practice, a full 2386-window rolling run
   (window=250, step=1, 2015-2026) measured at ~6 seconds — faster
   than the ~30-45 second estimate made before the engine was
   implemented. The precompute/cache split was the right call
   independent of that number: even at 6 seconds, recomputing on every
   slider drag would still be a bad interaction, so the full pipeline
   runs once, offline, and writes its output to a cache file that the
   frontend only reads.

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

## 6. SpilloverNetwork Edge Threshold

GFEVD produces a dense matrix — nearly every off-diagonal cell is
non-zero. Without a cutoff for "this edge exists," the network would be
fully connected at all times, and RegimeDetector's edge
appearance/disappearance signal could never fire, since there would
never be a transition from "edge present" to "edge absent." A threshold
is required.

**The distribution of edge weights is continuous, with no natural gap.**
Across all 47,720 off-diagonal cells from a full rolling run on the real
2015-2026 sample (2386 windows), the percentiles are: p10=0.32%,
p25=0.87%, median=2.45%, p75=7.65%, p90=16.87%, mean=5.79%. There is no
bimodal split into an obvious "noise cluster" and "signal cluster" —
choosing a threshold here is a calibration judgment, not a discovery
of some natural breakpoint the data reveals on its own.

**The threshold was recalibrated from 1% to 2% on Day 5, based on a
sensitivity check across 0.5% / 1% / 2%** run against the full
2015-2026 rolling series (full numbers and the comparison table are in
docs/notes.md). The initial 1% choice (see the version-controlled
history of this file for that reasoning) was based on staying below
every core edge's historical minimum with a safety margin. That
reasoning was not wrong on its own terms, but the sensitivity check
surfaced a more important consideration that changed the decision:

**The switch to 2% was driven by signal composition, not event count.**
Total edge appearance/disappearance events were similar across all
three thresholds (731 at 0.5%, 682 at 1%, 668 at 2% — under a 10%
range), so raw event count could not distinguish them. What did
distinguish them was *which* edges were generating the events. At 0.5%
and 1%, the top 5 most volatile edges were almost entirely USD_CNY
pairs — the same channel Day 3's `validate_against_paper()` diagnostic
already flagged as unverified against the paper. At those thresholds,
the "edge appeared/disappeared" signal was mostly tracking noise in a
channel already known to be unreliable, not genuine structural change.
At 2%, the top 5 shifted to edges involving SPX and SSEC's connections
to HSI and each other — the paper-validated core channels — crossing
the threshold at meaningful moments instead.

**This is a deliberate change in what the threshold is for, not a
parameter tweak, and it has a real side effect worth stating plainly:**
the original 1% was chosen specifically so that SPX→HSI — the
strongest, most paper-validated channel — could never register as
"disappeared," even at its historical worst (a minimum of 1.68% in this
sample). At 2%, that safety margin is gone: SPX→HSI *would* be flagged
as a disappeared edge at that same historical low point. This is now
treated as intended behavior rather than a bug to avoid — if the
US-equity channel into HSI genuinely weakens to a multi-year low, an
alert is exactly what a structural-break detector should produce. The
design goal shifted from "guarantee the core channel never falsely
disappears" to "let the core channel's disappearance be a real,
actionable signal," and 2% is the threshold consistent with the latter
goal.

**This threshold is still a calibration on this specific sample period,
not a universal constant** — see docs/notes.md for the full comparison
table and the decision to additionally downgrade edge
appearance/disappearance to a secondary signal in RegimeDetector,
independent of which exact threshold is used.

## 7. Path Weight Combination (find_all_paths / strongest_path)

An indirect path like A→B→C needs a single number representing its
strength, combined from the two edge weights along it. The choice is
multiplication, not addition.

Each edge weight is a variance share — a proportion, not a raw
magnitude. Multiplying two shares has a property that matches economic
intuition: **an indirect path's combined weight can never exceed either
of its individual edge weights**, since both weights are in [0, 1] and
a product of two fractions is never larger than either factor. Going
through an extra hop can only leave a transmission chain as strong as
its weakest link, never stronger — a two-hop path can never be reported
as a more powerful transmission channel than a strong direct edge.

Addition does not have this property. Summing two variance shares from
two different decompositions can produce a number larger than either
individual edge — and larger, in some cases, than 1 — with no
interpretation in the variance-decomposition framework this tool is
built on: there is no real quantity that "0.3 (A's contribution to B)
plus 0.4 (B's contribution to C)" corresponds to. Multiplication
preserves an interpretation (a compounding transmission efficiency,
analogous to combining two conditional shares); addition does not.

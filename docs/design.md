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

## 8. RegimeDetector Signal Definitions and Parameter Calibration

### What "weight jump" tracks

The primary signal tracks the three dashboard headline numbers —
`us_share`, `china_share`, `idio_share` from `WindowResult` — not raw
edge weights from the underlying 5x5 FEVD matrix. This was an
inconsistency in earlier drafts of this project's docs (CLAUDE.md's
prose example and this file's original phrasing implied a raw edge,
e.g. "SPX→HSI jumping from 30% to 70%," while the original wireframe's
example — "US份额 45%→71%" — was always about the aggregated headline).
Resolved in favor of the aggregated headline, using one test: **if this
number changes, does anyone change what they do?** `us_share` moving
from 35% to 70% is the example CLAUDE.md's decision-scenario table
already uses to explain when a hedger should re-evaluate their hedge
ratio — a decision-relevant number. A single raw edge like
SSEC→USD_YIELD jumping does not correspond to any hedging decision by
itself. Making every one of the 20 raw edges part of the primary signal
would reproduce the exact problem edge appearance/disappearance already
had (section 6 above, and the Day 5 result in docs/notes.md): technically
real movement that maps to no action. Raw edge-level structure is
already covered by the other two signals (edge threshold crossings, and
which of SPX/SSEC dominates), so the primary signal does not need to
re-cover that ground.

`scan()` therefore takes `list[WindowResult]`, not
`list[SpilloverNetwork]` — `SpilloverNetwork` has no date field, and
`Alert` needs one, so a `SpilloverNetwork`-only signature could not have
produced dated alerts in the first place. `SpilloverNetwork` objects are
constructed internally from each `WindowResult.fevd_matrix` wherever the
edge-level signals need them.

### Parameter calibration: threshold_pp and lookback

Both parameters were set by looking at the real distribution of
`|share(t) - share(t-lookback)|` across the full 2015-2026 rolling
series (2386 windows), the same method used for `EDGE_THRESHOLD` in
section 6 — not guessed.

`lookback=5` (trading days, about a calendar week) was kept from the
original wireframe's framing ("5日内") and matches the horizon a hedger
would plausibly re-check a position on.

At `lookback=5`, the percentile table for each headline share was:

| Percentile | us_share | china_share | idio_share |
|---|---|---|---|
| p50 | 0.38pp | 0.31pp | 0.35pp |
| p90 | 1.32pp | 1.14pp | 1.28pp |
| p95 | 1.97pp | 1.65pp | 1.93pp |
| p99 | 4.32pp | 2.97pp | 4.50pp |
| p99.9 | 8.23pp | 4.12pp | 11.18pp |
| max | 9.75pp | 4.90pp | 11.64pp |

`threshold_pp=4.0` was chosen to sit close to the 99th percentile for
`us_share` and `idio_share` — i.e. a genuinely rare event in this
sample, consistent with treating this as a structural-break signal
rather than routine noise (the same "too frequent to be meaningful"
standard that got edge appearance/disappearance demoted to secondary in
section 6). The same numeric threshold is applied to all three shares
rather than a separate threshold per share, which has a real,
data-driven consequence worth stating rather than hiding: `china_share`
is structurally calmer in this sample (p99 = 2.97pp, below the 4.0pp
threshold entirely), so it will fire far less often than `us_share` or
`idio_share` under the same cutoff. That is read as a fact about this
sample (China-driven risk moved less dramatically over 5-day windows
than the US-driven or idiosyncratic components did, 2015-2026), not as
a flaw in using one shared threshold.

As with `EDGE_THRESHOLD`, known stress periods (2018 trade war, 2020
COVID, 2022 rate hikes) were checked *after* the parameters were fixed
from the percentile table above, not used to tune them — see
docs/notes.md for what `RegimeDetector.scan()` actually produced on the
full history, reported as-is.

## 9. Dual-Timescale Share Jumps and Episode Aggregation

### Why two timescales

A single `lookback=5` share-jump signal caught COVID (a sudden shock)
but missed the 2018 trade war and largely missed the 2022 rate-hike
cycle — both multi-month processes, not discrete jumps. A
sensitivity experiment (docs/notes.md) confirmed the mechanism: a
5-day window structurally cannot detect a change that accumulates
gradually over months, no matter how `threshold_pp` is set, because the
change per 5-day slice can stay small even while the cumulative change
over the full period is large.

The fix is not to replace the 5-day signal with a longer one, but to
run both:

- **acute** (`lookback=5`): a sudden shock. Business framing: re-evaluate
  the hedge ratio *now*.
- **chronic** (`lookback=60`, ~1 quarter): a sustained drift. Business
  framing: flag for the next periodic hedge-allocation review, not an
  immediate action.

Both are share-jump signals with the same mechanism (compare
`WindowResult` shares `lookback` windows apart against a threshold);
they differ only in `lookback` and — necessarily — in `threshold_pp`,
since a bar calibrated for a 5-day distribution is not the right bar
for a 60-day one (60-day moves are naturally larger; see the acute vs.
chronic percentile tables in docs/notes.md).

### Episode aggregation

A sustained drift crosses the threshold on every day it stays past it,
so `lookback=60` alone produced 1572 raw share_jump crossings — not
because the underlying signal is noisy, but because one real,
multi-week drift was being reported as dozens of near-duplicate daily
alerts. `RegimeDetector` now collapses consecutive (by trading-day
index), same-share, same-direction crossings into a single episode:
start date, end date, net change, and duration in trading days (e.g.
"China-driven share drifted 17.0% → 24.6% over 2018-08-14 to 2018-08-20,
+7.5pp, 5 trading days"), instead of one alert per day. This is not
cosmetic deduplication — it changes what a `share_jump` alert *means*:
one alert now corresponds to one real event, not one day of an ongoing
event.

### Chronic threshold: from percentile calibration to frequency calibration

The acute threshold (section 8, unchanged) is calibrated by percentile
(p99 of the 5-day distribution) because, without episode aggregation
protecting it, a low bar meant runaway alert volume — rarity was doing
double duty as both "statistically meaningful" and "keeps the log
usable." Once episode aggregation exists, that second job is no longer
the threshold's to do: a sustained drift is one episode no matter how
low the bar is set, so volume is no longer the constraint.

This was tested directly. A first attempt calibrated the chronic
threshold the same way as acute — p99 of the 60-day distribution,
giving `chronic_threshold_pp=9.5`. Under that bar, only COVID cleared
it at either timescale; the trade war's largest 60-day move (7.98pp)
and the rate-hike period's moves both fell short of a threshold defined
as "the top 1% of this same sample's own 60-day moves." This is not a
sign the trade war didn't matter — see docs/notes.md's "three-layer
conclusion" for why a real, economically coherent, sustained shift can
still fail a strict self-referential rarity bar, and why that is a
meaningful distinction (news event vs. statistical anomaly in
transmission structure) rather than a threshold that needs to be forced
higher or lower until a known event shows up.

**The chronic threshold is instead calibrated by target business
frequency**, not percentile: candidate thresholds were tested once each
and the resulting *episode count per year* measured, targeting 3-5
chronic episodes/year — a plausible cadence for periodic hedge-review
prompting. `chronic_threshold_pp=7.0` (3.90 episodes/year over the
10.5-year sample) was selected on that basis and run once, final,
regardless of which known events it did or didn't flag. It happened to
detect the trade war (3 episodes) and partially detect the rate-hike
period (2 episodes, at its tail end) — a result, not a selection
criterion.

The acute threshold keeps the percentile rule (section 8) unchanged: an
acute, sudden shock *should* be rare by construction, so "top 1% of
5-day moves" is the right kind of bar for it. Only the chronic
threshold's calibration method changed, and only because episode
aggregation removed the reason percentile-based rarity was needed there
in the first place.

### The dual-timescale design holds regardless of chronic's hit rate

`chronic_threshold_pp` and `chronic_lookback` are constructor
parameters, not hardcoded — a user with a different risk tolerance can
adjust them. The architectural claim this section defends is narrower
and does not depend on any specific threshold's hit rate: acute shocks
and chronic drift are genuinely different phenomena that a single
lookback cannot both serve, and a tool meant to support both an
immediate hedge re-evaluation and a periodic portfolio review needs a
signal for each.

## 10. Planned: Historical Event Annotations (Day 6 frontend)

The share-history line chart will overlay reference markers (vertical
lines + labels) for well-known macro events, so a user can visually see
how the three headline shares behaved during past stress periods and
calibrate their expectations for a similar future scenario.

**This is an annotation layer, not a detection output, and the two must
stay visibly distinct in both the UI and any accompanying text.** The
event markers are a hand-maintained reference list curated from
standard macro chronology (e.g. the COVID market crash, the US-China
trade war escalation, the start of the Fed's 2022 hiking cycle, the
811 RMB reform) — never selected because a particular date happens to
line up with an interesting-looking move in the chart. `RegimeDetector`
is the tool's actual detector; the annotation list exists to give the
detector's output a historical reference frame, not to imply the tool
"found" these dates itself. Wording in the UI ("reference events," not
"detected events") and in the README must not blur this line.

Implementation: the event list lives in its own config (e.g.
`src/events.py` or a JSON file), separate from the plotting code, so it
can be reviewed and extended independently of chart logic.

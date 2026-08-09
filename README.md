# HK Equity Risk Attribution Monitor

HK equity risk attribution monitoring system — full documentation will be completed on Day 7.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Engineering Decisions

### FEVD horizon = 10

The rolling engine decomposes forecast error variance 10 steps ahead
(`fevd_horizon=10`, exposed as a constructor parameter). Reasons for
this choice:

- It matches the standard horizon used in the Diebold-Yilmaz spillover
  index literature, so the tool's output is comparable to existing
  published spillover studies rather than an arbitrary one-off choice.
- 10 trading days is about two calendar weeks — a reasonable window for
  cross-market transmission to fully play out, long enough to capture
  indirect effects but short enough to still describe a "current"
  regime rather than a multi-month average.
- Under the fitted VAR(1), transmission effects decay quickly with each
  additional lag step, so horizons beyond 10 (e.g. 20) produce
  materially the same decomposition — 10 is not an undershoot of where
  the decomposition stabilizes.

## Limitations

The USD_CNY → HSI channel could not be validated against the paper: the
tool finds no significant Granger causality (F=0.03, p=0.86) where the
paper reports F=36.61, and this holds even when the sample is truncated
to the paper's exact time window. Other Table 3 relationships (SPX→HSI,
SSEC→HSI, USD_YIELD→USD_CNY) reproduce at the same order of magnitude,
so this appears specific to the USD_CNY series rather than an issue
with the engine. See [docs/notes.md](docs/notes.md) for the full
diagnostic trail. USD_CNY → HSI should be treated as an unverified
channel in the tool's output.

**2016-2017 is a detection blind spot, by design.** `RegimeDetector`'s
share-jump thresholds are calibrated out-of-sample (see below): a given
year's threshold only uses prior years' data, and requires at least 252
trading days of history before it is trusted at all. The tool's
`WindowResult` series starts 2016-02-02, so 2016-2017 falls short of
that floor — this period produces **no share_jump alerts regardless of
what happened**, because the tool did not yet have enough history to
judge what counts as extreme. This must not be read as "these years
were calm." The chart marks this period explicitly rather than leaving
it silently empty.

**`SpilloverNetwork.EDGE_THRESHOLD` (2%) is still calibrated on the full
sample**, not out-of-sample — only `RegimeDetector`'s acute/chronic
thresholds were addressed (see below). This is a smaller concern than
the share-jump thresholds were: `EDGE_THRESHOLD` was calibrated from a
structural property (staying below the core paper-validated edges'
historical minimums, docs/design.md section 6), not a percentile of
noisy tail events, so it is less exposed to a single event defining its
own bar — but it has not been re-verified out-of-sample.

**The specific alert list is sensitive to calibration methodology.**
Comparing in-sample (`scan()`) and out-of-sample
(`scan_out_of_sample()`) calibration over the same 2015-2026 history
yields the same qualitative conclusions on all three known stress
periods, but only 61% overlapping episodes at the individual-alert
level (matched by field, timescale, and overlapping dates) — the
remaining 39% appear in only one of the two versions, concentrated in
years where the calibrated threshold differs most from the fixed
full-sample value (2018, 2021). Users should treat individual alerts
as indicative of a real regime shift worth investigating, not as a
precise, uniquely-determined boundary — the exact date range and
magnitude of a given episode depends on which calibration method
produced it. See docs/notes.md for the full comparison.

## Future Work

### Third-Party Capital Flow Attribution

The current model attributes HK equity risk exclusively to US and China
channels, treating everything else as idiosyncratic. In practice, a
portion of that residual likely reflects non-US/non-China capital flows
— European and Japanese allocators, Middle East sovereign funds, and
global risk-off rotations — for which Hong Kong serves as a liquid
access point.

Why this matters operationally: idiosyncratic risk is, by definition,
un-hedgeable within the current framework. If a meaningful share of it
is actually driven by identifiable third-party flows, it becomes
hedgeable. Decomposing the residual would directly expand the set of
positions a manager can hedge rather than simply absorb.

Proposed approach: introduce third-party proxies (STOXX 50 / Nikkei 225
for alternative equity pools, gold and JPY for global risk-off rotation,
CNH-CNY spread for offshore capital sentiment) and test whether they
explain a statistically meaningful share of what the current model
classifies as idiosyncratic.

### Offshore CNH as the USD_CNY Replacement

Offshore CNH is conceptually the more appropriate measure for this
tool — it trades in Hong Kong and shares the same investor base and
capital flow channels as HK equities, whereas onshore CNY is subject
to PBOC central parity management and largely inaccessible to
offshore participants. Substituting CNH would require a paid data
source (Bloomberg, Refinitiv, or CEIC); no free provider tested
offers daily historical CNH spot.

### FX as a Separate Fourth Attribution Category

The current grouping folds USD_CNY into US-driven alongside SPX and
USD_YIELD (see docs/design.md section 5 for the rationale). A more
granular decomposition would split out FX as its own fourth category —
US-equity-driven / US-rates-and-FX-driven / China-driven / Idiosyncratic
— rather than merging the rates and FX channels together. This was not
implemented in order to keep the dashboard to three headline numbers;
revisiting it would trade frontend simplicity for attribution
precision.

### Sample Start Date and Extending the History

This tool's sample starts in 2015 — not because earlier data is
unavailable, but because the market structure it measures did not
exist in its current form before then. Stock Connect (the direct
cross-border trading link between HK and mainland exchanges) launched
in phases: Shanghai-HK Connect in November 2014, Shenzhen-HK Connect in
December 2016. Before Stock Connect, there was no direct capital
channel between mainland and HK equity markets for this tool to
measure. Separately, the August 2015 "811" RMB reform fundamentally
changed how the USD_CNY midpoint is set. A transmission-attribution
tool run across that boundary would be mixing two structurally
different regimes into one VAR, not just adding more data points.

Extending the sample further back (e.g. to include the 2008 financial
crisis) would help address the in-sample threshold-calibration
circularity noted in docs/notes.md (a longer, more event-rich history
gives percentile-based thresholds more independent extreme events to be
calibrated against) — but doing so responsibly would require treating
2008-2014 and 2015-present as structurally distinct regimes, likely
via segmented modeling, rather than pooling them into a single rolling
window that crosses the Stock Connect / 811 reform boundary.

### Out-of-Sample Threshold Calibration — Done for RegimeDetector, Not Yet for EDGE_THRESHOLD

`RegimeDetector.scan_out_of_sample()` now calibrates the acute/chronic
share-jump thresholds using an annually-recalibrated expanding window
(see docs/design.md section 11), resolving the in-sample circularity
this section originally flagged as unresolved. Verified against the
original in-sample `scan()`: the same three known stress periods
produce the same qualitative verdicts (COVID detected at both
timescales, trade war chronic-only, rate hikes weakly detected either
way), though the specific episode-level detections differ by about 39%
between the two versions — see docs/notes.md for the full comparison,
including why (year-to-year threshold differences change which
consecutive days merge into an episode).

`SpilloverNetwork.EDGE_THRESHOLD` was not part of this fix and remains
calibrated on the full sample. Extending the same expanding-window
approach to it is the natural next step, though it calibrates
differently (structural margin below core edges' minimums, not a
percentile) so the method would need adapting rather than reusing
`RegimeDetector`'s calibration functions directly.

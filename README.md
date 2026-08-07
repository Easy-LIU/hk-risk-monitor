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

### Out-of-Sample Threshold Calibration

Both `SpilloverNetwork.EDGE_THRESHOLD` and `RegimeDetector`'s
percentile-based acute threshold are calibrated using percentiles of
the full historical sample — which means events like COVID that sit
inside that same sample partly define the bar they are then evaluated
against (see docs/notes.md's "three-layer conclusion" on RegimeDetector
for the fullest discussion of this circularity). A more rigorous design
would calibrate out-of-sample: an expanding window (using only data
available up to each historical point in time) or a rolling percentile,
so a threshold is never partly defined by the event it is later used to
flag. Not implemented in this pass.

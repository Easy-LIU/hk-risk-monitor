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

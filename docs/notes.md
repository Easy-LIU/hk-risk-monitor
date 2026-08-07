# Engineering Notes

## yfinance MultiIndex column bug (found during Day 2, MarketDataLoader)

### Symptom

`MarketDataLoader.fetch()` was implemented as:

```python
def _fetch_yfinance(self, ticker: str) -> pd.Series:
    data = yf.download(ticker, start=self.start, end=self.end, progress=False)
    return data["Close"]
```

Running the real pipeline against live data, `raw.columns` came back as
tuples instead of plain strings:

```
[('HSI', '^HSI'), ('SPX', '^GSPC'), ('SSEC', '000001.SS'),
 ('USD_CNY', 'USDCNY=X'), ('USD_YIELD', 'USD_YIELD')]
```

instead of the expected `['HSI', 'SPX', 'SSEC', 'USD_CNY', 'USD_YIELD']`.

### Root cause

Recent versions of `yfinance` return a `(Price, Ticker)` MultiIndex on
`data.columns` even when a single ticker string (not a list) is passed to
`yf.download()`. As a result, `data["Close"]` does not return a 1-D
`pd.Series` as the type hint promised — it returns a **one-column
DataFrame** whose single column is still labeled with the ticker.

`MarketDataLoader.fetch()` then combines the five per-variable results
with `pd.concat(columns, axis=1)`, where `columns` is a dict mapping each
variable name (`"HSI"`, `"SPX"`, ...) to its fetched series. When some of
those values are actually one-column DataFrames rather than Series,
`pd.concat` falls back to building a two-level `MultiIndex` on the output
columns — outer level from the dict key, inner level from each
DataFrame's own (ticker-named) column — to keep the concatenation
consistent across all inputs.

### Fix

`_fetch_yfinance` now explicitly unwraps a one-column DataFrame back into
a plain Series before returning, so every entry handed to `pd.concat` is
a genuine `pd.Series` regardless of the installed `yfinance` version:

```python
def _fetch_yfinance(self, ticker: str) -> pd.Series:
    data = yf.download(ticker, start=self.start, end=self.end, progress=False)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(None)
```

The same defensive `.rename(None)` is applied to the FRED-sourced series
so that no per-series `.name` attribute can reintroduce a second column
level later, even if a future edit mixes Series and DataFrames again.

### Why this bug was dangerous

It never raised an exception. `fetch()`, `align_calendars()`, and
`to_log_returns()` all ran to completion and produced a DataFrame of the
right shape — the corruption was silent. The danger surfaces downstream,
the first time any code accesses a column by its intended flat name, e.g.
`returns["HSI"]` inside `RollingVAREngine` or `SpilloverNetwork`. With
`MultiIndex` columns like `("HSI", "^HSI")`, a plain `returns["HSI"]`
lookup raises a `KeyError` at best — or, if handled carelessly (e.g. by
falling back to positional indexing or a broad `try/except`), could
silently select the wrong variable instead of failing loudly. Because the
bug is invisible at the point it's introduced (`fetch()`) and only bites
several layers later, it is exactly the kind of defect that a "does it
run without crashing" smoke test would miss — it needed either an
explicit dtype/shape assertion on `fetch()`'s output or, as happened
here, actually running the full pipeline against live data before
declaring the module done.

## Unverified Channel — USD_CNY → HSI

### Symptom

`validate_against_paper()` was run on the full aligned sample (2015-2026)
against four benchmark relationships from the paper's Table 3. Three of
the four reproduced at the same order of magnitude:

| Pair | Paper F | Tool F |
|---|---|---|
| SPX → HSI | 155.65 | 211.63 |
| SSEC → HSI | 11.32 | 8.33 |
| USD_YIELD → USD_CNY | 16.58 | 29.35 |
| USD_CNY → HSI | 36.61 | **0.03** (p=0.86) |

The fourth is not a magnitude mismatch — it is a flip in the causality
conclusion itself, from highly significant in the paper to
statistically indistinguishable from no relationship at all.

### Four rounds of diagnosis

1. **Sample-period control.** Truncated the sample to the paper's exact
   window (2015-01-05 to 2024-12-31) and re-ran the Granger test.
   Result: F=0.4652, p=0.4952. Still far below 36.61, ruling out
   "the extra 2025-2026 data is diluting the relationship" as the
   explanation.

2. **Data quality check.** USD_CNY log-return std = 0.0031 vs. the
   paper's Table 1 std = 0.0026 — same order of magnitude. Exact-zero
   returns were 2.58% of observations, not enough to account for the
   result. Descriptive stats (mean, min, max) showed no anomaly.

3. **Correlation with HSI.** -0.126 (both full sample and paper-period
   subsample) vs. the paper's Table 2 value of -0.265 — about half the
   magnitude, consistently across both periods.

4. **Offshore RMB substitution.** Attempted to replace onshore USD_CNY
   with offshore USD_CNH as a data-source cross-check. yfinance
   (`CNH=X` and `USDCNH=X`, tried via both `yf.download()` and
   `yf.Ticker().history()`, plus `period="max"` and datetime-typed
   start/end — four calling variants total) returned no historical
   data for either ticker. Stooq was tried as a second, independent
   backend (`pandas_datareader.data.DataReader(..., "stooq")` and a
   direct CSV request to `stooq.com/q/d/l/`) and also returned no
   series. No further data sources were attempted.

### Inference

The volatility magnitude matches the paper (std ratio ~1.2x) but the
co-movement with HSI is roughly half of what the paper reports,
consistently across both the full sample and the paper's exact period.
This combination — matching volatility, mismatched co-movement — points
toward the two studies using different underlying exchange-rate series,
rather than an implementation error in this codebase: a bug in the
alignment, return calculation, or Granger test would be expected to
distort volatility and correlation together, not leave one intact while
roughly halving the other. The most likely explanation is onshore CNY
vs. offshore CNH (the paper does not specify which was used), or a
difference in daily closing-time convention between the two data
providers.

### Resolution

USD_CNY is retained in the VAR system. Removing it would change the
system's structure and invalidate the three relationships that were
already verified (a 5-variable VAR is a different model from a
4-variable one). Instead, USD_CNY → HSI is documented and surfaced as
an **unverified channel** in both this file and the project README, so
that anyone reading the tool's output knows this specific edge has not
been reproduced against the published paper, while the rest of the
engine has.

## Cholesky FEVD ordering sensitivity (found during Day 3, RollingVAREngine)

### Symptom

`fit_window()` was originally implemented using statsmodels' built-in
`results.fevd(horizon)`, with the five variables in fetch order
(`HSI, SPX, SSEC, USD_CNY, USD_YIELD`, HSI first). On real data, HSI's
decomposition came out to US-driven=12.8%, China-driven=0.2%,
Idiosyncratic=87.0%. The China-driven number was hard to reconcile with
the paper: Table 2 reports SSEC-HSI as the *highest* pairwise
correlation in the whole matrix (0.542, higher than SPX-HSI's 0.197),
yet SSEC's variance contribution to HSI was reported as effectively
zero.

Reordering the same window's columns to put HSI last (or SSEC first,
HSI second) and rerunning `fit_window()` produced wildly different
results from the identical data and identical model:

| Order | US-driven | China-driven | Idiosyncratic |
|---|---|---|---|
| HSI first (original) | 12.75% | 0.22% | 87.03% |
| HSI last | 25.45% | 21.91% | 52.64% |
| SSEC first, HSI second | 12.85% | 28.42% | 58.73% |

Same window, same VAR fit, same forecast horizon — only the column
order changed, and China-driven swung between 0.2% and 28.4%.

### Root cause

`statsmodels`' `.fevd()` implements the standard Cholesky-orthogonalized
FEVD. Cholesky orthogonalization attributes all contemporaneous
(same-period) covariance between variables to whichever variable comes
first in the ordering — it treats that first variable's shock as
"causing" the shared movement, and every later variable's shock as
independent of it. HSI and SSEC trade in overlapping hours and move
together contemporaneously (hence the paper's 0.542 correlation). With
HSI listed first, the Cholesky decomposition assigned essentially all of
that shared same-day movement to HSI's own shock — which is exactly why
it showed up as inflated "idiosyncratic" risk instead of as SSEC's
contribution.

This is a known, textbook property of Cholesky FEVD (ordering
dependence), not a coding error in the strict sense — but choosing to
use it, with variables in an arbitrary fetch order, and reporting the
result as "China-driven risk" without accounting for this, would have
been a real methodological error in the tool's headline output.

### Fix

Replaced the Cholesky FEVD with the Generalized FEVD (GFEVD, Pesaran &
Shin 1998), implemented manually in `_generalized_fevd()` since
`statsmodels` does not provide it directly. GFEVD does not orthogonalize
the shocks, so it does not privilege whichever variable happens to be
listed first; it decomposes variance using the raw residual covariance
matrix directly. Verified order-invariance by rerunning the same window
under all three orderings above — all three now produce identical
results (US-driven=19.81%, China-driven=19.74%, Idiosyncratic=60.45%).

One implementation detail: GFEVD rows do not sum to 1 by construction
(each row is normalized separately per Diebold-Yilmaz's standard
treatment).

### Why this bug was dangerous

Unlike the yfinance MultiIndex bug, this one never crashed and never
looked obviously wrong in isolation — 87% idiosyncratic risk is a
plausible-looking number for a small open equity market, not an
obvious red flag. It was caught only because a domain expert (the
paper's author) noticed the FEVD result was inconsistent with a
different, already-validated statistic (the correlation matrix) from
the same paper. Without that cross-check against an independent ground
truth, this would have shipped as the tool's headline "China-driven"
number, silently determined by an arbitrary implementation detail (the
order columns happened to come back from `MarketDataLoader.fetch()`)
rather than by the actual economics of the data.

### Reconciling Granger causality and GFEVD on SPX vs. SSEC

`validate_against_paper()`'s Granger results rank SPX far above SSEC as
a driver of HSI (F=211.63 vs. F=8.33). The rolling engine's GFEVD result
gives them comparable weight (SPX ≈ 16-20%, SSEC ≈ 19-20% depending on
the window). These are not in conflict — they are measuring different
transmission mechanisms, and the difference is informative rather than
a bug:

- **Granger causality tests lagged predictive power**: does yesterday's
  value of X help predict today's value of Y, beyond what Y's own past
  already predicts. HSI closes at 16:00 HKT, before the US market opens
  at 22:30 HKT the same calendar day — so SPX's lag-1 term genuinely
  captures "last night's US close → this morning's HK open," a real
  temporal lead (see the timezone discussion in CLAUDE.md). This is
  exactly the kind of relationship Granger causality is built to detect.

- **GFEVD includes contemporaneous (same-period) covariance**: HSI and
  SSEC trade in overlapping hours on the same calendar day, so a
  same-day shock hitting both markets shows up as a strong
  contemporaneous co-movement — which is precisely what the paper's
  correlation table also captures (SSEC-HSI at 0.542, the highest pair
  in Table 2). Granger causality, by definition, only looks at whether
  *past* values predict *future* values; it structurally cannot detect
  a same-day, same-session co-movement, because there is no lag between
  the two markets' trading hours for it to exploit.

So SPX's strength shows up in Granger (a real lagged, next-morning
effect) while SSEC's strength shows up in GFEVD (a real same-session
co-movement effect). Neither measure is wrong; they answer different
questions about how the two markets are connected to HSI, and a tool
that only reported one of them would be missing half of the actual
transmission structure.

## Edge threshold sensitivity check (Day 5) — result and decision

The Day 4 TODO was to rerun the full 2015-2026 rolling series (2386
windows, 2385 consecutive window-pairs) at EDGE_THRESHOLD = 0.5% / 1% /
2%, counting total edge appearance/disappearance events at each, before
trusting that signal.

### Results

| Threshold | Appeared | Disappeared | Total | Top 5 most volatile edges |
|---|---|---|---|---|
| 0.5% | 364 | 367 | 731 | all 5 involve USD_CNY |
| 1.0% | 340 | 342 | 682 | all 5 involve USD_CNY |
| 2.0% | 334 | 334 | 668 | USD_YIELD-SPX, SSEC-SPX, HSI-SPX, SPX-USD_CNY, HSI-USD_YIELD — USD_CNY almost entirely absent from the top 5 |

### Decision 1: threshold changed from 1% to 2%

**The reason is composition, not event count.** Total events only
dropped ~9% across the full 0.5%→2% range (731→668) — the raw count is
not sensitive to the exact threshold and could not by itself justify a
choice. What changes materially is *which* edges dominate the signal.
At 0.5% and 1%, the top 5 most volatile edges are almost entirely
USD_CNY pairs — the same channel already flagged as unverified against
the paper in the "Unverified Channel" section above. At those
thresholds, this signal was mostly re-detecting known noise in a
channel we already don't trust, not real structural change. At 2%, the
dominant edges shift to SPX/SSEC/HSI pairs — the paper-validated core
channels — crossing the threshold at what look like genuine moments of
change rather than noise. 2% was chosen for this reason, not because it
produces fewer alerts.

### Decision 2: side effect on Day 4's original safety-margin reasoning

Day 4 set the threshold at 1% specifically so that SPX→HSI (minimum
observed value 1.68% in this sample) could never cross into
"disappeared" territory. At 2%, that margin is gone: SPX→HSI's
historical minimum (1.68%) is now below the threshold, so a window at
or near that low would register as the core channel disappearing. This
is now the intended behavior, not a regression to guard against — see
docs/design.md section 6 for the full reasoning. Documenting this
explicitly because it is a change in what the threshold is *for*
(guarantee against false "disappeared" vs. permit true "disappeared"),
not a routine parameter adjustment.

### Decision 3: edge appearance/disappearance downgraded to a secondary signal

Independent of which exact threshold is used, this signal is weaker
than originally hoped, for two reasons visible in the data above:

1. **Composition is threshold-sensitive.** The set of edges driving the
   count changes substantially between 1% and 2% (see table above).
   Total-count stability alone does not make the signal robust if what
   it's actually measuring shifts underneath that stable-looking total.
2. **Base rate is too high for a "structural break" signal.** 668-731
   events over 2385 window-pairs is roughly one event every 3.3-3.5
   windows — far too frequent to serve as a "regime change" alert on
   its own; a signal that fires this often is closer to routine noise
   than to a structural break indicator.

This is consistent with the Day 1 design intent (docs/design.md
originally listed weight jumps as the *primary* signal and edge
appearance/disappearance as *secondary*) — the data confirms that
initial instinct rather than overturning it. RegimeDetector treats edge
appearance/disappearance as a secondary/supporting signal, not a
standalone trigger.

## RegimeDetector first run on full history (Day 5)

`RegimeDetector(threshold_pp=4.0, lookback=5)`, calibrated per section 8
of docs/design.md, run once against the full 2015-2026 rolling series
(2387 windows). **This is the first-run result, unmodified — no
parameters were adjusted after seeing it.**

### Totals

| Signal | Count |
|---|---|
| edge_change (secondary) | 1719 |
| share_jump (primary) | 62 |
| centrality_flip | 28 |
| **Total** | **1809** |

62 primary-signal alerts over ~11.5 years is roughly 5-6/year — a
plausible "worth a look" cadence rather than either silence or noise.

### Known-event check (looked at after the run, not used to tune it)

| Event | Window | Primary signal (share_jump) | Secondary (edge_change) |
|---|---|---|---|
| 2020 COVID | 2020-02-15 to 2020-04-15 | **Hit.** Multiple alerts; largest is 2020-03-09, US-driven +8.26pp over 5 windows (20.9%→29.2%) — one of the largest share_jump magnitudes in the entire history. Further alerts through 2020-03-20/23/27. | Also active |
| 2018 trade war | 2018-06-15 to 2018-08-15 | **Miss.** Zero share_jump alerts in this window. | 32 edge_change alerts (mostly USD_YIELD↔HSI edges appearing/disappearing) |
| 2022 rate hikes | 2022-02-01 to 2022-04-30 | **Near-miss.** Exactly 1 share_jump alert (2022-03-16, Idiosyncratic -4.2pp) out of 99 total alerts in the window. | 98 edge_change alerts |

COVID is a genuine hit; the trade war is a genuine miss (not a
near-threshold near-miss — zero primary alerts in the entire two-month
window); rate hikes barely register on the primary signal. Per the
project's standing rule, threshold_pp/lookback were not adjusted to try
to make the trade war or rate-hike periods show up — this is reported
as-is, including the negative result. See the "lookback experiment"
section below for a hypothesis-driven follow-up, not a parameter hunt.

## Lookback experiment: does a longer window catch gradual regime change?

### Hypothesis

The primary signal (`lookback=5`) measures a jump within one calendar
week. COVID was a sudden shock and fits that definition; the 2018 trade
war and 2022 rate-hike cycle were both multi-month processes. Hypothesis:
a 5-day window structurally cannot detect gradual drift, regardless of
`threshold_pp`, because a slow accumulation may never move more than
`threshold_pp` within any single 5-day slice even while moving far more
than that over months. This has a falsifiable prediction: a longer
`lookback` (same `threshold_pp=4.0`, only `lookback` changed) should
detect the trade war / rate-hike periods if the hypothesis is correct,
and should still miss them if it is wrong.

### Method

`RegimeDetector(threshold_pp=4.0, lookback=20)` and
`RegimeDetector(threshold_pp=4.0, lookback=60)`, each run once against
the same full rolling series. Only `lookback` was changed; `threshold_pp`
held fixed. Each was run exactly once and reported — not iterated
against the trade-war/rate-hike window to search for a value that "works."

### Result (exploratory — threshold_pp reused, not yet recalibrated)

| lookback | Total alerts | share_jump alerts | Trade war hits | Rate hikes hits |
|---|---|---|---|---|
| 5 (original) | 1809 | 62 | 0 | 1 |
| 20 | 4297 | 366 | 2 | 16 |
| 60 | 8828 | 1572 | 51 | 45 |

At `lookback=60`, the trade war window produced a strikingly clean
signature: China-driven share rose from ~16% to ~25% in a steady,
nearly monotonic climb across the entire June-August 2018 window — not
noise, a sustained drift. **Hypothesis mechanically confirmed**: a
longer lookback can detect gradual drift a 5-day window structurally
cannot.

**But this result reused `threshold_pp=4.0`, which was calibrated for
the 5-day distribution, not the 60-day one.** At `lookback=60`, 1572
share_jump alerts over 2387 windows is ~66% of all windows — far too
frequent to serve as a "regime change" signal on its own. The 4.0pp bar
that was rare at 5 days is common at 60 days, because ordinary drift
accumulates further over a longer horizon. This result established that
long-lookback detection is *possible*, but not that `threshold_pp=4.0`
was the right bar for it — see the calibration and episode-aggregation
work below.

## Dual-timescale RegimeDetector: calibration journey and final result

Two things were added after the lookback experiment above: (1) episode
aggregation, so a sustained drift produces one dated event instead of
one alert per day it stays past threshold, and (2) a properly
independent calibration of the chronic (60-day) threshold, rather than
reusing the acute (5-day) threshold as the exploratory run above did.
The chronic threshold went through two calibration attempts before
landing on the final one; both are recorded here because the first
one's result — and specifically what changed between it and the
second — is itself informative.

### Attempt 1: same rule as acute (p99 of the 60-day distribution)

Following the same method used for `acute_threshold_pp` (percentile of
`|share(t) - share(t-lookback)|`): at `lookback=60`, p99 was 9.62pp
(us_share), 8.05pp (china_share), 11.97pp (idio_share).
`chronic_threshold_pp=9.5` was set from this.

**Result: only COVID cleared this bar, at either timescale.** The 2018
trade war's actual china_share drift topped out at +7.98pp over 60
days — real, sustained, but short of a strict, self-referential "top 1%
of this sample's 60-day moves" cutoff. The 2022 rate-hike period's
moves were similarly short of 9.5pp. Both the trade war and rate-hike
detections from the exploratory (threshold=4.0) run above disappeared
once the threshold was calibrated rigorously.

### Attempt 2 (final): business-frequency calibration

The p99 rule was appropriate for the acute signal because, before
episode aggregation existed, a low bar meant alert flooding — the whole
point of picking something rare was to keep the log usable. **Episode
aggregation removes that constraint for the chronic signal**: a
sustained drift now produces one dated event regardless of how many
raw days it spans, so the threshold no longer needs to be pushed high
just to keep alert *volume* down. What it should instead target is
whether the resulting *episode frequency* matches how often a risk desk
would plausibly want a "reconsider the hedge" prompt — a business
question, not a statistical-rarity question. See docs/design.md section
9 for the full reasoning.

Candidate thresholds were run once each and their resulting chronic
episode frequency measured (10.51-year sample):

| threshold_pp | chronic episodes | episodes/year |
|---|---|---|
| 5.0 | 94 | 8.95 |
| 5.5 | 79 | 7.52 |
| 6.0 | 60 | 5.71 |
| 6.5 | 49 | 4.66 |
| **7.0** | **41** | **3.90** |
| 7.5 | 34 | 3.24 |
| 8.0 | 35 | 3.33 |

`chronic_threshold_pp=7.0` was chosen as the value landing cleanly
inside a target range of 3-5 episodes/year (a plausible cadence for a
periodic hedge-allocation review), then run once, final, without
further adjustment regardless of what it did or didn't detect.

### Final result (chronic_threshold_pp=7.0, acute_threshold_pp=4.0 unchanged)

| Signal | Count |
|---|---|
| edge_change (secondary) | 1719 |
| share_jump episodes, acute | 26 |
| share_jump episodes, chronic | 41 |
| centrality_flip | 28 |
| **Total** | **1814** |

| Event | Acute | Chronic |
|---|---|---|
| 2020 COVID | Hit — multiple episodes, e.g. US-driven 16.3%→25.9% over 2020-03-09 to 03-17 (+9.5pp, 7 trading days) | Hit — same window, plus an earlier +7.6pp single-day episode on 2020-02-24 |
| 2018 trade war | Miss | **Hit** — 3 episodes, all China-driven, e.g. 17.0%→24.6% over 2018-08-14 to 08-20 (+7.5pp, 5 trading days) |
| 2022 rate hikes | 1 episode (2022-03-16, Idiosyncratic -4.2pp) | 2 episodes, both starting 2022-04-28 (US-driven +7.0pp, Idiosyncratic -7.8pp) — a few days after this project's nominal window end (04-30), not cut to fit |

This is the actual, final, unmodified result of the last threshold
tested. No further tuning was done after this run.

## Three-layer conclusion

**(a) Fact.** Under a strict, self-referential p99 calibration (the
"top 1% of this 10.5-year sample's own 60-day moves"), only COVID
clears the bar at either timescale; the trade war's largest 60-day
china_share move (7.98pp) and the rate-hike period's moves fall short.
Under a business-frequency calibration targeting 3-5 chronic episodes
per year, the trade war is detected (3 episodes) and the rate-hike
period is partially detected (2 episodes, at its tail end).

**(b) Interpretation.** This is not "the p99 calibration was too
strict" as a criticism — p99 correctly measured what it was built to
measure. The deeper point: **a well-known news event and a statistical
anomaly in risk-attribution structure are two different things, and
this tool measures the second one, not the first.** COVID was a global,
simultaneous market freeze — an acute shock, and it registers as
extreme by construction. The trade war was a policy process that
markets priced in gradually over months through escalating tariff
announcements and retaliation cycles — real and economically
significant, but a smaller-magnitude, slower-moving shift in HSI's risk
attribution structure specifically, not a discontinuity of COVID's
scale. A regime-change detector that measures transmission-structure
anomalies is not the same instrument as a "was this in the news" check,
and conflating the two would be a category error.

**(c) Methodological limitation (important, not a footnote).** The p99
calibration is in-sample: COVID itself is part of the 10.5-year sample
used to define "what counts as the top 1%," so COVID partly defines the
bar it then passes — circular by construction. This is a real limit on
how much the p99-based acute threshold's rarity claim should be trusted
as evidence of genuine tail-event detection, as opposed to "detects
things at least as extreme as the most extreme thing already in the
training data." A more rigorous approach would calibrate out-of-sample
— e.g. an expanding window (calibrate only on data available up to each
point in time) or a rolling percentile — so a threshold is never partly
defined by the event it's being asked to detect. Not implemented in
this pass; see README Future Work.

## Note for Day 6 frontend

`edge_change` is 1719 of 1814 total alerts (95%). The default alert
view must show only `share_jump` episodes and `centrality_flip` —
`edge_change` belongs in a collapsed section or separate tab, or it
will drown out the primary signal entirely.

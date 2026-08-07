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

# HK Equity Risk Attribution Monitor

HK equity risk attribution monitoring system — full documentation will be completed on Day 7.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

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

# HK Equity Risk Attribution Monitor

HK equity risk attribution monitoring system — full documentation will be completed on Day 7.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

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

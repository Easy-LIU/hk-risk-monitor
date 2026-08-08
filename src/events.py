"""Hand-curated reference events for the frontend's time-series chart.

This is a static, manually maintained list — NOT a RegimeDetector
output. It exists to give the tool's detected alerts a historical
reference frame (see docs/design.md section 10). Every date here is
chosen from standard macro/market chronology, not because it lines up
with an interesting-looking move in the chart; the `source` field on
each entry states that basis so it can be checked.

category distinguishes two different kinds of event:
- "shock": a discrete event expected to produce a short-term jump in
  transmission shares (COVID crash, trade war escalation, rate hike
  cycle start).
- "structural": a change to the transmission channels themselves,
  expected to produce a gradual, lasting shift rather than a jump (the
  811 RMB reform changed how USD_CNY is set; Shenzhen-HK Stock Connect
  opened a new direct capital channel). See README "Sample Start Date
  and Extending the History" for why this distinction matters for this
  tool's 2015 sample start.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceEvent:
    date: str  # ISO 8601
    label: str
    category: str  # "shock" | "structural"
    source: str


REFERENCE_EVENTS: list[ReferenceEvent] = [
    ReferenceEvent(
        date="2015-08-11",
        label="RMB '811' Reform",
        category="structural",
        source=(
            "PBOC changed the USD/CNY central parity formation "
            "mechanism, devaluing the daily fix ~2% on this date and "
            "moving toward a more market-determined rate. Widely cited "
            "as 'the 811 reform' in FX literature."
        ),
    ),
    ReferenceEvent(
        date="2016-12-05",
        label="Shenzhen-HK Stock Connect Launch",
        category="structural",
        source=(
            "Official launch date of the Shenzhen-Hong Kong Stock "
            "Connect cross-border trading link, extending the direct "
            "capital channel between mainland and HK equity markets "
            "established by Shanghai-HK Connect in Nov 2014."
        ),
    ),
    ReferenceEvent(
        date="2018-06-15",
        label="US-China Trade War Tariff Escalation",
        category="shock",
        source=(
            "US Trade Representative published the first tariff list "
            "(25% on $34B of Chinese goods, effective July 6, 2018) on "
            "this date; standard reference point for the trade war's "
            "start in market commentary."
        ),
    ),
    ReferenceEvent(
        date="2020-03-09",
        label="COVID Market Crash",
        category="shock",
        source=(
            "Global equities fell sharply amid COVID fears and an "
            "oil price war; the S&P 500 triggered its first trading "
            "circuit breaker since 1997. Commonly referenced as "
            "'Black Monday I' of the 2020 crash."
        ),
    ),
    ReferenceEvent(
        date="2022-03-16",
        label="Fed Begins 2022 Hiking Cycle",
        category="shock",
        source=(
            "FOMC raised the federal funds rate 25bp at its March "
            "15-16, 2022 meeting, the first hike since 2018 and the "
            "start of the most aggressive tightening cycle in decades."
        ),
    ),
]

"""Structural break detection over a rolling series of RollingVAREngine
results. See docs/design.md section 8 for why the primary signal tracks
the aggregated US/China/Idiosyncratic headline shares (not raw edge
weights), and section 9 for the acute/chronic dual-timescale design and
episode aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.network import SpilloverNetwork
from src.var_engine import WindowResult

# Minimum trailing history (trading days) required before an expanding-window
# calibration is trusted enough to produce alerts. 252 ~= one trading year,
# matching the "12-month" convention already used elsewhere in this project
# (see app.py's TRAILING_WINDOW_DAYS). This is a floor, not a guarantee of
# stability: even at 252 observations, a p99 estimate rests on roughly the
# 2nd-3rd most extreme value seen so far -- an inherently noisy tail estimate
# that only gets more reliable as more years accumulate. See docs/design.md
# section 11.
MIN_CALIBRATION_HISTORY_DAYS = 252

# Search grid for expanding-window chronic threshold calibration. Wider than
# the single full-sample sweep in Day 5 (docs/notes.md) since an early-history
# calibration may genuinely need a different value than the full-sample 7.0pp.
CHRONIC_CANDIDATE_THRESHOLDS_PP = tuple(np.arange(3.0, 12.5, 0.5))
CHRONIC_TARGET_EPISODES_PER_YEAR = (3.0, 5.0)

SHARE_FIELDS = [
    ("us_share", "US-driven"),
    ("china_share", "China-driven"),
    ("idio_share", "Idiosyncratic"),
]

# The two candidates for "dominant risk source" — see docs/design.md /
# CLAUDE.md: hedging instrument choice is a discrete US-linked vs.
# China-linked decision, so this signal only tracks a flip between
# these two, not a full ranking over all five nodes.
DOMINANCE_CANDIDATES = ("SPX", "SSEC")


@dataclass
class Alert:
    date: pd.Timestamp  # reporting date: episode end_date, or trigger date for single-day signals
    signal_type: str  # "share_jump" | "edge_change" | "centrality_flip"
    magnitude: float  # percentage points, signed
    description: str
    # share_jump only: which timescale detected it. "acute" = short lookback,
    # sudden-shock framing (re-evaluate hedge now). "chronic" = long
    # lookback, gradual-drift framing (flag for periodic review).
    timescale: str | None = None
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    duration_days: int | None = None


@dataclass
class _RawCrossing:
    """One day's threshold crossing for one share field, before episode
    aggregation collapses consecutive same-direction crossings into a
    single Alert."""

    index: int
    date: pd.Timestamp
    field: str
    label: str
    before_val: float
    after_val: float
    direction: int  # +1 or -1


class RegimeDetector:
    """Scans a chronological series of WindowResults for three kinds of
    structural break: share jumps (primary, acute + chronic timescales,
    episode-aggregated), edge appearance/disappearance (secondary), and
    centrality rank flips."""

    def __init__(
        self,
        acute_threshold_pp: float = 4.0,
        acute_lookback: int = 5,
        chronic_threshold_pp: float = 7.0,
        chronic_lookback: int = 60,
    ):
        self.acute_threshold_pp = acute_threshold_pp
        self.acute_lookback = acute_lookback
        self.chronic_threshold_pp = chronic_threshold_pp
        self.chronic_lookback = chronic_lookback

    def scan(self, window_results: list[WindowResult]) -> list[Alert]:
        node_names = list(window_results[0].fevd_matrix.columns) if window_results else []
        networks = [SpilloverNetwork(w.fevd_matrix, node_names) for w in window_results]

        alerts: list[Alert] = []

        acute_raw = self._raw_share_crossings(
            window_results, self.acute_threshold_pp, self.acute_lookback
        )
        alerts.extend(self._aggregate_episodes(acute_raw, "acute"))

        chronic_raw = self._raw_share_crossings(
            window_results, self.chronic_threshold_pp, self.chronic_lookback
        )
        alerts.extend(self._aggregate_episodes(chronic_raw, "chronic"))

        alerts.extend(self._scan_edge_changes(window_results, networks, self.acute_lookback))
        alerts.extend(self._scan_centrality_flips(window_results, networks))

        return sorted(alerts, key=lambda a: a.date)

    def scan_out_of_sample(
        self,
        window_results: list[WindowResult],
        min_history_days: int = MIN_CALIBRATION_HISTORY_DAYS,
    ) -> tuple[list[Alert], list[dict]]:
        """Expanding-window version of the share_jump signal: each calendar
        year's acute/chronic thresholds are calibrated using only
        WindowResults strictly before that year (never the year's own data,
        and never future years), then applied to that year's alerts.
        Recalibration happens once per calendar year, not per day -- see
        docs/design.md section 11 for why annual recalibration is both
        sufficient for out-of-sample validity and far cheaper than daily.

        This exists to address the in-sample circularity of scan()'s
        threshold calibration (docs/notes.md): scan() calibrates thresholds
        on the full sample, so an event like COVID partly defines the bar
        it is then evaluated against. Here, a given year's threshold is
        fixed before that year's events happen.

        edge_change and centrality_flip are unaffected by this (they don't
        use a percentile/frequency-calibrated threshold) and are included
        unchanged, via the same logic as scan().

        Returns (alerts, calibration_report). calibration_report has one
        entry per calendar year present in window_results:
        {"year", "history_days", "status", "acute_threshold_pp",
        "chronic_threshold_pp"}, where status is "insufficient_history"
        (fewer than min_history_days of prior data -- no share_jump alerts
        possible that year, NOT the same as "no alerts occurred") or
        "calibrated".
        """
        years = sorted({w.date.year for w in window_results})
        year_calibration: dict[int, dict] = {}
        for year in years:
            historical = [w for w in window_results if w.date.year < year]
            history_days = len(historical)
            if history_days < min_history_days:
                year_calibration[year] = {
                    "year": year,
                    "history_days": history_days,
                    "status": "insufficient_history",
                    "acute_threshold_pp": None,
                    "chronic_threshold_pp": None,
                }
                continue
            year_calibration[year] = {
                "year": year,
                "history_days": history_days,
                "status": "calibrated",
                "acute_threshold_pp": self._calibrate_acute_from_history(historical),
                "chronic_threshold_pp": self._calibrate_chronic_from_history(historical),
            }

        acute_raw = self._raw_share_crossings_by_year(
            window_results, self.acute_lookback, year_calibration, "acute_threshold_pp"
        )
        chronic_raw = self._raw_share_crossings_by_year(
            window_results, self.chronic_lookback, year_calibration, "chronic_threshold_pp"
        )

        alerts: list[Alert] = []
        alerts.extend(self._aggregate_episodes(acute_raw, "acute"))
        alerts.extend(self._aggregate_episodes(chronic_raw, "chronic"))

        node_names = list(window_results[0].fevd_matrix.columns) if window_results else []
        networks = [SpilloverNetwork(w.fevd_matrix, node_names) for w in window_results]
        alerts.extend(self._scan_edge_changes(window_results, networks, self.acute_lookback))
        alerts.extend(self._scan_centrality_flips(window_results, networks))

        return sorted(alerts, key=lambda a: a.date), sorted(
            year_calibration.values(), key=lambda r: r["year"]
        )

    def _calibrate_acute_from_history(self, historical: list[WindowResult]) -> float:
        """p99 of pooled |share(t) - share(t-acute_lookback)| across all
        three shares, computed only from the given (already-prior-years)
        history. Same methodology as scan()'s fixed acute calibration
        (docs/design.md section 8), just re-run on less data each time."""
        lookback = self.acute_lookback
        deltas = []
        for field, _ in SHARE_FIELDS:
            series = np.array([getattr(w, field) for w in historical])
            deltas.extend(np.abs(series[lookback:] - series[:-lookback]) * 100)
        return float(np.percentile(deltas, 99))

    def _calibrate_chronic_from_history(self, historical: list[WindowResult]) -> float:
        """Business-frequency calibration (target 3-5 episodes/year), same
        methodology as scan()'s fixed chronic calibration (docs/design.md
        section 9), re-run on only the given prior-years history."""
        lookback = self.chronic_lookback
        span_years = (historical[-1].date - historical[0].date).days / 365.25
        if span_years <= 0:
            span_years = len(historical) / 252

        target_low, target_high = CHRONIC_TARGET_EPISODES_PER_YEAR
        target_mid = (target_low + target_high) / 2
        best_threshold, best_distance, best_in_range = None, None, False
        for threshold_pp in CHRONIC_CANDIDATE_THRESHOLDS_PP:
            raw = self._raw_share_crossings(historical, threshold_pp, lookback)
            episodes = self._aggregate_episodes(raw, "chronic")
            rate = len(episodes) / span_years
            in_range = target_low <= rate <= target_high
            distance = abs(rate - target_mid)
            is_better = best_threshold is None or (
                (in_range and not best_in_range)
                or (in_range == best_in_range and distance < best_distance)
            )
            if is_better:
                best_threshold, best_distance, best_in_range = threshold_pp, distance, in_range
        return float(best_threshold)

    def _raw_share_crossings_by_year(
        self,
        window_results: list[WindowResult],
        lookback: int,
        year_calibration: dict[int, dict],
        threshold_key: str,
    ) -> list["_RawCrossing"]:
        raw = []
        for i in range(lookback, len(window_results)):
            after = window_results[i]
            threshold_pp = year_calibration[after.date.year][threshold_key]
            if threshold_pp is None:
                continue  # insufficient history that year -- no alerts possible
            before = window_results[i - lookback]
            for field, label in SHARE_FIELDS:
                before_val = getattr(before, field)
                after_val = getattr(after, field)
                delta_pp = (after_val - before_val) * 100
                if abs(delta_pp) >= threshold_pp:
                    raw.append(
                        _RawCrossing(
                            index=i,
                            date=after.date,
                            field=field,
                            label=label,
                            before_val=before_val,
                            after_val=after_val,
                            direction=1 if delta_pp > 0 else -1,
                        )
                    )
        return raw

    def _raw_share_crossings(
        self, window_results: list[WindowResult], threshold_pp: float, lookback: int
    ) -> list[_RawCrossing]:
        raw = []
        for i in range(lookback, len(window_results)):
            before, after = window_results[i - lookback], window_results[i]
            for field, label in SHARE_FIELDS:
                before_val = getattr(before, field)
                after_val = getattr(after, field)
                delta_pp = (after_val - before_val) * 100
                if abs(delta_pp) >= threshold_pp:
                    raw.append(
                        _RawCrossing(
                            index=i,
                            date=after.date,
                            field=field,
                            label=label,
                            before_val=before_val,
                            after_val=after_val,
                            direction=1 if delta_pp > 0 else -1,
                        )
                    )
        return raw

    def _aggregate_episodes(self, raw: list[_RawCrossing], timescale: str) -> list[Alert]:
        """Collapse consecutive (by window index), same-field, same-direction
        crossings into one Alert per episode, so a sustained drift produces
        one dated event instead of one alert per day it stays past
        threshold."""
        by_field: dict[str, list[_RawCrossing]] = {}
        for crossing in raw:
            by_field.setdefault(crossing.field, []).append(crossing)

        episodes: list[Alert] = []
        for crossings in by_field.values():
            crossings.sort(key=lambda c: c.index)
            run = [crossings[0]]
            for prev, curr in zip(crossings, crossings[1:]):
                if curr.index == prev.index + 1 and curr.direction == prev.direction:
                    run.append(curr)
                else:
                    episodes.append(self._episode_alert(run, timescale))
                    run = [curr]
            episodes.append(self._episode_alert(run, timescale))
        return episodes

    def _episode_alert(self, run: list[_RawCrossing], timescale: str) -> Alert:
        first, last = run[0], run[-1]
        magnitude = (last.after_val - first.before_val) * 100
        duration_days = last.index - first.index + 1
        return Alert(
            date=last.date,
            signal_type="share_jump",
            magnitude=magnitude,
            description=(
                f"{first.label} share drifted {first.before_val * 100:.1f}% -> "
                f"{last.after_val * 100:.1f}% over {first.date.date()} to "
                f"{last.date.date()} ({magnitude:+.1f}pp, {duration_days} trading days)"
            ),
            timescale=timescale,
            start_date=first.date,
            end_date=last.date,
            duration_days=duration_days,
        )

    def _scan_edge_changes(
        self,
        window_results: list[WindowResult],
        networks: list[SpilloverNetwork],
        lookback: int,
    ) -> list[Alert]:
        alerts = []
        for i in range(lookback, len(window_results)):
            net_before, net_after = networks[i - lookback], networks[i]
            diff = net_before.diff(net_after)
            date = window_results[i].date
            for source, target in diff.edges_appeared:
                weight = net_after.get_edge_weight(source, target)
                alerts.append(
                    Alert(
                        date=date,
                        signal_type="edge_change",
                        magnitude=weight * 100,
                        description=f"edge {source}->{target} appeared (now {weight * 100:.1f}%)",
                    )
                )
            for source, target in diff.edges_disappeared:
                weight = net_before.get_edge_weight(source, target)
                alerts.append(
                    Alert(
                        date=date,
                        signal_type="edge_change",
                        magnitude=-weight * 100,
                        description=f"edge {source}->{target} disappeared (was {weight * 100:.1f}%)",
                    )
                )
        return alerts

    def _scan_centrality_flips(
        self, window_results: list[WindowResult], networks: list[SpilloverNetwork]
    ) -> list[Alert]:
        if not networks:
            return []

        def dominant(net: SpilloverNetwork):
            centralities = {n: net.out_degree_centrality(n) for n in DOMINANCE_CANDIDATES}
            return max(centralities, key=centralities.get), centralities

        alerts = []
        prev_dominant, _ = dominant(networks[0])
        for i in range(1, len(networks)):
            curr_dominant, centralities = dominant(networks[i])
            if curr_dominant != prev_dominant:
                gap = (centralities[curr_dominant] - centralities[prev_dominant]) * 100
                alerts.append(
                    Alert(
                        date=window_results[i].date,
                        signal_type="centrality_flip",
                        magnitude=gap,
                        description=(
                            f"dominant risk source flipped from {prev_dominant} to "
                            f"{curr_dominant} (lead {gap:.1f}pp)"
                        ),
                    )
                )
            prev_dominant = curr_dominant
        return alerts

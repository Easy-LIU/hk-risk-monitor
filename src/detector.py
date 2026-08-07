"""Structural break detection over a rolling series of RollingVAREngine
results. See docs/design.md section 8 for why the primary signal tracks
the aggregated US/China/Idiosyncratic headline shares (not raw edge
weights), and section 9 for the acute/chronic dual-timescale design and
episode aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.network import SpilloverNetwork
from src.var_engine import WindowResult

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
        chronic_threshold_pp: float = 9.5,
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

"""Tests for RegimeDetector: hand-constructed WindowResult sequences,
since exact episode boundaries/magnitudes need to be verifiable by hand.
Real-history validation (the acute/chronic calibration and the known
stress-period results) lives in docs/notes.md, not here.
"""

import numpy as np
import pandas as pd
import pytest

from src.detector import RegimeDetector
from src.network import SpilloverNetwork
from src.var_engine import WindowResult

NODE_NAMES = ["HSI", "SPX", "SSEC", "USD_CNY", "USD_YIELD"]


def identity_matrix() -> pd.DataFrame:
    """A neutral FEVD matrix (100% self, 0% everyone else) so edge_change
    and centrality_flip stay silent in tests focused on share_jump."""
    return pd.DataFrame(np.eye(len(NODE_NAMES)), index=NODE_NAMES, columns=NODE_NAMES)


def make_window(date, us_share, china_share, idio_share, fevd_matrix=None) -> WindowResult:
    return WindowResult(
        date=pd.Timestamp(date),
        fevd_matrix=identity_matrix() if fevd_matrix is None else fevd_matrix,
        us_share=us_share,
        china_share=china_share,
        idio_share=idio_share,
    )


def dates(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


def test_share_jump_merges_consecutive_same_direction_crossings_into_one_episode():
    # us_share: flat, then three consecutive +10pp jumps (a single event),
    # flat, then one isolated -30pp jump (a separate event).
    values = [0.10, 0.10, 0.20, 0.30, 0.40, 0.40, 0.40, 0.10, 0.10, 0.10]
    ds = dates(len(values))
    windows = [make_window(d, v, 0.50, 0.30) for d, v in zip(ds, values)]

    detector = RegimeDetector(
        acute_threshold_pp=5.0, acute_lookback=1,
        chronic_threshold_pp=100.0, chronic_lookback=100,  # structurally cannot fire
    )
    alerts = detector.scan(windows)
    us_alerts = [a for a in alerts if a.signal_type == "share_jump" and "US-driven" in a.description]

    assert len(us_alerts) == 2

    rising = us_alerts[0]
    assert rising.start_date == ds[2]
    assert rising.end_date == ds[4]
    assert rising.duration_days == 3
    assert rising.magnitude == pytest.approx(30.0)

    falling = us_alerts[1]
    assert falling.start_date == ds[7]
    assert falling.end_date == ds[7]
    assert falling.duration_days == 1
    assert falling.magnitude == pytest.approx(-30.0)


def test_acute_and_chronic_are_independent_timescales():
    # A slow, steady +1pp/day drift: too gradual for a short acute lookback
    # to ever see a big-enough jump, but a long chronic lookback sees the
    # cumulative move.
    n = 20
    values = [0.10 + 0.01 * i for i in range(n)]
    ds = dates(n)
    windows = [make_window(d, v, 0.50, 0.20) for d, v in zip(ds, values)]

    detector = RegimeDetector(
        acute_threshold_pp=5.0, acute_lookback=2,
        chronic_threshold_pp=8.0, chronic_lookback=10,
    )
    alerts = detector.scan(windows)
    share_jumps = [a for a in alerts if a.signal_type == "share_jump"]

    acute_alerts = [a for a in share_jumps if a.timescale == "acute"]
    chronic_alerts = [a for a in share_jumps if a.timescale == "chronic"]

    assert acute_alerts == []  # 2-day delta is always +2pp, never crosses 5pp
    assert len(chronic_alerts) == 1  # merged into a single sustained episode
    assert chronic_alerts[0].magnitude == pytest.approx(19.0)
    assert chronic_alerts[0].duration_days == 10


def test_edge_change_detects_appearance_and_disappearance():
    weak_matrix = identity_matrix().copy()
    weak_matrix.loc["HSI", "SPX"] = SpilloverNetwork.EDGE_THRESHOLD / 2  # SPX->HSI, below threshold

    strong_matrix = identity_matrix().copy()
    strong_matrix.loc["HSI", "SPX"] = SpilloverNetwork.EDGE_THRESHOLD + 0.03  # SPX->HSI, above threshold

    ds = dates(2)
    windows = [
        make_window(ds[0], 0.20, 0.20, 0.60, fevd_matrix=weak_matrix),
        make_window(ds[1], 0.20, 0.20, 0.60, fevd_matrix=strong_matrix),
    ]

    detector = RegimeDetector(
        acute_threshold_pp=1000.0, acute_lookback=1,  # share_jump can't fire
        chronic_threshold_pp=1000.0, chronic_lookback=100,
    )
    alerts = detector.scan(windows)
    edge_alerts = [a for a in alerts if a.signal_type == "edge_change"]

    assert len(edge_alerts) == 1
    assert edge_alerts[0].date == ds[1]
    assert "SPX->HSI" in edge_alerts[0].description
    assert "appeared" in edge_alerts[0].description


def test_centrality_flip_detects_dominance_switch_between_spx_and_ssec():
    spx_dominant = identity_matrix().copy()
    for target in ["HSI", "USD_CNY", "USD_YIELD"]:
        spx_dominant.loc[target, "SPX"] = 0.10
        spx_dominant.loc[target, "SSEC"] = 0.01

    ssec_dominant = identity_matrix().copy()
    for target in ["HSI", "USD_CNY", "USD_YIELD"]:
        ssec_dominant.loc[target, "SPX"] = 0.01
        ssec_dominant.loc[target, "SSEC"] = 0.10

    ds = dates(2)
    windows = [
        make_window(ds[0], 0.20, 0.20, 0.60, fevd_matrix=spx_dominant),
        make_window(ds[1], 0.20, 0.20, 0.60, fevd_matrix=ssec_dominant),
    ]

    detector = RegimeDetector(
        acute_threshold_pp=1000.0, acute_lookback=1,
        chronic_threshold_pp=1000.0, chronic_lookback=100,
    )
    alerts = detector.scan(windows)
    flips = [a for a in alerts if a.signal_type == "centrality_flip"]

    assert len(flips) == 1
    assert flips[0].date == ds[1]
    assert "SPX to SSEC" in flips[0].description


def test_scan_handles_empty_input():
    detector = RegimeDetector()
    assert detector.scan([]) == []


def test_alerts_are_sorted_by_date():
    values = [0.10, 0.30, 0.10, 0.40]  # jump up, jump back down
    ds = dates(len(values))
    windows = [make_window(d, v, 0.50, 0.20) for d, v in zip(ds, values)]

    detector = RegimeDetector(
        acute_threshold_pp=5.0, acute_lookback=1,
        chronic_threshold_pp=1000.0, chronic_lookback=100,
    )
    alerts = detector.scan(windows)

    assert [a.date for a in alerts] == sorted(a.date for a in alerts)


def _make_year(year: int, n: int, values: list[float]) -> list[WindowResult]:
    ds = pd.bdate_range(f"{year}-01-02", periods=n)
    return [make_window(d, v, 0.50, 1 - 0.50 - v) for d, v in zip(ds, values)]


def test_scan_out_of_sample_first_year_is_insufficient_history():
    year_2020 = _make_year(2020, 15, [0.20] * 15)
    detector = RegimeDetector(acute_lookback=1)

    _, report = detector.scan_out_of_sample(year_2020, min_history_days=10)

    assert len(report) == 1
    assert report[0]["year"] == 2020
    assert report[0]["history_days"] == 0
    assert report[0]["status"] == "insufficient_history"
    assert report[0]["acute_threshold_pp"] is None
    assert report[0]["chronic_threshold_pp"] is None


def test_scan_out_of_sample_no_alerts_during_insufficient_history():
    # Even a huge jump can't produce an alert without enough prior history
    # to calibrate a threshold against.
    year_2020 = _make_year(2020, 15, [0.20] * 7 + [0.60] + [0.20] * 7)
    detector = RegimeDetector(acute_lookback=1)

    alerts, report = detector.scan_out_of_sample(year_2020, min_history_days=10)

    assert report[0]["status"] == "insufficient_history"
    assert [a for a in alerts if a.signal_type == "share_jump"] == []


def test_scan_out_of_sample_calibrates_using_only_prior_years():
    # 2020: calm, small day-to-day noise only -- this is what 2021's
    # threshold must be calibrated from.
    year_2020 = _make_year(2020, 15, [0.20 + 0.001 * i for i in range(15)])
    # 2021: includes one large mid-year jump. If this year's own data leaked
    # into its threshold calibration, the test below comparing against a
    # calibration computed purely from 2020 would fail.
    year_2021_values = [0.20] * 5 + [0.35] + [0.20] * 4
    year_2021 = _make_year(2021, len(year_2021_values), year_2021_values)

    detector = RegimeDetector(acute_lookback=1, chronic_threshold_pp=1000.0, chronic_lookback=1000)
    alerts, report = detector.scan_out_of_sample(year_2020 + year_2021, min_history_days=10)

    report_by_year = {r["year"]: r for r in report}
    assert report_by_year[2020]["status"] == "insufficient_history"
    assert report_by_year[2021]["status"] == "calibrated"

    expected_threshold = detector._calibrate_acute_from_history(year_2020)
    assert report_by_year[2021]["acute_threshold_pp"] == pytest.approx(expected_threshold)

    acute_jumps = [a for a in alerts if a.signal_type == "share_jump" and a.timescale == "acute"]
    assert any(a.start_date.year == 2021 for a in acute_jumps)


def test_scan_out_of_sample_leaves_edge_and_centrality_signals_unchanged():
    # edge_change and centrality_flip don't use calibrated thresholds, so
    # scan() and scan_out_of_sample() should agree on them exactly.
    spx_dominant = identity_matrix().copy()
    for target in ["HSI", "USD_CNY", "USD_YIELD"]:
        spx_dominant.loc[target, "SPX"] = 0.10
        spx_dominant.loc[target, "SSEC"] = 0.01
    ssec_dominant = identity_matrix().copy()
    for target in ["HSI", "USD_CNY", "USD_YIELD"]:
        ssec_dominant.loc[target, "SPX"] = 0.01
        ssec_dominant.loc[target, "SSEC"] = 0.10

    ds = pd.bdate_range("2020-01-02", periods=20)
    matrices = [spx_dominant if i % 2 == 0 else ssec_dominant for i in range(20)]
    windows = [
        make_window(d, 0.20, 0.20, 0.60, fevd_matrix=m) for d, m in zip(ds, matrices)
    ]

    detector = RegimeDetector(acute_threshold_pp=1000.0, chronic_threshold_pp=1000.0)
    old_alerts = detector.scan(windows)
    new_alerts, _ = detector.scan_out_of_sample(windows, min_history_days=10)

    old_other = [a for a in old_alerts if a.signal_type != "share_jump"]
    new_other = [a for a in new_alerts if a.signal_type != "share_jump"]
    assert [(a.date, a.signal_type, a.description) for a in old_other] == [
        (a.date, a.signal_type, a.description) for a in new_other
    ]


def test_scan_out_of_sample_handles_empty_input():
    detector = RegimeDetector()
    alerts, report = detector.scan_out_of_sample([])
    assert alerts == []
    assert report == []

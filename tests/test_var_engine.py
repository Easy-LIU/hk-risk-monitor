"""Tests for RollingVAREngine's rolling-window logic (fit_window / run_rolling).

validate_against_paper() is exercised against live data in the Day 3
diagnostic scripts and docs/notes.md rather than here, since its
correctness is judged against the published paper's real-world numbers,
not a synthetic fixture.
"""

import numpy as np
import pandas as pd
import pytest

from src.var_engine import RollingVAREngine

COLUMNS = ["HSI", "SPX", "SSEC", "USD_CNY", "USD_YIELD"]


def make_synthetic_returns(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        rng.normal(scale=0.01, size=(n, len(COLUMNS))), index=dates, columns=COLUMNS
    )


def test_fit_window_shares_sum_to_one():
    engine = RollingVAREngine(window=60, lag=1, fevd_horizon=10)
    window = make_synthetic_returns(60)

    result = engine.fit_window(window)

    assert result.date == window.index[-1]
    total = result.us_share + result.china_share + result.idio_share
    assert total == pytest.approx(1.0)
    assert result.fevd_matrix.shape == (5, 5)
    # Every row of a FEVD matrix should also sum to 1.0.
    assert result.fevd_matrix.sum(axis=1).to_numpy() == pytest.approx(
        np.ones(5)
    )


def test_fit_window_grouping_matches_design_decision():
    engine = RollingVAREngine(window=60, lag=1, fevd_horizon=10)
    window = make_synthetic_returns(60)

    result = engine.fit_window(window)
    hsi_row = result.fevd_matrix.loc["HSI"]

    expected_us = hsi_row[["SPX", "USD_YIELD", "USD_CNY"]].sum()
    expected_china = hsi_row[["SSEC"]].sum()
    expected_idio = hsi_row[["HSI"]].sum()

    assert result.us_share == pytest.approx(expected_us)
    assert result.china_share == pytest.approx(expected_china)
    assert result.idio_share == pytest.approx(expected_idio)


def test_run_rolling_produces_expected_window_count():
    n, window, step = 120, 60, 5
    returns = make_synthetic_returns(n)
    engine = RollingVAREngine(window=window, step=step, lag=1, fevd_horizon=10)

    results = engine.run_rolling(returns)

    expected_count = len(range(0, n - window + 1, step))
    assert len(results) == expected_count
    assert [r.date for r in results] == sorted(r.date for r in results)


def test_run_rolling_report_matches_output():
    returns = make_synthetic_returns(120)
    engine = RollingVAREngine(window=60, step=10, lag=1, fevd_horizon=10)

    results = engine.run_rolling(returns)
    report = engine.get_rolling_report()

    assert report["total_windows"] == report["successful_windows"] + report["failed_windows"]
    assert report["successful_windows"] == len(results)
    assert report["failure_rate"] == pytest.approx(
        report["failed_windows"] / report["total_windows"]
    )


def test_run_rolling_skips_numerically_unstable_windows():
    # A column with zero variance makes the residual covariance matrix
    # singular, which should be caught rather than raised.
    n, window = 120, 60
    returns = make_synthetic_returns(n)
    returns["USD_CNY"] = 0.0

    engine = RollingVAREngine(window=window, step=10, lag=1, fevd_horizon=10)
    results = engine.run_rolling(returns)
    report = engine.get_rolling_report()

    assert report["failed_windows"] == report["total_windows"]
    assert len(results) == 0
    assert all("date" in failure and "error" in failure for failure in report["failures"])


def test_get_rolling_report_before_run_rolling_raises():
    engine = RollingVAREngine(window=60, lag=1, fevd_horizon=10)
    with pytest.raises(RuntimeError):
        engine.get_rolling_report()

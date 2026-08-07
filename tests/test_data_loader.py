"""Tests for MarketDataLoader, focused on trading calendar alignment."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import MarketDataLoader

COLUMNS = ["HSI", "SPX", "SSEC", "USD_CNY", "USD_YIELD"]


def make_loader() -> MarketDataLoader:
    return MarketDataLoader(start="2020-01-01", end="2020-01-10")


def test_align_calendars_keeps_only_common_trading_days():
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    )
    raw = pd.DataFrame(
        {
            "HSI": [1.0, 2.0, np.nan, 4.0, 5.0],  # HK holiday on 01-03
            "SPX": [1.0, 2.0, 3.0, np.nan, 5.0],  # US holiday on 01-06
            "SSEC": [1.0, 2.0, 3.0, 4.0, 5.0],
            "USD_CNY": [1.0, 2.0, 3.0, 4.0, 5.0],
            "USD_YIELD": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=dates,
    )

    aligned = make_loader().align_calendars(raw)

    expected_dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-07"])
    assert list(aligned.index) == list(expected_dates)
    assert aligned.isna().sum().sum() == 0


def test_get_alignment_report_matches_align_calendars_output():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    raw = pd.DataFrame(
        {col: np.arange(10, dtype=float) for col in COLUMNS}, index=dates
    )
    raw.loc[dates[3], "HSI"] = np.nan
    raw.loc[dates[7], "SSEC"] = np.nan

    loader = make_loader()
    loader.align_calendars(raw)
    report = loader.get_alignment_report()

    assert report["raw_days"] == 10
    assert report["days_lost"] == 2
    assert report["aligned_days"] == 8
    assert report["pct_lost"] == pytest.approx(0.2)


def test_get_alignment_report_before_align_calendars_raises():
    with pytest.raises(RuntimeError):
        make_loader().get_alignment_report()


def test_to_log_returns_matches_known_values():
    prices = pd.DataFrame(
        {"HSI": [100.0, 110.0, 121.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    returns = make_loader().to_log_returns(prices)

    assert len(returns) == 2
    assert returns["HSI"].iloc[0] == pytest.approx(np.log(1.1))
    assert returns["HSI"].iloc[1] == pytest.approx(np.log(1.1))


def test_check_stationarity_returns_expected_keys():
    rng = np.random.default_rng(seed=0)
    returns = pd.DataFrame(
        {"HSI": rng.normal(size=300), "SPX": rng.normal(size=300)}
    )
    report = make_loader().check_stationarity(returns)

    assert set(report.keys()) == {"HSI", "SPX"}
    for stats in report.values():
        assert set(stats.keys()) == {"adf_stat", "p_value", "is_stationary"}
        assert isinstance(stats["is_stationary"], (bool, np.bool_))

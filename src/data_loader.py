"""Fetching, calendar alignment, and stationarity testing for the five
HK Equity Risk Attribution Monitor market variables (HSI, SPX, SSEC,
USD_CNY, USD_YIELD)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller

load_dotenv()

YFINANCE_TICKERS = {
    "HSI": "^HSI",
    "SPX": "^GSPC",
    "SSEC": "000001.SS",
    "USD_CNY": "USDCNY=X",
}

FRED_SERIES_ID = "DGS10"
FRED_COLUMN = "USD_YIELD"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

ADF_SIGNIFICANCE_LEVEL = 0.05


class MarketDataLoader:
    """Loads and prepares the five market variables used by the VAR engine."""

    def __init__(self, start: str, end: str, fred_api_key: str | None = None):
        self.start = start
        self.end = end
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self._alignment_report: dict | None = None

    def fetch(self) -> pd.DataFrame:
        """Fetch raw daily close prices for all five variables into one
        wide DataFrame, indexed by calendar date. Days a given market was
        closed are NaN in that column."""
        columns = {
            name: self._fetch_yfinance(ticker)
            for name, ticker in YFINANCE_TICKERS.items()
        }
        columns[FRED_COLUMN] = self._fetch_fred()
        raw = pd.concat(columns, axis=1)
        raw.index.name = "date"
        return raw.sort_index()

    def _fetch_yfinance(self, ticker: str) -> pd.Series:
        data = yf.download(ticker, start=self.start, end=self.end, progress=False)
        close = data["Close"]
        # Recent yfinance versions return a (Price, Ticker) MultiIndex even
        # for a single ticker, so data["Close"] is a one-column DataFrame
        # rather than a Series. Normalize to a plain Series either way.
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.rename(None)

    def _fetch_fred(self) -> pd.Series:
        if not self.fred_api_key:
            raise RuntimeError(
                "FRED_API_KEY is not set. Add it to a local .env file "
                "(see .env.example) before calling fetch()."
            )
        params = {
            "series_id": FRED_SERIES_ID,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": self.start,
            "observation_end": self.end,
        }
        response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
        response.raise_for_status()
        observations = response.json()["observations"]
        dates = pd.to_datetime([obs["date"] for obs in observations])
        values = [
            np.nan if obs["value"] == "." else float(obs["value"])
            for obs in observations
        ]
        return pd.Series(values, index=dates)

    def align_calendars(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Inner join across all five columns: keep only the days every
        market has a value. Records the alignment stats for
        get_alignment_report()."""
        raw_days = len(raw)
        aligned = raw.dropna(how="any")
        aligned_days = len(aligned)
        days_lost = raw_days - aligned_days
        self._alignment_report = {
            "raw_days": raw_days,
            "aligned_days": aligned_days,
            "days_lost": days_lost,
            "pct_lost": days_lost / raw_days if raw_days else 0.0,
        }
        return aligned

    def to_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert a price-level DataFrame to log returns."""
        return np.log(prices / prices.shift(1)).dropna(how="any")

    def check_stationarity(self, returns: pd.DataFrame) -> dict:
        """Run an ADF test on each column. is_stationary is True when
        p_value < ADF_SIGNIFICANCE_LEVEL."""
        report = {}
        for column in returns.columns:
            adf_stat, p_value, *_ = adfuller(returns[column].dropna())
            report[column] = {
                "adf_stat": adf_stat,
                "p_value": p_value,
                "is_stationary": p_value < ADF_SIGNIFICANCE_LEVEL,
            }
        return report

    def get_alignment_report(self) -> dict:
        if self._alignment_report is None:
            raise RuntimeError(
                "align_calendars() must be called before get_alignment_report()."
            )
        return self._alignment_report

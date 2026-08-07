"""Rolling VAR / FEVD engine for the HK Equity Risk Attribution Monitor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Key results from the author's published paper, Table 3 (full-sample
# Granger causality F-statistics). Used as a correctness check, not an
# exact-match target: the paper's sample (Investing.com, through
# end-2024) differs from this tool's (yfinance, through 2026) in source
# and period, so agreement is judged on order of magnitude and relative
# ranking rather than digit-for-digit equality.
PAPER_TABLE_3 = {
    ("SPX", "HSI"): 155.65,
    ("USD_CNY", "HSI"): 36.61,
    ("SSEC", "HSI"): 11.32,
    ("USD_YIELD", "USD_CNY"): 16.58,
}

TARGET_VARIABLE = "HSI"

# FEVD attribution grouping for the dashboard's three headline shares.
# USD_CNY is grouped under US-driven, not China-driven — see
# docs/design.md section 5 for the rationale: validate_against_paper()
# shows USD_YIELD -> USD_CNY is a strong, verified channel, so USD_CNY's
# variation is substantially a downstream effect of US monetary policy
# rather than an independent China-side shock.
US_DRIVEN_VARIABLES = {"SPX", "USD_YIELD", "USD_CNY"}
CHINA_DRIVEN_VARIABLES = {"SSEC"}
IDIOSYNCRATIC_VARIABLES = {"HSI"}


def _generalized_fevd(results, horizon: int) -> np.ndarray:
    """Generalized FEVD (Pesaran & Shin, 1998).

    statsmodels' built-in .fevd() is Cholesky-based, which means it is
    sensitive to variable ordering: whichever variable is placed first
    absorbs all contemporaneous correlation with the others, since the
    Cholesky factorization attributes shared same-period covariance
    entirely to the first variable in the ordering. HSI and SSEC trade
    in the same session and move together contemporaneously, so with
    HSI first this manifested as HSI's own ("idiosyncratic") share
    absorbing SSEC's true contribution — see docs/notes.md for the
    diagnostic that found this. GFEVD is order-invariant because it
    does not orthogonalize the shocks; it decomposes variance using the
    raw (non-orthogonalized) residual covariance matrix directly.

    Per Diebold-Yilmaz, GFEVD rows do not sum to 1 by construction and
    must be row-normalized afterward; this function returns the
    normalized matrix.
    """
    sigma_u = np.asarray(results.sigma_u)
    phi = results.ma_rep(horizon - 1)  # shape (horizon, k, k); phi[0] = identity
    k = sigma_u.shape[0]
    sigma_jj = np.diag(sigma_u)

    numerator = np.zeros((k, k))
    denominator = np.zeros(k)
    for phi_h in phi:
        phi_sigma = phi_h @ sigma_u
        numerator += phi_sigma**2
        denominator += np.diag(phi_h @ sigma_u @ phi_h.T)

    # Division by zero is expected and handled explicitly below (a
    # near-zero-variance variable drives sigma_jj/denominator to zero);
    # suppress numpy's warning since the resulting NaN/inf is checked for.
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = numerator / (denominator[:, None] * sigma_jj[None, :])
        normalized = theta / theta.sum(axis=1, keepdims=True)

    if not np.all(np.isfinite(normalized)):
        # A near-zero-variance variable (e.g. a degenerate window) drives
        # sigma_jj or the denominator to zero, which this manual numpy
        # computation turns into NaN/inf instead of raising the way
        # statsmodels' own linear algebra calls would. Surface it as the
        # same kind of numerical-instability failure run_rolling() already
        # catches, rather than letting NaNs flow silently into the shares.
        raise ValueError(
            "Generalized FEVD produced non-finite values, likely due to a "
            "near-zero-variance variable in this window."
        )
    return normalized


@dataclass
class WindowResult:
    """The output of a single rolling window: the full FEVD matrix plus
    HSI's decomposition rolled up into the three dashboard categories."""

    date: pd.Timestamp
    fevd_matrix: pd.DataFrame  # rows=target variable, cols=source variable, each row sums to 1.0
    us_share: float
    china_share: float
    idio_share: float


class RollingVAREngine:
    """Fits VAR models on the five aligned market-return series."""

    def __init__(
        self,
        window: int = 250,
        step: int = 1,
        lag: int = 1,
        fevd_horizon: int = 10,
    ):
        self.window = window
        self.step = step
        self.lag = lag
        self.fevd_horizon = fevd_horizon
        self._rolling_report: dict | None = None

    def validate_against_paper(self, returns: pd.DataFrame) -> dict:
        """Fit a single VAR(lag) model on the full sample (no rolling) and
        run pairwise Granger causality tests for every ordered variable
        pair. Returns both the full pairwise result table and a
        comparison table against PAPER_TABLE_3."""
        model = VAR(returns)
        results = model.fit(self.lag)

        variables = list(returns.columns)
        rows = []
        for causing in variables:
            for caused in variables:
                if causing == caused:
                    continue
                test = results.test_causality(
                    caused=caused, causing=causing, kind="f"
                )
                rows.append(
                    {
                        "cause": causing,
                        "effect": caused,
                        "f_stat": test.test_statistic,
                        "p_value": test.pvalue,
                    }
                )
        all_pairs = pd.DataFrame(rows)

        comparison_rows = []
        for (causing, caused), paper_f in PAPER_TABLE_3.items():
            match = all_pairs[
                (all_pairs["cause"] == causing) & (all_pairs["effect"] == caused)
            ].iloc[0]
            comparison_rows.append(
                {
                    "pair": f"{causing}→{caused}",
                    "paper_f": paper_f,
                    "tool_f": match["f_stat"],
                    "tool_p": match["p_value"],
                    "ratio": match["f_stat"] / paper_f,
                }
            )
        comparison = pd.DataFrame(comparison_rows)

        return {"all_pairs": all_pairs, "comparison": comparison}

    def fit_window(self, data_slice: pd.DataFrame) -> WindowResult:
        """Fit a VAR(lag) on a single window and decompose HSI's forecast
        error variance at self.fevd_horizon steps ahead, using the
        order-invariant Generalized FEVD (see _generalized_fevd)."""
        model = VAR(data_slice)
        results = model.fit(self.lag)

        variables = list(results.names)
        decomp_at_horizon = _generalized_fevd(results, self.fevd_horizon)
        fevd_matrix = pd.DataFrame(decomp_at_horizon, index=variables, columns=variables)

        hsi_row = fevd_matrix.loc[TARGET_VARIABLE]
        us_share = hsi_row[hsi_row.index.isin(US_DRIVEN_VARIABLES)].sum()
        china_share = hsi_row[hsi_row.index.isin(CHINA_DRIVEN_VARIABLES)].sum()
        idio_share = hsi_row[hsi_row.index.isin(IDIOSYNCRATIC_VARIABLES)].sum()

        return WindowResult(
            date=data_slice.index[-1],
            fevd_matrix=fevd_matrix,
            us_share=us_share,
            china_share=china_share,
            idio_share=idio_share,
        )

    def run_rolling(self, returns: pd.DataFrame) -> list[WindowResult]:
        """Slide a self.window-sized window across returns, self.step days
        at a time, calling fit_window() on each. Numerically unstable
        windows are skipped rather than aborting the run; see
        get_rolling_report() for how many were skipped."""
        window_starts = list(range(0, len(returns) - self.window + 1, self.step))

        results: list[WindowResult] = []
        failures: list[dict] = []
        for start in window_starts:
            window_slice = returns.iloc[start : start + self.window]
            try:
                results.append(self.fit_window(window_slice))
            except (np.linalg.LinAlgError, ValueError) as exc:
                failures.append({"date": window_slice.index[-1], "error": str(exc)})

        total_windows = len(window_starts)
        self._rolling_report = {
            "total_windows": total_windows,
            "successful_windows": len(results),
            "failed_windows": len(failures),
            "failure_rate": len(failures) / total_windows if total_windows else 0.0,
            "failures": failures,
        }
        return results

    def get_rolling_report(self) -> dict:
        if self._rolling_report is None:
            raise RuntimeError(
                "run_rolling() must be called before get_rolling_report()."
            )
        return self._rolling_report

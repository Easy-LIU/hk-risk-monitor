"""Rolling VAR / FEVD engine for the HK Equity Risk Attribution Monitor.

Day 3, step 1: only validate_against_paper() is implemented here. The
rolling window logic (fit_window / run_rolling) is added only after this
full-sample validation confirms the engine reproduces the published
paper's Granger causality results at the right order of magnitude.
"""

from __future__ import annotations

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


class RollingVAREngine:
    """Fits VAR models on the five aligned market-return series."""

    def __init__(self, window: int = 250, step: int = 1, lag: int = 1):
        self.window = window
        self.step = step
        self.lag = lag

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

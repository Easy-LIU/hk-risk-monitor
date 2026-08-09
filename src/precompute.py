"""Offline precompute script.

Runs the full pipeline (MarketDataLoader -> RollingVAREngine ->
RegimeDetector) once and writes the results to cache/. The Streamlit
frontend (app.py) only ever reads these cache files -- it never calls
yfinance/FRED or re-runs the rolling computation, which also makes the
frontend deployable to an environment (e.g. Streamlit Community Cloud)
that can't hold API keys or make outbound calls to those APIs at
request time.

Run with: python -m src.precompute
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_loader import MarketDataLoader
from src.detector import MIN_CALIBRATION_HISTORY_DAYS, RegimeDetector
from src.network import SpilloverNetwork
from src.var_engine import RollingVAREngine

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

SAMPLE_START = "2015-01-01"

WINDOW = 250
STEP = 1
LAG = 1
FEVD_HORIZON = 10

# acute/chronic lookback are fixed; the thresholds themselves are no longer
# fixed constants -- RegimeDetector.scan_out_of_sample() calibrates them
# per calendar year, out-of-sample (see docs/design.md section 11 and
# docs/notes.md for why this replaced the original fixed-threshold scan()).
ACUTE_LOOKBACK = 5
CHRONIC_LOOKBACK = 60


def _flatten_fevd(matrix: pd.DataFrame, node_names: list[str]) -> dict:
    return {
        f"fevd__{target}__{source}": matrix.loc[target, source]
        for target in node_names
        for source in node_names
    }


def _write_rolling_results(window_results, node_names: list[str]) -> None:
    rows = []
    for w in window_results:
        row = {
            "date": w.date,
            "us_share": w.us_share,
            "china_share": w.china_share,
            "idio_share": w.idio_share,
        }
        row.update(_flatten_fevd(w.fevd_matrix, node_names))
        rows.append(row)
    pd.DataFrame(rows).to_parquet(CACHE_DIR / "rolling_results.parquet", index=False)


def _write_alerts(alerts) -> None:
    rows = [asdict(a) for a in alerts]
    pd.DataFrame(rows).to_parquet(CACHE_DIR / "alerts.parquet", index=False)


def _write_metadata(
    sample_end: str,
    alignment_report: dict,
    rolling_report: dict,
    validation: dict,
    node_names: list[str],
    calibration_report: list[dict],
) -> None:
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_start": SAMPLE_START,
        "sample_end": sample_end,
        "node_names": node_names,
        "params": {
            "window": WINDOW,
            "step": STEP,
            "lag": LAG,
            "fevd_horizon": FEVD_HORIZON,
            "edge_threshold": SpilloverNetwork.EDGE_THRESHOLD,
            "acute_lookback": ACUTE_LOOKBACK,
            "chronic_lookback": CHRONIC_LOOKBACK,
            "min_calibration_history_days": MIN_CALIBRATION_HISTORY_DAYS,
        },
        "alignment_report": alignment_report,
        "rolling_report": rolling_report,
        "paper_validation": validation["comparison"].to_dict(orient="records"),
        # share_jump thresholds are calibrated out-of-sample per calendar
        # year (RegimeDetector.scan_out_of_sample) -- see docs/design.md
        # section 11 -- so there is no longer a single fixed
        # acute/chronic_threshold_pp to report; this table replaces it.
        "regime_detector_calibration": calibration_report,
    }
    with open(CACHE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def run_pipeline() -> None:
    sample_end = datetime.now().strftime("%Y-%m-%d")

    t0 = time.time()
    loader = MarketDataLoader(start=SAMPLE_START, end=sample_end)
    raw = loader.fetch()
    aligned = loader.align_calendars(raw)
    alignment_report = loader.get_alignment_report()
    returns = loader.to_log_returns(aligned)
    print(f"[1/4] Data loaded and aligned in {time.time() - t0:.1f}s: {alignment_report}")

    t0 = time.time()
    engine = RollingVAREngine(window=WINDOW, step=STEP, lag=LAG, fevd_horizon=FEVD_HORIZON)
    validation = engine.validate_against_paper(returns)
    window_results = engine.run_rolling(returns)
    rolling_report = engine.get_rolling_report()
    print(f"[2/4] Rolling VAR/FEVD complete in {time.time() - t0:.1f}s: {rolling_report}")

    t0 = time.time()
    detector = RegimeDetector(acute_lookback=ACUTE_LOOKBACK, chronic_lookback=CHRONIC_LOOKBACK)
    alerts, calibration_report = detector.scan_out_of_sample(
        window_results, min_history_days=MIN_CALIBRATION_HISTORY_DAYS
    )
    print(f"[3/4] RegimeDetector (out-of-sample) produced {len(alerts)} alerts in {time.time() - t0:.1f}s")

    t0 = time.time()
    node_names = list(window_results[0].fevd_matrix.columns)
    CACHE_DIR.mkdir(exist_ok=True)
    _write_rolling_results(window_results, node_names)
    _write_alerts(alerts)
    _write_metadata(
        sample_end, alignment_report, rolling_report, validation, node_names, calibration_report
    )
    print(f"[4/4] Cache written to {CACHE_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_pipeline()

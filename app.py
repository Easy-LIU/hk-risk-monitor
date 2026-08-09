"""Streamlit frontend for the HK Equity Risk Attribution Monitor.

Read-only: this app never fetches market data or runs the VAR/FEVD
pipeline live. It only reads cache/ (produced by `python -m
src.precompute`). See docs/design.md sections 2 and 9-10 for why.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.events import REFERENCE_EVENTS
from src.network import SpilloverNetwork

CACHE_DIR = Path(__file__).resolve().parent / "cache"

EVENT_STYLE = {
    "shock": {"line_dash": "dot", "line_color": "#e67e22"},
    "structural": {"line_dash": "dash", "line_color": "#8e44ad"},
}

SHARE_COLORS = {
    "us_share": "#1f77b4",
    "china_share": "#d62728",
    "idio_share": "#7f7f7f",
}
SHARE_LABELS = {
    "us_share": "US-Driven",
    "china_share": "China-Driven",
    "idio_share": "Idiosyncratic",
}


st.set_page_config(page_title="HK Equity Risk Attribution Monitor", layout="wide")


def _cache_missing() -> bool:
    required = ["rolling_results.parquet", "alerts.parquet", "metadata.json"]
    return not all((CACHE_DIR / name).exists() for name in required)


@st.cache_data
def load_rolling_results() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / "rolling_results.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_alerts() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / "alerts.parquet")
    for col in ["date", "start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


@st.cache_data
def load_metadata() -> dict:
    with open(CACHE_DIR / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def build_share_chart(rolling: pd.DataFrame, selected_date: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    for field in ["us_share", "china_share", "idio_share"]:
        fig.add_trace(
            go.Scatter(
                x=rolling["date"],
                y=rolling[field] * 100,
                name=SHARE_LABELS[field],
                line=dict(color=SHARE_COLORS[field]),
            )
        )

    for event in REFERENCE_EVENTS:
        style = EVENT_STYLE[event.category]
        fig.add_vline(
            x=pd.Timestamp(event.date).timestamp() * 1000,
            line_dash=style["line_dash"],
            line_color=style["line_color"],
            opacity=0.6,
            annotation_text=event.label,
            annotation_textangle=-90,
            annotation_position="top",
            annotation_font_size=10,
        )

    fig.add_vline(x=selected_date.timestamp() * 1000, line_color="black", line_width=1)

    fig.update_layout(
        yaxis_title="Share of HSI forecast error variance (%)",
        xaxis_title="Date",
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


TRAILING_WINDOW_DAYS = 252  # ~12 months of trading days


def build_kpi_observation(
    rolling: pd.DataFrame,
    field: str,
    label: str,
    counterpart_field: str | None = None,
    counterpart_label: str | None = None,
) -> str:
    """Purely observational: current level relative to its own recent and
    full-sample history, plus (if a counterpart is given) what the other
    share has historically averaged when this one is in its current
    below/above-median regime. Never a recommendation -- see docs/design.md
    section 10 / CLAUDE.md's "reference events, not detected" principle,
    which this extends to "describe state, don't prescribe action."
    """
    series = rolling[field]
    current = series.iloc[-1]

    trailing = series.iloc[-TRAILING_WINDOW_DAYS:] if len(series) >= TRAILING_WINDOW_DAYS else series
    pct_rank = (trailing < current).mean() * 100
    if pct_rank <= 10:
        recency_phrase = "near its 12-month low"
    elif pct_rank >= 90:
        recency_phrase = "near its 12-month high"
    else:
        recency_phrase = "within its typical 12-month range"

    sentence = f"Currently {current * 100:.1f}%, {recency_phrase}."

    if counterpart_field is not None:
        median_val = series.median()
        below_median = current < median_val
        regime_mask = series < median_val if below_median else series >= median_val
        conditional_mean = rolling.loc[regime_mask, counterpart_field].mean()
        overall_mean = rolling[counterpart_field].mean()
        relation = "below" if below_median else "at or above"
        sentence += (
            f" Over the full history, when {label} was {relation} its median "
            f"({median_val * 100:.1f}%), {counterpart_label} averaged "
            f"{conditional_mean * 100:.1f}% (vs {overall_mean * 100:.1f}% overall)."
        )

    return sentence


# "Reproduces" = within 3x of the paper's F-statistic in either direction
# and statistically significant. Same order-of-magnitude check used
# throughout docs/notes.md's diagnostics, not an exact-match bar.
REPRODUCES_RATIO_BOUNDS = (1 / 3, 3.0)
REPRODUCES_P_THRESHOLD = 0.05


def build_validation_table(paper_validation: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(paper_validation)
    lower, upper = REPRODUCES_RATIO_BOUNDS
    reproduces = (df["ratio"].between(lower, upper)) & (df["tool_p"] < REPRODUCES_P_THRESHOLD)
    df["verdict"] = reproduces.map(
        {
            True: "Reproduces at similar magnitude",
            False: "Does not reproduce (unverified)",
        }
    )
    return df


def _reconstruct_fevd_matrix(row: pd.Series, node_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[row[f"fevd__{target}__{source}"] for source in node_names] for target in node_names],
        index=node_names,
        columns=node_names,
    )


def build_network_figure(network: SpilloverNetwork) -> go.Figure:
    n = len(network.node_names)
    angle_step = 2 * math.pi / n
    positions = {
        name: (math.cos(i * angle_step), math.sin(i * angle_step))
        for i, name in enumerate(network.node_names)
    }

    fig = go.Figure()

    edge_mid_x, edge_mid_y, edge_hover = [], [], []
    for source in network.node_names:
        for target in network.node_names:
            if source == target:
                continue
            weight = network.get_edge_weight(source, target)
            if weight <= SpilloverNetwork.EDGE_THRESHOLD:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            # Pull both ends in from the node centers so the arrowhead is
            # visible instead of hidden under the target node's marker.
            dx, dy = x1 - x0, y1 - y0
            margin = 0.18
            ax, ay = x0 + dx * margin, y0 + dy * margin
            tx, ty = x1 - dx * margin, y1 - dy * margin
            fig.add_annotation(
                x=tx,
                y=ty,
                ax=ax,
                ay=ay,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=max(1.0, weight * 12),
                arrowcolor="rgba(31,119,180,0.65)",
                text="",
            )
            # Arrows (annotations) aren't hoverable in Plotly, so place a
            # small marker at each edge's midpoint (offset 65% toward the
            # arrowhead, closer to target than source) purely to carry a
            # hover tooltip with the exact weight.
            edge_mid_x.append(x0 + dx * 0.65)
            edge_mid_y.append(y0 + dy * 0.65)
            edge_hover.append(f"{source} → {target}: {weight:.1%}")

    if edge_mid_x:
        fig.add_trace(
            go.Scatter(
                x=edge_mid_x,
                y=edge_mid_y,
                mode="markers",
                marker=dict(size=14, color="rgba(31,119,180,0.01)"),
                hoverinfo="text",
                hovertext=edge_hover,
                showlegend=False,
            )
        )

    xs = [positions[name][0] for name in network.node_names]
    ys = [positions[name][1] for name in network.node_names]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=network.node_names,
            textposition="top center",
            textfont=dict(size=13),
            marker=dict(size=32, color="#2ca02c"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6]),
        height=420,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


def main():
    if _cache_missing():
        st.error(
            "Cache not found. Run `python -m src.precompute` to generate "
            "cache/rolling_results.parquet, cache/alerts.parquet, and "
            "cache/metadata.json before starting the app."
        )
        st.stop()

    rolling = load_rolling_results()
    alerts = load_alerts()
    metadata = load_metadata()
    node_names = metadata["node_names"]

    st.title("HK Equity Risk Attribution Monitor")
    st.caption(
        f"Data through {metadata['sample_end']} · precomputed offline "
        f"(cache generated {metadata['generated_at'][:19]} UTC) · "
        "this app never computes live — see cache/ and src/precompute.py"
    )

    latest = rolling.iloc[-1]
    lookback_row = rolling.iloc[-31] if len(rolling) > 30 else rolling.iloc[0]

    kpi_counterparts = {
        "us_share": ("china_share", SHARE_LABELS["china_share"]),
        "china_share": ("us_share", SHARE_LABELS["us_share"]),
        "idio_share": (None, None),
    }

    col1, col2, col3 = st.columns(3)
    for col, field in zip([col1, col2, col3], ["us_share", "china_share", "idio_share"]):
        delta_pp = (latest[field] - lookback_row[field]) * 100
        col.metric(
            SHARE_LABELS[field],
            f"{latest[field] * 100:.1f}%",
            f"{delta_pp:+.1f}pp (30 trading days)",
        )
        counterpart_field, counterpart_label = kpi_counterparts[field]
        col.caption(
            build_kpi_observation(
                rolling, field, SHARE_LABELS[field], counterpart_field, counterpart_label
            )
        )

    st.subheader("Transmission Shares Over Time")

    date_options = rolling["date"].dt.date.tolist()
    selected_date = st.select_slider(
        "As of date (drives the network graph below)",
        options=date_options,
        value=date_options[-1],
    )

    st.plotly_chart(
        build_share_chart(rolling, pd.Timestamp(selected_date)), width="stretch"
    )
    st.caption(
        "Dotted orange lines = shock events, dashed purple lines = structural changes "
        "(see src/events.py for sourcing). These are **annotated reference events** — "
        "a hand-curated list for historical context, not RegimeDetector output. "
        "See 'Detected Alerts' below for what the tool actually detected."
    )

    st.subheader(f"Transmission Network — {selected_date}")
    selected_row = rolling[rolling["date"].dt.date == selected_date].iloc[0]
    fevd_matrix = _reconstruct_fevd_matrix(selected_row, node_names)
    network = SpilloverNetwork(fevd_matrix, node_names)

    net_col, table_col = st.columns([2, 1])
    with net_col:
        st.plotly_chart(build_network_figure(network), width="stretch")
        st.caption(
            "An arrow from A to B means A's past values explain part of B's forecast "
            "error variance — e.g. SPX → HSI indicates US equity moves account for some "
            f"of HSI's variance. Edges shown have weight > EDGE_THRESHOLD "
            f"({SpilloverNetwork.EDGE_THRESHOLD:.0%}); arrow width scales with weight; "
            "hover an edge's midpoint for its exact value."
        )
    with table_col:
        edge_rows = []
        for source in node_names:
            for target in node_names:
                if source == target:
                    continue
                weight = network.get_edge_weight(source, target)
                if weight > SpilloverNetwork.EDGE_THRESHOLD:
                    edge_rows.append({"edge": f"{source} -> {target}", "weight": f"{weight:.1%}"})
        edge_df = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)
        st.dataframe(edge_df, width="stretch", hide_index=True, height=390)

    st.subheader("Alerts")
    detected = alerts[alerts["signal_type"].isin(["share_jump", "centrality_flip"])]
    edge_changes = alerts[alerts["signal_type"] == "edge_change"]

    tab_detected, tab_edges = st.tabs(
        [f"Detected Alerts ({len(detected)})", f"Edge Changes — raw ({len(edge_changes)})"]
    )
    with tab_detected:
        st.caption(
            "RegimeDetector's actual detection output: share-jump episodes "
            "(**acute** = sudden shock, re-evaluate hedge now; **chronic** = sustained "
            "drift, flag for periodic review) and centrality rank flips (dominant risk "
            "source switching between SPX and SSEC). Distinct from the reference event "
            "lines above, which are curated, not detected."
        )
        st.dataframe(
            detected[["date", "signal_type", "timescale", "magnitude", "description"]],
            width="stretch",
            hide_index=True,
        )
    with tab_edges:
        st.caption(
            "Raw edge appearance/disappearance events (secondary signal). This is "
            f"{len(edge_changes)} of {len(alerts)} total alerts ({len(edge_changes) / len(alerts):.0%}) "
            "and is not a reliable regime-change indicator on its own — see docs/notes.md."
        )
        st.dataframe(
            edge_changes[["date", "magnitude", "description"]],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Engine Correctness Check")
    st.caption(
        "This tool refits a VAR once on the entire sample (not the rolling windows) and "
        "runs Granger causality tests — a statistical check for whether one market's past "
        "values help predict another's. The F-statistic column below is compared against "
        "the same four relationships reported in the author's published paper (Table 3), "
        "as a check that this engine reproduces a previously validated result rather than "
        "computing something new and unverified."
    )
    validation_df = build_validation_table(metadata["paper_validation"])
    st.dataframe(
        validation_df,
        width="stretch",
        hide_index=True,
        column_config={
            "pair": "Relationship",
            "paper_f": st.column_config.NumberColumn("Paper F-stat", format="%.2f"),
            "tool_f": st.column_config.NumberColumn("This tool's F-stat", format="%.2f"),
            "tool_p": st.column_config.NumberColumn("p-value", format="%.2e"),
            "ratio": st.column_config.NumberColumn("Ratio (tool/paper)", format="%.2f"),
            "verdict": "Observation",
        },
    )
    st.caption(
        "\"Reproduces\" means the tool's F-statistic is within 3x of the paper's in either "
        "direction and statistically significant (p<0.05) — the same order of magnitude, "
        "not an exact match (different data source and sample period than the paper). "
        "USD_CNY→HSI does not reproduce; see docs/notes.md for the full diagnostic trail."
    )
    st.caption(
        f"{metadata['alignment_report']['aligned_days']} aligned trading days "
        f"({metadata['alignment_report']['pct_lost'] * 100:.1f}% lost to calendar "
        f"misalignment) · {metadata['rolling_report']['successful_windows']}/"
        f"{metadata['rolling_report']['total_windows']} rolling windows succeeded"
    )


if __name__ == "__main__":
    main()

"""
eda.py
------
Exploratory Data Analysis (EDA) module.

This module provides functions that return Plotly figures.
Each figure is built from the ACTUAL dataset values.
No fabricated data is used here.

Functions are called from the Streamlit pages to render charts.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Midnight Aurora colour palette ────────────────────────────────────────────
CYAN     = "#22D3EE"
VIOLET   = "#A78BFA"
MINT     = "#6EE7B7"
AMBER    = "#FBBF24"
CORAL    = "#FB7185"
BG_DARK  = "#080D1B"
CARD_BG  = "#14213D"
TEXT     = "#F1F5F9"
GRID     = "#1E2D4A"

PALETTE  = [CYAN, VIOLET, MINT, AMBER, CORAL, "#60A5FA", "#F472B6", "#34D399"]


def _base_layout(title: str, height: int = 420) -> dict:
    """Return a consistent dark-theme Plotly layout dict."""
    return dict(
        title      = dict(text=title, font=dict(color=TEXT, size=16)),
        paper_bgcolor = BG_DARK,
        plot_bgcolor  = CARD_BG,
        font       = dict(color=TEXT, family="Inter, sans-serif"),
        height     = height,
        margin     = dict(l=40, r=20, t=50, b=40),
        xaxis      = dict(gridcolor=GRID, showgrid=True, zeroline=False),
        yaxis      = dict(gridcolor=GRID, showgrid=True, zeroline=False),
        legend     = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
    )


# ── 1. RTT (latency) trend over time ─────────────────────────────────────────
def plot_latency_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of average RTT over time."""
    if "timestamp_local" not in df.columns or "rtt_avg_ms" not in df.columns:
        return _empty_fig("Latency trend: required columns not found.")

    plot_df = df[["timestamp_local", "rtt_avg_ms"]].dropna().copy()
    # Sample to max 2000 points for performance
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000).sort_values("timestamp_local")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_local"],
        y=plot_df["rtt_avg_ms"],
        mode="lines",
        line=dict(color=CYAN, width=1.5),
        name="RTT Avg (ms)",
        fill="tozeroy",
        fillcolor="rgba(34,211,238,0.08)",
    ))
    fig.update_layout(**_base_layout("Average RTT / Latency Over Time"))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="RTT (ms)")
    return fig


# ── 2. Latency distribution ───────────────────────────────────────────────────
def plot_latency_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of rtt_avg_ms values."""
    if "rtt_avg_ms" not in df.columns:
        return _empty_fig("rtt_avg_ms column not found.")

    vals = df["rtt_avg_ms"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=60,
        marker_color=CYAN,
        opacity=0.85,
        name="RTT Avg (ms)",
    ))
    fig.update_layout(**_base_layout("Latency (RTT) Distribution"))
    fig.update_xaxes(title_text="RTT Avg (ms)")
    fig.update_yaxes(title_text="Count")
    return fig


# ── 3. Packet loss distribution ───────────────────────────────────────────────
def plot_packet_loss_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart bucketing packet loss % into ranges."""
    if "packet_loss_pct" not in df.columns:
        return _empty_fig("packet_loss_pct column not found.")

    vals = df["packet_loss_pct"].dropna()
    bins   = [0, 1, 5, 10, 25, 50, 100]
    labels = ["0%", "1-5%", "5-10%", "10-25%", "25-50%", ">50%"]
    counts = pd.cut(vals, bins=bins, labels=labels, right=True).value_counts().sort_index()

    fig = go.Figure(go.Bar(
        x=counts.index.astype(str),
        y=counts.values,
        marker_color=[MINT, CYAN, AMBER, CORAL, "#EF4444", "#991B1B"],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(**_base_layout("Packet Loss Distribution"))
    fig.update_xaxes(title_text="Packet Loss Range")
    fig.update_yaxes(title_text="Count")
    return fig


# ── 4. Download vs Upload speed ───────────────────────────────────────────────
def plot_speed_comparison(df: pd.DataFrame) -> go.Figure:
    """Box plots comparing download and upload Mbps."""
    cols = [c for c in ["download_mbps", "upload_mbps"] if c in df.columns]
    if not cols:
        return _empty_fig("download_mbps / upload_mbps columns not found.")

    fig = go.Figure()
    colors = [CYAN, VIOLET]
    for col, color in zip(cols, colors):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col.replace("_", " ").title(),
            marker_color=color,
            line_color=color,
            boxmean=True,
        ))
    fig.update_layout(**_base_layout("Download vs Upload Speed (Mbps)"))
    fig.update_yaxes(title_text="Speed (Mbps)")
    return fig


# ── 5. Event category distribution ───────────────────────────────────────────
def plot_event_category(df: pd.DataFrame) -> go.Figure:
    """Pie / donut chart of event_category."""
    if "event_category" not in df.columns:
        return _empty_fig("event_category column not found.")

    counts = df["event_category"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=PALETTE, line=dict(color=BG_DARK, width=2)),
        textinfo="label+percent",
        insidetextorientation="radial",
    ))
    fig.update_layout(**_base_layout("Network Event Category Distribution (Derived from Binary Labels)"))
    return fig


# ── 6. Jitter trend ───────────────────────────────────────────────────────────
def plot_jitter_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of jitter_ms over time."""
    if "timestamp_local" not in df.columns or "jitter_ms" not in df.columns:
        return _empty_fig("Jitter trend: required columns not found.")

    plot_df = df[["timestamp_local", "jitter_ms"]].dropna().copy()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000).sort_values("timestamp_local")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_local"],
        y=plot_df["jitter_ms"],
        mode="lines",
        line=dict(color=VIOLET, width=1.5),
        name="Jitter (ms)",
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)",
    ))
    fig.update_layout(**_base_layout("Jitter Over Time"))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Jitter (ms)")
    return fig


# ── 7. Hourly average heatmap ─────────────────────────────────────────────────
def plot_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Average RTT by hour-of-day — shows peak usage patterns."""
    if "hour_local" not in df.columns or "rtt_avg_ms" not in df.columns:
        return _empty_fig("hour_local or rtt_avg_ms not found.")

    hourly = (
        df.groupby("hour_local")["rtt_avg_ms"]
        .mean()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hourly.columns = ["Hour", "Avg RTT (ms)"]

    fig = go.Figure(go.Bar(
        x=hourly["Hour"],
        y=hourly["Avg RTT (ms)"],
        marker=dict(
            color=hourly["Avg RTT (ms)"],
            colorscale=[[0, MINT], [0.5, AMBER], [1, CORAL]],
            showscale=True,
            colorbar=dict(title="ms"),
        ),
    ))
    fig.update_layout(**_base_layout("Average RTT by Hour of Day"))
    fig.update_xaxes(title_text="Hour (0 = midnight)")
    fig.update_yaxes(title_text="Avg RTT (ms)")
    return fig


# ── 8. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap for key numeric features."""
    cols = [
        c for c in [
            "rtt_avg_ms", "jitter_ms", "packet_loss_pct",
            "download_mbps", "upload_mbps", "dns_resolve_ms",
            "http_ms", "gateway_rtt_avg_ms", "wifi_signal_pct",
        ]
        if c in df.columns
    ]
    if len(cols) < 2:
        return _empty_fig("Not enough numeric columns for correlation.")

    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=[c.replace("_", "<br>") for c in corr.columns],
        y=[c.replace("_", "<br>") for c in corr.index],
        colorscale=[[0, CORAL], [0.5, CARD_BG], [1, CYAN]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(**_base_layout("Feature Correlation Heatmap", height=500))
    return fig


# ── 9. Feature importance bar chart ──────────────────────────────────────────
def plot_feature_importance(feature_names: list, importances: list) -> go.Figure:
    """Horizontal bar chart for Random Forest feature importances."""
    paired = sorted(zip(importances, feature_names), reverse=True)
    importances_sorted = [p[0] for p in paired]
    names_sorted       = [p[1].replace("_", " ") for p in paired]

    fig = go.Figure(go.Bar(
        x=importances_sorted,
        y=names_sorted,
        orientation="h",
        marker=dict(
            color=importances_sorted,
            colorscale=[[0, VIOLET], [1, CYAN]],
        ),
        text=[f"{v:.3f}" for v in importances_sorted],
        textposition="outside",
    ))
    fig.update_layout(**_base_layout("Feature Importance (Random Forest)", height=420))
    fig.update_xaxes(title_text="Importance Score")
    return fig


# ── 10. Confusion matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, labels: list) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, CARD_BG], [1, CYAN]],
        text=cm,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        **_base_layout("Confusion Matrix"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


# ── 11. Before / After comparison ────────────────────────────────────────────
def plot_before_after(metrics: list, before: list, after: list) -> go.Figure:
    """Grouped bar chart comparing original vs simulated scenario."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original Measurement",
        x=metrics, y=before,
        marker_color=CORAL,
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Simulated (Optimised) Scenario",
        x=metrics, y=after,
        marker_color=MINT,
        opacity=0.85,
    ))
    fig.update_layout(
        **_base_layout("Performance Comparison: Original vs Simulated", height=440),
        barmode="group",
    )
    fig.update_yaxes(title_text="Value")
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    """Return a blank figure with an annotation message."""
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=AMBER, size=13))
    fig.update_layout(**_base_layout("No Data"))
    return fig

"""Reusable visualization helpers for the anemia dashboard."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import io as pio

COLOR_PALETTE = {
    "primary": "#1C7CBA",
    "secondary": "#4AA96C",
    "accent": "#F1C40F",
    "danger": "#E74C3C",
    "neutral_light": "#F4F7F9",
    "neutral_dark": "#2C3E50",
}

TEMPLATE_NAME = "anemia_dashboard_template"

if TEMPLATE_NAME not in pio.templates:
    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout={
            "font": {"family": "Inter, sans-serif", "color": COLOR_PALETTE["neutral_dark"]},
            "paper_bgcolor": "white",
            "plot_bgcolor": COLOR_PALETTE["neutral_light"],
            "colorway": [
                COLOR_PALETTE["primary"],
                COLOR_PALETTE["secondary"],
                COLOR_PALETTE["accent"],
                COLOR_PALETTE["danger"],
            ],
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
        }
    )

px.defaults.template = TEMPLATE_NAME
px.defaults.color_discrete_sequence = list(pio.templates[TEMPLATE_NAME].layout.colorway)


def create_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    icon: Optional[str] = None,
    help_text: Optional[str] = None,
) -> str:
    """Compose an HTML block representing a metric card.

    Args:
        label: Metric title.
        value: Primary metric value as formatted string.
        delta: Optional delta indicator.
        icon: Optional emoji or icon HTML.
        help_text: Supplementary caption.

    Returns:
        HTML string suitable for rendering through Streamlit markdown.
    """
    icon_html = f"<span class='metric-icon'>{icon}</span>" if icon else ""
    delta_html = f"<span class='metric-delta'>{delta}</span>" if delta else ""
    help_html = f"<p class='metric-help'>{help_text}</p>" if help_text else ""
    return (
        "<div class='metric-card'>"
        f"<div class='metric-header'>{icon_html}<span class='metric-label'>{label}</span></div>"
        f"<div class='metric-value'>{value}{delta_html}</div>"
        f"{help_html}"
        "</div>"
    )


def create_age_distribution_chart(
    dataframe: pd.DataFrame,
    age_group_column: str = "Kelompok_Usia",
    count_column: Optional[str] = None,
) -> go.Figure:
    """Generate a bar chart representing respondent counts per age group.

    Args:
        dataframe: Source dataframe.
        age_group_column: Column containing categorical age group labels.
        count_column: Optional column to aggregate; defaults to row counts.

    Returns:
        Plotly Figure object.
    """
    if count_column:
        grouped = dataframe.groupby(age_group_column)[count_column].count().reset_index(name="Jumlah")
    else:
        grouped = dataframe[age_group_column].value_counts().rename_axis(age_group_column).reset_index(name="Jumlah")

    fig = px.bar(
        grouped,
        x=age_group_column,
        y="Jumlah",
        labels={age_group_column: "Kelompok Usia", "Jumlah": "Jumlah Responden"},
    )
    fig.update_traces(marker_radius=6, marker_line_color="white", marker_line_width=1)
    fig.update_layout(title="Distribusi Responden per Kelompok Usia")
    return fig


def create_anemia_pie_chart(
    dataframe: pd.DataFrame,
    status_column: str = "Status_Anemia",
) -> go.Figure:
    """Create anemia prevalence pie chart.

    Args:
        dataframe: Source dataframe.
        status_column: Column containing anemia status labels.

    Returns:
        Plotly Figure object.
    """
    counts = dataframe[status_column].value_counts().reset_index()
    counts.columns = [status_column, "Jumlah"]
    fig = px.pie(
        counts,
        names=status_column,
        values="Jumlah",
        hole=0.4,
        color=status_column,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(title="Prevalensi Anemia")
    return fig


def create_comparison_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    barmode: str = "group",
) -> go.Figure:
    """Render a comparative bar chart for subgroup analysis.

    Args:
        dataframe: Source dataframe.
        x: Column to display on the x-axis.
        y: Numerical column to aggregate (mean).
        color: Optional column for color grouping.
        barmode: Plotly bar mode.

    Returns:
        Plotly Figure object.
    """
    agg_df = dataframe.groupby([x] + ([color] if color else []))[y].mean().reset_index()
    fig = px.bar(
        agg_df,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        labels={x: x.replace("_", " "), y: y.replace("_", " ")},
    )
    fig.update_layout(title="Perbandingan Kelompok")
    return fig


def create_heatmap(
    dataframe: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    title: str = "Korelasi Antar Fitur",
) -> go.Figure:
    """Generate correlation heatmap.

    Args:
        dataframe: Source dataframe.
        columns: Optional subset of columns for correlation.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    selected = dataframe[columns] if columns else dataframe.select_dtypes(include=np.number)
    corr_matrix = selected.corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            hoverongaps=False,
        )
    )
    fig.update_layout(title=title)
    return fig


def apply_styling() -> str:
    """Return CSS styling for consistent visualization components."""
    return (
        "<style>"
        ".metric-card{background:#FFFFFF;border-radius:16px;padding:18px;box-shadow:0 6px 18px rgba(28,124,186,0.08);margin-bottom:16px;}"
        f".metric-header{{display:flex;align-items:center;font-size:0.9rem;color:{COLOR_PALETTE['secondary']};font-weight:600;margin-bottom:8px;}}"
        ".metric-icon{margin-right:8px;}"
        f".metric-value{{font-size:2rem;font-weight:700;color:{COLOR_PALETTE['neutral_dark']};display:flex;align-items:baseline;gap:12px;}}"
        f".metric-delta{{font-size:0.9rem;color:{COLOR_PALETTE['primary']};font-weight:600;}}"
        ".metric-help{margin:0;font-size:0.85rem;color:#6C7A89;}"
        "</style>"
    )

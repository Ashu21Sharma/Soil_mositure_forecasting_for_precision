"""Reusable branded dashboard components and Plotly indicators."""

import base64
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"
COLORS = {
    "forest": "#17553d",
    "green": "#237a57",
    "soft_green": "#eaf4ed",
    "muted": "#66756b",
    "orange": "#db8b32",
    "red": "#b54747",
    "gold": "#ad831d",
    "blue": "#4b8db8",
}


def logo_data_uri():
    """Return the local SVG logo as a self-contained HTML image URI."""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def apply_dashboard_style():
    """Apply consistent styling to all six pages and the persistent sidebar."""
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #ffffff; color: #23342a; }
        [data-testid="stSidebar"] {
            background: #f3f8f4;
            border-right: 1px solid #e1ebe3;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #17553d; }
        h1, h2, h3 { color: #17553d; }
        [data-testid="stMainMenu"] { display: none; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2ebe4;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(31, 72, 48, 0.045);
        }
        [data-testid="stMetricLabel"] { color: #66756b; }
        .brand-banner {
            display: flex;
            align-items: center;
            gap: 18px;
            background: linear-gradient(115deg, #123f2b 0%, #216643 58%, #2f7951 100%);
            border-radius: 14px;
            padding: 24px 30px;
            margin: 4px 0 22px;
            box-shadow: 0 8px 22px rgba(21, 69, 44, 0.13);
        }
        .brand-banner img { width: 68px; height: 68px; flex: 0 0 68px; }
        .brand-banner h1 {
            color: #ffffff !important;
            font-size: clamp(1.55rem, 3vw, 2.25rem) !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            margin: 0 0 7px 0 !important;
        }
        .brand-banner p { color: #e2efe5 !important; margin: 0; }
        .brand-kicker {
            color: #b7dfc2;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .section-note { color: #66756b; margin-top: -8px; }
        .overview-section-title {
            color: #174d32;
            font-size: 1.32rem;
            font-weight: 650;
            margin: 4px 0 13px;
        }
        .overview-copy { color: #46564b; line-height: 1.65; }
        .overview-attribute {
            display: inline-block;
            background: #edf5ef;
            border: 1px solid #dceade;
            border-radius: 20px;
            color: #245c3b;
            font-size: 0.87rem;
            margin: 3px 4px 3px 0;
            padding: 6px 11px;
        }
        .overview-telemetry {
            background: #f5f9f6;
            border: 1px solid #e1ebe3;
            border-radius: 10px;
            color: #33493a;
            line-height: 1.7;
            margin-bottom: 12px;
            padding: 12px 15px;
        }
        @media (max-width: 700px) {
            .brand-banner { align-items: flex-start; padding: 18px; gap: 12px; }
            .brand-banner img { width: 48px; height: 48px; flex-basis: 48px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_sidebar():
    """Show the local logo and product name at the top of navigation."""
    if not LOGO_PATH.is_file():
        st.sidebar.error(f"Dashboard logo not found: {LOGO_PATH}")
        return
    st.sidebar.image(str(LOGO_PATH), width=72)
    st.sidebar.markdown("## Precision Agriculture")
    st.sidebar.caption("Soil moisture forecasting")


def render_page_banner(title, subtitle, kicker="PRECISION AGRICULTURE"):
    """Render a consistent title banner with the original local crop logo."""
    image = logo_data_uri()
    st.markdown(
        f'<div class="brand-banner"><img src="{image}" alt="Precision Agriculture logo">'
        '<div><div class="brand-kicker">'
        f"{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def style_plot(figure, title=None, height=390):
    """Apply shared chart fonts, spacing, and white/green theme."""
    figure.update_layout(
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#23342a", "family": "sans-serif"},
        margin={"l": 35, "r": 25, "t": 65 if title else 30, "b": 40},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#edf1ee", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#edf1ee", zeroline=False)
    return figure


def sensor_gauge(title, value, minimum, maximum, suffix, bar_color, steps):
    """Build a compact number-plus-gauge Plotly figure."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 27, "color": "#23342a"}},
            title={"text": title, "font": {"size": 15, "color": COLORS["forest"]}},
            gauge={
                "axis": {
                    "range": [minimum, maximum],
                    "tickcolor": COLORS["muted"],
                    "tickfont": {"color": COLORS["muted"]},
                },
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#ffffff",
                "bordercolor": "#e2ebe4",
                "borderwidth": 1,
                "steps": steps,
            },
        )
    )
    figure.update_layout(
        height=245,
        margin={"l": 20, "r": 20, "t": 53, "b": 10},
        paper_bgcolor="white",
    )
    return figure




def threshold_gauge(value, low_threshold, upper_threshold):
    """Show a predicted moisture value against configurable advisory zones."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 30, "color": "#23342a"}},
            title={
                "text": "Forecast soil moisture",
                "font": {"size": 16, "color": COLORS["forest"]},
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["green"], "thickness": 0.25},
                "steps": [
                    {"range": [0, low_threshold], "color": "#f8e9e9"},
                    {"range": [low_threshold, upper_threshold], "color": "#eaf4ed"},
                    {"range": [upper_threshold, 100], "color": "#fff3d8"},
                ],
                "threshold": {
                    "line": {"color": COLORS["forest"], "width": 4},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    figure.update_layout(
        height=275,
        margin={"l": 22, "r": 22, "t": 58, "b": 15},
        paper_bgcolor="white",
    )
    return figure

# src/charts/utils.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------- helpers (axes, formatting) ----------
def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        try:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    return df

def _kwh_yaxis(fig, showgrid=True):
    # SI format (~s) with unit suffix → 1.2M kWh, 3.4G kWh, etc.
    fig.update_yaxes(
        tickformat="~s",
        ticksuffix=" kWh",
        showgrid=showgrid,
        zeroline=False,
    )

def _date_xaxis(fig):
    # Rotate labels, allow margins so nothing clips; readable month/day by default
    fig.update_xaxes(
        tickangle=-30,
        automargin=True,
        showgrid=False,
        tickformat="%b %d\n%Y"  # multi-line helps prevent clipping
    )

def _layout(fig, x_title: str, y_title: str, height=360):
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        legend=dict(orientation="v", bgcolor="rgba(255,255,255,0.7)"),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


# ---------- charts used by the app ----------
def line_consumption(df: pd.DataFrame):
    """
    Expects: reading_date, total_kwh, region
    For <= 3 regions we use a clean line chart.
    """
    if df is None or df.empty:
        return px.line(pd.DataFrame({"reading_date": [], "total_kwh": []}),
                       x="reading_date", y="total_kwh", title="Consumption")

    df = _ensure_dt(df, "reading_date")
    fig = px.line(
        df,
        x="reading_date",
        y="total_kwh",
        color="region",
        markers=True,
        hover_data={"total_kwh": ":,.0f", "reading_date": "|%Y-%m-%d"},
    )
    _kwh_yaxis(fig)
    _date_xaxis(fig)
    _layout(fig, "Date", "Total (kWh)")
    # Crisper hover
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} kWh<extra>%{fullData.name}</extra>")
    return fig


def area_stacked(df: pd.DataFrame):
    """
    Expects: reading_date, total_kwh, region
    For > 3 regions we stack areas to show contribution by region.
    """
    if df is None or df.empty:
        return px.area(pd.DataFrame({"reading_date": [], "total_kwh": []}),
                       x="reading_date", y="total_kwh", title="Consumption")

    df = _ensure_dt(df, "reading_date")
    fig = px.area(
        df,
        x="reading_date",
        y="total_kwh",
        color="region",
        groupnorm=None,
        hover_data={"total_kwh": ":,.0f", "reading_date": "|%Y-%m-%d"},
    )
    _kwh_yaxis(fig)
    _date_xaxis(fig)
    _layout(fig, "Date", "Total (kWh)", height=380)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} kWh<extra>%{fullData.name}</extra>")
    return fig


def bar_top_regions(df: pd.DataFrame):
    """
    Expects: region, total_kwh
    Shows totals by region (friendly names already mapped in dbx).
    """
    if df is None or df.empty:
        return px.bar(pd.DataFrame({"region": [], "total_kwh": []}),
                      x="region", y="total_kwh", title="Top regions")

    # sort descending for a nicer bar chart
    df = df.sort_values("total_kwh", ascending=False)
    fig = px.bar(
        df,
        x="region",
        y="total_kwh",
        text="total_kwh",
        hover_data={"total_kwh": ":,.0f", "region": True},
    )
    # show value labels in compact form
    fig.update_traces(texttemplate="%{y:~s} kWh", textposition="outside", cliponaxis=False)
    _kwh_yaxis(fig)
    fig.update_xaxes(tickangle=-30, automargin=True, showgrid=False)
    _layout(fig, "Region", "Total (kWh)", height=360)
    return fig


def donut_voltage(df: pd.DataFrame):
    """
    Expects: voltage_band, kwh
    """
    if df is None or df.empty:
        return px.pie(values=[1], names=["No data"], hole=0.55)

    fig = px.pie(df, names="voltage_band", values="kwh", hole=0.55)
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent:.1%}",
        hovertemplate="%{label}: %{value:,.0f} kWh<extra></extra>",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=0), legend_title_text="Voltage/Connection")
    return fig


def line_forecast(hist_df: pd.DataFrame, fcst_df: pd.DataFrame):
    """
    hist_df: columns ['date','kwh']
    fcst_df: columns ['date','kwh']
    """
    hist_df = hist_df.copy() if hist_df is not None else pd.DataFrame(columns=["date", "kwh"])
    fcst_df = fcst_df.copy() if fcst_df is not None else pd.DataFrame(columns=["date", "kwh"])
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
    if not fcst_df.empty:
        fcst_df["date"] = pd.to_datetime(fcst_df["date"])

    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["kwh"],
            mode="lines+markers", name="History",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} kWh<extra></extra>",
        ))
    if not fcst_df.empty:
        fig.add_trace(go.Scatter(
            x=fcst_df["date"], y=fcst_df["kwh"],
            mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} kWh<extra></extra>",
        ))

    _kwh_yaxis(fig)
    _date_xaxis(fig)
    _layout(fig, "Date", "kWh (history & forecast)", height=340)
    return fig

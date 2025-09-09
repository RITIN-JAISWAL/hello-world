import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def line_consumption(df: pd.DataFrame):
    fig = px.line(df, x="reading_date", y="total_kwh", color="region", markers=True)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def area_stacked(df: pd.DataFrame):
    # stacked area by region over time
    fig = px.area(df, x="reading_date", y="total_kwh", color="region", groupnorm=None)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def bar_top_regions(df_totals: pd.DataFrame):
    fig = px.bar(df_totals, x="region", y="total_kwh")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def pie_import_export(df: pd.DataFrame):
    fig = px.pie(df, names="import_export_flag", values="kwh_consumed", hole=0.55)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=0))
    return fig

def heatmap_period(df: pd.DataFrame):
    # expects reading_date, period, kwh
    if df.empty:
        return go.Figure()
    pivot = df.pivot_table(index="period", columns="reading_date", values="kwh", aggfunc="sum").fillna(0)
    fig = px.imshow(pivot, aspect="auto", origin="lower")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig

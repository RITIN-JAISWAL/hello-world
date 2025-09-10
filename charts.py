def donut_voltage(df: pd.DataFrame):
    """Expect columns: voltage_band, kwh"""
    if df is None or df.empty:
        return px.pie(values=[1], names=["No data"], hole=0.55)
    fig = px.pie(df, names="voltage_band", values="kwh", hole=0.55)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=0))
    return fig



voltage_df = dbx.voltage_breakdown(dstr(start), dstr(end), region)
if not voltage_df.empty:
    st.subheader("Voltage mix")
    st.plotly_chart(donut_voltage(voltage_df), use_container_width=True, theme=None)
else:
    st.info("No voltage/connection-type data.")




import plotly.express as px
import pandas as pd

# ... your existing helpers (line_consumption, area_stacked, bar_top_regions, donut_voltage, etc.)

def line_forecast(hist_df: pd.DataFrame, fcst_df: pd.DataFrame):
    """
    hist_df: columns ['date','kwh']
    fcst_df: columns ['date','kwh']
    """
    if hist_df is None or hist_df.empty:
        return px.line(pd.DataFrame({"date": [], "kwh": []}), x="date", y="kwh", title="Forecast")
    hist = hist_df.assign(series="History")
    fcst = fcst_df.assign(series="Forecast")
    combined = pd.concat([hist, fcst], ignore_index=True)
    fig = px.line(combined, x="date", y="kwh", color="series")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=0), height=320)
    return fig

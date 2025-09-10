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
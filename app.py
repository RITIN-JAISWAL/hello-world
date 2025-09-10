# app.py
import os
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# our code
from src.agents.energy_agent import build_agent
from src.data_access import dbx
from src.charts.utils import (
    line_consumption,
    area_stacked,
    bar_top_regions,
    donut_voltage,      # NEW: voltage donut
    heatmap_period,
)

# -------------------- Page & Styles --------------------
st.set_page_config(page_title="Metering Agentic POC", layout="wide")

st.markdown("""
<style>
/* Blue header block */
.header-block {
  background:#0b3d91;
  color:#ffffff;
  padding:18px 22px;
  border-radius:16px;
  margin-bottom:14px;
}
.header-block h1 {
  color:#ffffff !important;
  margin:0; font-weight:800; line-height:1.15; font-size:28px;
}

/* Cards */
.card {
  background:#ffffff;
  border:1px solid #e9ecef;
  border-radius:16px;
  padding:16px;
  box-shadow:0 6px 18px rgba(0,0,0,0.06);
  margin-bottom:16px;
}

/* Highlights */
.kpi-title { font-size:14px; color:#6c757d; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
.kpi-number { font-size:34px; font-weight:800; margin-bottom:2px; }
.kpi-sub { color:#6c757d; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="header-block">
  <h1>Metering Agentic AI – Settlement POC</h1>
  <div>Gold: <code>sdr_gold_dev.meter_data</code> • Silver: <code>sdr_silver_dev.silver_meter_data.tbl_meterdata</code></div>
</div>
""", unsafe_allow_html=True)

# -------------------- Filters --------------------
region_options = ["All Regions"] + dbx.distinct_regions()
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with c2:
        end = st.date_input("End date", value=date.today())
    with c3:
        region_choice = st.selectbox("Region / GSP", options=region_options, index=0)
        region = None if region_choice == "All Regions" else region_choice
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Data pulls (always reflect filters) --------------------
def dstr(d: date) -> str:
    return d.strftime("%Y-%m-%d")

start_s, end_s = dstr(start), dstr(end)

daily_df   = dbx.agg_daily(start_s, end_s, region)
totals_df  = dbx.totals_by_region(start_s, end_s, top_n=12, region=region)
heat_df    = dbx.heatmap_day_period(start_s, end_s, region)
voltage_df = dbx.voltage_breakdown(start_s, end_s, region)

# KPIs
total_kwh       = float(daily_df["total_kwh"].sum()) if not daily_df.empty else 0.0
days_in_window  = daily_df["reading_date"].nunique() if not daily_df.empty else 0
regions_in_view = daily_df["region"].nunique() if not daily_df.empty else 0
mpans           = dbx.distinct_mpan_count(start_s, end_s, region)

# -------------------- Highlights --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Highlights")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi-title">Total Consumption</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{total_kwh:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-sub">kWh</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-title">Days</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{days_in_window}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{start_s} → {end_s}</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-title">Regions in view</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{regions_in_view}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{region_choice}</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-title"># MPANs</div>', unsafe_allow_html=True)
        if mpans is not None:
            st.markdown(f'<div class="kpi-number">{mpans:,}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-number">N/A</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-sub">distinct</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Charts (all in one block) --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Charts")

    # Row 1: time-series + voltage donut
    r1c1, r1c2 = st.columns([2,1])
    with r1c1:
        if not daily_df.empty:
            if daily_df["region"].nunique() > 3:
                st.plotly_chart(area_stacked(daily_df), use_container_width=True, theme=None)
            else:
                st.plotly_chart(line_consumption(daily_df), use_container_width=True, theme=None)
        else:
            st.info("No data for selected filters.")
    with r1c2:
        if not voltage_df.empty:
            st.subheader("Voltage mix")
            st.plotly_chart(donut_voltage(voltage_df), use_container_width=True, theme=None)
        else:
            st.info("No voltage/connection-type data.")

    # Row 2: Top regions + heatmap (both respond to Region filter)
    r2c1, r2c2 = st.columns([1,1])
    with r2c1:
        if not totals_df.empty:
            st.subheader("Totals (kWh)" if region else "Top regions (total kWh)")
            st.plotly_chart(bar_top_regions(totals_df), use_container_width=True, theme=None)
        else:
            st.info("No region totals.")
    with r2c2:
        if not heat_df.empty:
            st.subheader("Period heatmap")
            st.plotly_chart(heatmap_period(heat_df), use_container_width=True, theme=None)
        else:
            st.info("No period grain available for heatmap.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Chat (agentic) --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Chat with your data")
agent = build_agent()

if "lc_messages" not in st.session_state:
    st.session_state["lc_messages"] = []

# show history
for m in st.session_state["lc_messages"]:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    st.chat_message(role).markdown(m.content)

# input
prompt = st.chat_input("Ask e.g. 'Predict next month for London' or 'Peak vs off-peak last 14 days'")
if prompt:
    st.session_state["lc_messages"].append(HumanMessage(content=prompt))
    st.chat_message("user").markdown(prompt)
    try:
        result = agent.invoke({"messages": st.session_state["lc_messages"]})
        ai_msg = result["messages"][-1]
        content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
    except Exception as e:
        content = f"Error: {e}"
    st.session_state["lc_messages"].append(AIMessage(content=content))
    st.chat_message("assistant").markdown(content)
st.markdown('</div>', unsafe_allow_html=True)

st.caption("Auto schema discovery • Region names via Gold view • Filters drive every chart • Voltage donut replaces Import/Export pie")

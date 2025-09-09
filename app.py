import os
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.energy_agent import build_agent
from src.data_access import dbx
from src.charts.utils import (
    line_consumption, area_stacked, bar_top_regions,
    pie_import_export, heatmap_period
)

st.set_page_config(page_title="Metering Agentic POC", layout="wide")

# ---------- Styles ----------
st.markdown("""
<style>
.header-block { background:#0b3d91; color:#fff; padding:18px 22px; border-radius:16px; margin-bottom:14px; }
.header-block h1 { color:#fff !important; margin:0; font-weight:800; font-size:28px; line-height:1.15; }
.card { background:#fff; border:1px solid #e9ecef; border-radius:16px; padding:16px; box-shadow:0 6px 18px rgba(0,0,0,.06); margin-bottom:16px; }
.kpi-title { font-size:14px; color:#6c757d; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
.kpi-number { font-size:34px; font-weight:800; margin-bottom:2px; }
.kpi-sub { color:#6c757d; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="header-block">
  <h1>Metering Agentic AI – Settlement POC</h1>
  <div>Gold: <code>sdr_gold_dev.meter_data</code> • Silver: <code>sdr_silver_dev.silver_meter_data.tbl_meterdata</code></div>
</div>
""", unsafe_allow_html=True)

# ---------- Filters ----------
region_list = ["All Regions"] + dbx.distinct_regions()
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with c2:
        end = st.date_input("End date", value=date.today())
    with c3:
        region_choice = st.selectbox("Region / GSP", options=region_list, index=0)
        region = None if region_choice == "All Regions" else region_choice
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Data pulls (no caching; always reflect UI) ----------
def dstr(d: date) -> str: return d.strftime("%Y-%m-%d")

daily_df = dbx.agg_daily(dstr(start), dstr(end), region)
totals_df = dbx.totals_by_region(dstr(start), dstr(end), top_n=12, region=region)
heat_df = dbx.heatmap_day_period(dstr(start), dstr(end), region)
pie_df = dbx.import_export_breakdown(dstr(start), dstr(end), region)

total_kwh = float(daily_df["total_kwh"].sum()) if not daily_df.empty else 0.0
mpans = dbx.distinct_mpan_count(dstr(start), dstr(end), region)
days = daily_df["reading_date"].nunique() if not daily_df.empty else 0
regions_in_view = daily_df["region"].nunique() if not daily_df.empty else 0

# ---------- Highlights ----------
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
        st.markdown(f'<div class="kpi-number">{days}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{start} → {end}</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-title">Regions in view</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{regions_in_view}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{region_choice}</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-title"># MPANs</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{mpans:,}</div>' if mpans is not None else '<div class="kpi-number">N/A</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-sub">distinct</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Charts (single block) ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Charts")

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
        if not pie_df.empty:
            st.plotly_chart(pie_import_export(pie_df), use_container_width=True, theme=None)
        else:
            st.info("No import/export rows.")

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

# ---------- Chat ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Chat with your data")
agent = build_agent()

if "lc_messages" not in st.session_state:
    st.session_state["lc_messages"] = []

for m in st.session_state["lc_messages"]:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    st.chat_message(role).markdown(m.content)

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

st.caption("Auto schema discovery • Uses ALL gold tables via information_schema • Region names resolved from dim_gspgroup • Filters drive every chart")

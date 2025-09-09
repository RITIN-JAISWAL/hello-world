import os
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_core.messages import HumanMessage, AIMessage

from src.agents.energy_agent import build_agent
from src.data_access import dbx
from src.charts.utils import (
    line_consumption, area_stacked, bar_top_regions,
    pie_import_export, heatmap_period
)

# ---------- Page ----------
st.set_page_config(page_title="Metering Agentic POC", layout="wide")
st.title("⚡ Metering Agentic AI – Settlement POC")
st.caption("Gold: sdr_gold_dev.meter_data • Silver: sdr_silver_dev.silver_meter_data.tbl_meterdata")

# ---------- Filters (with Region dropdown) ----------
region_list = ["All Regions"] + dbx.distinct_regions()
c1, c2, c3, c4 = st.columns([1,1,2,1])
with c1:
    start = st.date_input("Start date", value=date.today() - timedelta(days=30))
with c2:
    end = st.date_input("End date", value=date.today())
with c3:
    region_choice = st.selectbox("Region / GSP", options=region_list, index=0)
    region = None if region_choice == "All Regions" else region_choice
with c4:
    if st.button("Refresh data"):
        for k in ("daily_df", "pp_df", "totals_df", "heat_df"):
            st.session_state.pop(k, None)

# ---------- Data pulls ----------
def load_daily():
    return dbx.agg_daily(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region)

daily_df = st.session_state.get("daily_df")
if daily_df is None:
    try:
        daily_df = load_daily()
        st.session_state["daily_df"] = daily_df
    except Exception as e:
        st.error(f"Failed to pull daily consumption. {e}")
        daily_df = pd.DataFrame(columns=["reading_date","region","total_kwh"])

totals_df = st.session_state.get("totals_df")
if totals_df is None:
    try:
        totals_df = dbx.totals_by_region(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), top_n=12)
        st.session_state["totals_df"] = totals_df
    except Exception:
        totals_df = pd.DataFrame()

heat_df = st.session_state.get("heat_df")
if heat_df is None:
    try:
        heat_df = dbx.heatmap_day_period(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region)
        st.session_state["heat_df"] = heat_df
    except Exception:
        heat_df = pd.DataFrame()

# KPIs
total_kwh = float(daily_df["total_kwh"].sum()) if not daily_df.empty else 0.0
mpans = dbx.distinct_mpan_count(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total kWh", f"{total_kwh:,.0f}")
k2.metric("Days", f"{daily_df['reading_date'].nunique() if not daily_df.empty else 0}")
k3.metric("Regions in view", f"{daily_df['region'].nunique() if not daily_df.empty else 0}")
k4.metric("#MPANs", f"{mpans:,}" if mpans is not None else "N/A")

# ---------- Charts row 1 ----------
c1, c2 = st.columns([2,1])
with c1:
    if not daily_df.empty:
        # show stacked area if many regions; else line
        if daily_df["region"].nunique() > 3:
            st.plotly_chart(area_stacked(daily_df), use_container_width=True, theme=None)
        else:
            st.plotly_chart(line_consumption(daily_df), use_container_width=True, theme=None)
    else:
        st.info("No data for selected filters.")

with c2:
    try:
        pie_df = dbx.import_export_breakdown(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region
        )
        if not pie_df.empty:
            st.plotly_chart(pie_import_export(pie_df), use_container_width=True, theme=None)
        else:
            st.info("No import/export rows.")
    except Exception:
        st.warning("Import/Export pie unavailable.")

# ---------- Charts row 2 ----------
c3, c4 = st.columns([1,1])
with c3:
    if not totals_df.empty:
        st.subheader("Top regions (total kWh)")
        st.plotly_chart(bar_top_regions(totals_df), use_container_width=True, theme=None)
    else:
        st.info("No region totals.")

with c4:
    if not heat_df.empty:
        st.subheader("Period heatmap")
        st.plotly_chart(heatmap_period(heat_df), use_container_width=True, theme=None)
    else:
        st.info("No period grain available for heatmap.")

st.divider()

# ---------- Chat (fixed to use LangChain Messages) ----------
st.subheader("Chat with your data")
agent = build_agent()

# Quick suggestions
sg = st.container()
b1, b2, b3 = sg.columns(3)
if b1.button("What regions are there?"):
    st.session_state.setdefault("queued_prompt", "List distinct regions available.")
if b2.button("Predict the target kWh for London for the month of July"):
    st.session_state.setdefault("queued_prompt", "Predict the target kWh for London for July this year.")
if b3.button("Peak vs off-peak last 14 days"):
    st.session_state.setdefault("queued_prompt", "Show peak vs off-peak for last 14 days for all regions.")

# Message history as LangChain objects
if "lc_messages" not in st.session_state:
    st.session_state["lc_messages"] = []

for m in st.session_state["lc_messages"]:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    st.chat_message(role).markdown(m.content)

prompt = st.chat_input("Ask e.g. 'Predict next month for London' or 'Peak vs off-peak last 14 days'")
if st.session_state.get("queued_prompt"):
    prompt = st.session_state.pop("queued_prompt")

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

st.caption("Agent tools: schema_map, daily_consumption, peak_offpeak, predict_month • Auto schema discovery • Azure OpenAI")

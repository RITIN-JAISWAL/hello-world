import os
import json
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px

from src.agents.energy_agent import build_agent
from src.data_access import dbx
from src.charts.utils import line_consumption, pie_import_export

# ---------- Page ----------
st.set_page_config(page_title="Metering Agentic POC", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #ffffff !important; }
    .kpi { font-size: 28px; font-weight: 700; }
    .subkpi { color:#666; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Metering Agentic AI – Settlement POC")
st.caption("Gold: sdr_gold_dev.meter_data • Silver: sdr_silver_dev.silver_meter_data.tbl_meterdata")

# ---------- Filters ----------
colf1, colf2, colf3, colf4 = st.columns([1, 1, 1, 2])
with colf1:
    start = st.date_input("Start date", value=date.today() - timedelta(days=30))
with colf2:
    end = st.date_input("End date", value=date.today())
with colf3:
    region = st.text_input("Region / GSP (optional)", value="")
with colf4:
    if st.button("Refresh data"):
        st.session_state.pop("daily_df", None)
        st.session_state.pop("pp_df", None)

# ---------- Data / KPIs ----------
def load_daily():
    # ✅ uses auto-schema function (no hard-coded cols)
    return dbx.agg_daily(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region or None)


daily_df = st.session_state.get("daily_df")
if daily_df is None:
    try:
        daily_df = load_daily()
        st.session_state["daily_df"] = daily_df
    except Exception as e:
        st.error(f"Failed to pull daily consumption. {e}")
        daily_df = pd.DataFrame(columns=["reading_date", "region", "total_kwh"])

total_kwh = float(daily_df["total_kwh"].sum()) if not daily_df.empty else 0.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total kWh", f"{total_kwh:,.0f}")
k2.metric("Days", f"{daily_df['reading_date'].nunique() if not daily_df.empty else 0}")
k3.metric("Regions", f"{daily_df['region'].nunique() if not daily_df.empty else 0}")
k4.metric("Data Window", f"{start} → {end}")

# ---------- Charts ----------
c1, c2 = st.columns([2, 1])
with c1:
    if not daily_df.empty:
        st.plotly_chart(line_consumption(daily_df), use_container_width=True, theme=None)
    else:
        st.info("No data for selected filters.")

with c2:
    try:
        # ✅ auto-schema import/export breakdown
        raw = dbx.import_export_breakdown(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region or None
        )
        if not raw.empty:
            st.plotly_chart(pie_import_export(raw), use_container_width=True, theme=None)
        else:
            st.info("No rows for import/export breakdown.")
    except Exception:
        st.warning("Import/Export pie unavailable.")

# Peak vs Off-Peak
pp_df = st.session_state.get("pp_df")
if pp_df is None:
    try:
        pp_df = dbx.peak_offpeak(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), region or None)
        st.session_state["pp_df"] = pp_df
    except Exception:
        pp_df = pd.DataFrame()

if not pp_df.empty and "peak_kwh" in pp_df.columns and pp_df["peak_kwh"].notna().any():
    st.subheader("Peak vs Off-Peak")
    mdf = pp_df.melt(
        id_vars=["reading_date"], value_vars=["peak_kwh", "offpeak_kwh"], var_name="type", value_name="kwh"
    )
    fig = px.bar(mdf, x="reading_date", y="kwh", color="type", barmode="group")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True, theme=None)

st.divider()

# ---------- Agentic Chat ----------
st.subheader("Chat with your data")
agent = build_agent()

if "history" not in st.session_state:
    st.session_state["history"] = []

for role, msg in st.session_state["history"]:
    st.chat_message(role).markdown(msg)

user_msg = st.chat_input("Ask e.g. 'Predict next month for London' or 'Peak vs off-peak last 14 days'")
if user_msg:
    st.session_state["history"].append(("user", user_msg))
    st.chat_message("user").markdown(user_msg)

    try:
        response = agent.invoke({"messages": [m for _, m in st.session_state["history"]]})
        assistant_msg = response["messages"][-1].content
    except Exception as e:
        assistant_msg = f"Error: {e}"

    st.session_state["history"].append(("assistant", assistant_msg))
    st.chat_message("assistant").markdown(assistant_msg)

st.caption(
    "Agent tools: schema_map, daily_consumption, peak_offpeak, predict_month. Uses Azure OpenAI + auto schema discovery."
)
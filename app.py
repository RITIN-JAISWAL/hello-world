# app.py
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from src.data_access import dbx
from src.agents.energy_agent import RichEnergyAgent
from src.personas.loader import load_personas
from src.charts.utils import (
    line_consumption,
    area_stacked,
    bar_top_regions,
    donut_voltage,
    line_forecast,
)

# -------------------- Page & Styles --------------------
st.set_page_config(page_title="Metering Agentic POC", layout="wide")

st.markdown("""
<style>
.header-block { background:#0b3d91; color:#fff; padding:18px 22px; border-radius:16px; margin-bottom:14px; }
.header-block h1 { color:#fff !important; margin:0; font-weight:800; line-height:1.15; font-size:28px; }
.card { background:#fff; border:1px solid #e9ecef; border-radius:16px; padding:16px; box-shadow:0 6px 18px rgba(0,0,0,0.06); margin-bottom:16px; }
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

# -------------------- Filters (UI unchanged) --------------------
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

def dstr(d: date) -> str: return d.strftime("%Y-%m-%d")
start_s, end_s = dstr(start), dstr(end)

# -------------------- Data pulls (respect filters) --------------------
daily_df   = dbx.agg_daily(start_s, end_s, region)
totals_df  = dbx.totals_by_region(start_s, end_s, top_n=12, region=region)
voltage_df = dbx.voltage_breakdown(start_s, end_s, region)

# -------------------- Highlights (unchanged) --------------------
total_kwh = float(daily_df["total_kwh"].sum()) if not daily_df.empty else 0.0
days_in_window = daily_df["reading_date"].nunique() if not daily_df.empty else 0
regions_in_view = daily_df["region"].nunique() if not daily_df.empty else 0
mpans = dbx.distinct_mpan_count(start_s, end_s, region)

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

# -------------------- Charts (unchanged) --------------------
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
        if not voltage_df.empty:
            st.subheader("Voltage mix")
            st.plotly_chart(donut_voltage(voltage_df), use_container_width=True, theme=None)
        else:
            st.info("No voltage/connection-type data.")

    if not totals_df.empty:
        st.subheader("Top regions (total kWh)" if region is None else f"Totals for {region}")
        st.plotly_chart(bar_top_regions(totals_df), use_container_width=True, theme=None)
    else:
        st.info("No region totals.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Chat (persona affects TEXT ONLY) --------------------
PERSONAS = load_personas()
PERSONA_KEYS = list(PERSONAS.keys())
PERSONA_LABELS = [PERSONAS[k]["display_name"] for k in PERSONA_KEYS]

st.markdown('<div class="card">', unsafe_allow_html=True)
hdr_left, hdr_right = st.columns([1,1])
with hdr_left:
    st.subheader("Chat with your data")
with hdr_right:
    persona_label = st.radio("View as", PERSONA_LABELS, horizontal=True, key="persona_radio")
    persona_id = PERSONA_KEYS[PERSONA_LABELS.index(persona_label)]

if "agent" not in st.session_state:
    st.session_state["agent"] = RichEnergyAgent()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # [(role, text)]

# replay
for role, text in st.session_state["chat_history"]:
    st.chat_message("user" if role == "user" else "assistant").markdown(text)

user_msg = st.chat_input("Try: 'total kWh for London last month', 'plot usage for Yorkshire', 'forecast next 2 months for London'")
if user_msg:
    st.session_state["chat_history"].append(("user", user_msg))
    st.chat_message("user").markdown(user_msg)

    try:
        res = st.session_state["agent"].answer(user_msg, start_s, end_s, region, persona_id=persona_id)
        st.session_state["chat_history"].append(("assistant", res.text))
        st.chat_message("assistant").markdown(res.text)

        chart = getattr(res, "chart", None)
        if chart:
            ctype = chart.get("type")
            if ctype == "timeseries":
                df = chart["data"]
                st.plotly_chart(area_stacked(df) if df["region"].nunique() > 3 else line_consumption(df),
                                use_container_width=True, theme=None)
            elif ctype == "top_regions":
                st.plotly_chart(bar_top_regions(chart["data"]), use_container_width=True, theme=None)
            elif ctype == "voltage":
                st.plotly_chart(donut_voltage(chart["data"]), use_container_width=True, theme=None)
            elif ctype == "forecast":
                st.plotly_chart(line_forecast(chart["hist"], chart["forecast"]),
                                use_container_width=True, theme=None)
    except Exception as e:
        err = f"Sorry, I hit an error: {e}"
        st.session_state["chat_history"].append(("assistant", err))
        st.chat_message("assistant").markdown(err)

st.markdown('</div>', unsafe_allow_html=True)

st.caption("Persona switch changes chat tone only — UI & charts remain constant")

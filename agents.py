# src/agents/energy_agent.py
"""
SimpleEnergyAgent:
A lightweight, dependency-free chat layer that understands a few intents
and uses src.data_access.dbx to answer with real numbers.

Understands:
- "kwh"/"consumption" (sum) for a region or all regions
- basic time windows: "this month", "last month", "last 7/14/30 days"
- "predict next month" (naive forecast = mean of last 3 months)

If a region is not found in the prompt, it falls back to the Region
filter selected in the UI (passed into answer()).
"""
import re
from datetime import date, timedelta
import pandas as pd

from src.data_access import dbx


def _best_region_from_prompt(prompt: str) -> str | None:
    p = prompt.lower()
    regions = dbx.distinct_regions()
    # try exact or contained match (case-insensitive)
    for r in regions:
        rn = r.strip()
        if not rn:
            continue
        if rn.lower() in p:
            return rn
    return None


def _parse_window(prompt: str, today: date) -> tuple[str, str, bool]:
    """
    Returns (start_iso, end_iso, is_forecast) where is_forecast means "predict next month".
    """
    p = prompt.lower()
    # Predict next month
    if "predict" in p or "forecast" in p:
        if "next month" in p or "nextmonth" in p:
            # we'll compute last 3 finished months and forecast the NEXT month
            # caller will ignore start/end and pass a larger history window
            return ("", "", True)

    # This month
    if "this month" in p:
        start = today.replace(day=1)
        return (start.isoformat(), today.isoformat(), False)

    # Last month
    if "last month" in p:
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return (last_month_start.isoformat(), last_month_end.isoformat(), False)

    # last N days
    m = re.search(r"last\s+(\d+)\s*days?", p)
    if m:
        n = int(m.group(1))
        start = today - timedelta(days=n)
        return (start.isoformat(), today.isoformat(), False)

    # default (UI window)
    return ("", "", False)


class SimpleEnergyAgent:
    """Stateless helper; Streamlit keeps the chat history UI-side."""

    def answer(self, prompt: str, ui_start_iso: str, ui_end_iso: str, ui_region: str | None) -> str:
        today = date.today()

        region = _best_region_from_prompt(prompt) or ui_region
        start_iso, end_iso, is_forecast = _parse_window(prompt, today)

        # Resolve default window to the UI selection
        start_iso = start_iso or ui_start_iso
        end_iso = end_iso or ui_end_iso

        # CONSUMPTION total?
        if any(k in prompt.lower() for k in ["kwh", "consumption", "total"]):
            # forecast branch
            if is_forecast:
                # get ~180 days history to build a monthly series
                hist_start = (today.replace(day=1) - timedelta(days=120)).isoformat()
                hist_end = ui_end_iso
                df = dbx.agg_daily(hist_start, hist_end, region)
                if df.empty:
                    return "I couldn't find enough history to forecast next month."
                s = (
                    df.assign(reading_date=pd.to_datetime(df["reading_date"]))
                      .set_index("reading_date")["total_kwh"]
                      .resample("MS").sum()  # monthly start
                )
                if len(s) < 3:
                    return "Not enough complete months to build a forecast."
                # naive forecast: mean of last 3 complete months
                last_complete = s.index.max()
                s3 = s[s.index <= last_complete].tail(3)
                pred = float(s3.mean())
                next_month = (last_complete + pd.offsets.MonthBegin(1)).strftime("%B %Y")
                scope = region or "All regions"
                return f"**Forecast** for {scope} in {next_month}: **{pred:,.0f} kWh** (mean of last 3 months)."

            # factual total for a window
            df = dbx.agg_daily(start_iso, end_iso, region)
            total = float(df["total_kwh"].sum()) if not df.empty else 0.0
            scope = region or "All regions"
            return f"Total consumption for **{scope}** from **{start_iso}** to **{end_iso}** is **{total:,.0f} kWh**."

        # Peak vs Off-peak?
        if "peak" in prompt.lower():
            df = dbx.peak_offpeak(start_iso, end_iso, region)
            if df.empty or df["peak_kwh"].isna().all():
                return "Peak vs Off-peak breakdown isn't available in the underlying dimensions."
            pk = float(df["peak_kwh"].sum())
            op = float(df["offpeak_kwh"].sum())
            scope = region or "All regions"
            return f"**Peak**: {pk:,.0f} kWh, **Off-peak**: {op:,.0f} kWh for **{scope}** between {start_iso} and {end_iso}."

        # Regions list?
        if "what regions" in prompt.lower() or "list regions" in prompt.lower():
            regs = dbx.distinct_regions()
            return "Available regions: " + ", ".join(regs)

        # Default: try total
        df = dbx.agg_daily(start_iso, end_iso, region)
        total = float(df["total_kwh"].sum()) if not df.empty else 0.0
        scope = region or "All regions"
        return f"{scope}: {total:,.0f} kWh from {start_iso} to {end_iso}."

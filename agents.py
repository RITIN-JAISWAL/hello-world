# src/agents/energy_agent.py
"""
RichEnergyAgent:
- Answers totals (kWh) using src.data_access.dbx
- Can create charts when the user says "plot/chart/graph"
  * timeseries (default)
  * "top regions" bar
  * "voltage" donut
- Forecasts:
  * "predict/forecast next N months" (monthly line with history+forecast)
    - simple trend from last 6 complete months
  * "predict/forecast next N days" (daily seasonal-naïve from last 4 weeks)
Returns a dict:
{
  "text": "...",
  "chart": { "type": "...", "data": <DataFrame> }                    # for timeseries/top/voltage
  or
  "chart": { "type": "forecast", "hist": <DataFrame>, "forecast": <DataFrame> }  # for forecasts
}
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.data_access import dbx


def _best_region_from_prompt(prompt: str) -> Optional[str]:
    p = prompt.lower()
    for r in dbx.distinct_regions():
        rr = (r or "").strip()
        if not rr:
            continue
        if rr.lower() in p:
            return rr
    return None


def _parse_window(prompt: str, today: date) -> tuple[str, str]:
    """Returns (start_iso, end_iso); empty strings mean 'use UI defaults'."""
    p = prompt.lower()

    if "this month" in p:
        start = today.replace(day=1)
        return (start.isoformat(), today.isoformat())

    if "last month" in p:
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return (last_month_start.isoformat(), last_month_end.isoformat())

    m = re.search(r"last\s+(\d+)\s*days?", p)
    if m:
        n = int(m.group(1))
        start = today - timedelta(days=n)
        return (start.isoformat(), today.isoformat())

    return ("", "")


def _want_plot(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in ["plot", "chart", "graph", "visualise", "visualize", "show me"])


def _want_voltage(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in ["voltage", "connection", "donut", "doughnut"])


def _want_top_regions(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in ["top region", "top regions", "bar", "ranking", "leaderboard"])


def _want_forecast(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in ["forecast", "predict"])


def _extract_horizon(prompt: str, unit: str, default_n: int) -> int:
    """
    Extract 'next N <unit>' from prompt, else default.
    unit: 'day' or 'month'
    """
    p = prompt.lower()
    m = re.search(r"next\s+(\d+)\s*%ss?" % unit, p)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            pass
    # also accept "next month" / "next day" without number
    if f"next {unit}" in p:
        return 1
    return default_n


@dataclass
class ChatResult:
    text: str
    chart: Optional[Dict[str, Any]] = None


class RichEnergyAgent:
    """Stateless; Streamlit stores chat history. Build answers and optional charts."""

    def answer(self, prompt: str, ui_start_iso: str, ui_end_iso: str, ui_region: Optional[str]) -> ChatResult:
        today = date.today()
        region = _best_region_from_prompt(prompt) or ui_region
        start_iso, end_iso = _parse_window(prompt, today)
        start_iso = start_iso or ui_start_iso
        end_iso = end_iso or ui_end_iso

        p = prompt.lower()

        # ---------------------- Forecasts ----------------------
        if _want_forecast(p):
            if "month" in p:
                n = _extract_horizon(p, "month", default_n=1)
                return self._forecast_months(region, n, today)
            else:
                n = _extract_horizon(p, "day", default_n=7)
                return self._forecast_days(region, n, today)

        # ---------------------- Charts on demand ----------------------
        if _want_plot(p):
            if _want_voltage(p):
                vdf = dbx.voltage_breakdown(start_iso, end_iso, region)
                if vdf.empty:
                    return ChatResult(text="No voltage/connection-type data in that window.")
                return ChatResult(
                    text=f"Voltage mix for {region or 'all regions'} from {start_iso} to {end_iso}.",
                    chart={"type": "voltage", "data": vdf},
                )

            if _want_top_regions(p):
                tdf = dbx.totals_by_region(start_iso, end_iso, top_n=12, region=region)
                if tdf.empty:
                    return ChatResult(text="No region totals available.")
                hdr = f"Top regions by kWh from {start_iso} to {end_iso}" if region is None else f"Totals for {region}"
                return ChatResult(
                    text=hdr,
                    chart={"type": "top_regions", "data": tdf},
                )

            # default chart: time series
            ddf = dbx.agg_daily(start_iso, end_iso, region)
            if ddf.empty:
                return ChatResult(text="No time-series data for the requested window.")
            return ChatResult(
                text=f"Time series for {region or 'all regions'} from {start_iso} to {end_iso}.",
                chart={"type": "timeseries", "data": ddf},
            )

        # ---------------------- Numeric totals (fallback) ----------------------
        if any(k in p for k in ["kwh", "consumption", "total"]):
            ddf = dbx.agg_daily(start_iso, end_iso, region)
            total = float(ddf["total_kwh"].sum()) if not ddf.empty else 0.0
            scope = region or "All regions"
            return ChatResult(text=f"Total consumption for **{scope}** from **{start_iso}** to **{end_iso}** is **{total:,.0f} kWh**.")

        # regions list
        if "what regions" in p or "list regions" in p:
            return ChatResult(text="Available regions: " + ", ".join(dbx.distinct_regions()))

        # default
        ddf = dbx.agg_daily(start_iso, end_iso, region)
        total = float(ddf["total_kwh"].sum()) if not ddf.empty else 0.0
        scope = region or "All regions"
        return ChatResult(text=f"{scope}: {total:,.0f} kWh from {start_iso} to {end_iso}.")

    # ---------------------- Forecast helpers ----------------------
    def _forecast_months(self, region: Optional[str], n_months: int, today: date) -> ChatResult:
        """Monthly forecast using a simple linear trend on the last 6 complete months."""
        hist_start = (today.replace(day=1) - timedelta(days=210)).isoformat()  # ~7 months back
        hist_end = today.isoformat()
        df = dbx.agg_daily(hist_start, hist_end, region)
        if df.empty:
            return ChatResult(text="Not enough history to forecast monthly usage.")

        s = (
            df.assign(reading_date=pd.to_datetime(df["reading_date"]))
              .set_index("reading_date")["total_kwh"]
              .resample("MS").sum()
        )
        # keep last 6 complete months
        s = s.dropna().iloc[:-0]  # ensure Series
        if len(s) < 3:
            return ChatResult(text="Need ≥3 months of history to forecast.")

        # linear trend on last up to 6 months
        y = s.tail(6).values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        last_month_start = s.tail(1).index[0]
        hist_df = s.tail(6).rename("kwh").reset_index().rename(columns={"reading_date":"date", "index":"date"})

        # build forecast months
        fcst_vals = []
        base_idx = len(y) - 1
        last_val = y[-1]
        for i in range(1, n_months + 1):
            # extrapolate from trend
            idx = base_idx + i
            val = slope * idx + intercept
            val = float(max(0.0, val))
            fcst_vals.append(val)
        future_idx = pd.date_range(last_month_start + pd.offsets.MonthBegin(1), periods=n_months, freq="MS")
        fcst_df = pd.DataFrame({"date": future_idx, "kwh": fcst_vals})

        scope = region or "All regions"
        msg = f"**Forecast** for {scope} (monthly): " + ", ".join(
            [f"{d.strftime('%b %Y')}: {v:,.0f} kWh" for d, v in zip(fcst_df['date'], fcst_df['kwh'])]
        )
        return ChatResult(
            text=msg,
            chart={"type": "forecast", "hist": hist_df, "forecast": fcst_df},
        )

    def _forecast_days(self, region: Optional[str], n_days: int, today: date) -> ChatResult:
        """Daily forecast using a seasonal naïve model (average of the last 4 same weekdays)."""
        hist_start = (today - timedelta(days=60)).isoformat()
        hist_end = today.isoformat()
        df = dbx.agg_daily(hist_start, hist_end, region)
        if df.empty:
            return ChatResult(text="Not enough history to forecast daily usage.")

        d = (df.assign(reading_date=pd.to_datetime(df["reading_date"]))
               .groupby("reading_date", as_index=False)["total_kwh"].sum())
        d["weekday"] = d["reading_date"].dt.weekday  # 0=Mon
        # mean by weekday over last 4 weeks
        recent = d[d["reading_date"] >= (pd.to_datetime(hist_end) - pd.Timedelta(days=28))]
        prof = recent.groupby("weekday")["total_kwh"].mean().to_dict()

        # build future dates
        start_next = pd.to_datetime(hist_end) + pd.Timedelta(days=1)
        future_dates = pd.date_range(start_next, periods=n_days, freq="D")
        fcst_vals = [float(max(0.0, prof.get(dt.weekday(), recent["total_kwh"].mean()))) for dt in future_dates]
        hist_df = d.rename(columns={"reading_date": "date", "total_kwh": "kwh"})[["date", "kwh"]].tail(30)
        fcst_df = pd.DataFrame({"date": future_dates, "kwh": fcst_vals})

        scope = region or "All regions"
        msg = f"**Daily forecast** for {scope} (next {n_days} days): total {sum(fcst_vals):,.0f} kWh."
        return ChatResult(text=msg, chart={"type": "forecast", "hist": hist_df, "forecast": fcst_df})

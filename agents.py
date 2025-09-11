# src/agents/energy_agent.py
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from src.data_access import dbx
from src.personas.loader import load_personas, persona_system_prompt
from src.llm.azure_openai import rewrite_with_persona

# ---------- prompt parsing helpers ----------
def _best_region_from_prompt(prompt: str) -> Optional[str]:
    p = prompt.lower()
    for r in dbx.distinct_regions():
        rr = (r or "").strip()
        if rr and rr.lower() in p:
            return rr
    return None

def _parse_window(prompt: str, today: date) -> tuple[str, str]:
    p = prompt.lower()
    if "this month" in p:
        s = today.replace(day=1); return (s.isoformat(), today.isoformat())
    if "last month" in p:
        ft = today.replace(day=1); le = ft - timedelta(days=1); ls = le.replace(day=1)
        return (ls.isoformat(), le.isoformat())
    m = re.search(r"last\s+(\d+)\s*days?", p)
    if m:
        n = int(m.group(1)); s = today - timedelta(days=n); return (s.isoformat(), today.isoformat())
    return ("","")

def _has_any(p: str, words): return any(w in p for w in words)
def _want_plot(p): return _has_any(p, ["plot","chart","graph","visualise","visualize","show me"])
def _want_voltage(p): return _has_any(p, ["voltage","connection","donut","doughnut"])
def _want_top_regions(p): return _has_any(p, ["top region","top regions","bar","ranking","leaderboard"])
def _want_forecast(p): return _has_any(p, ["forecast","predict"])

def _horizon(p: str, unit: str, default_n: int) -> int:
    m = re.search(rf"next\s+(\d+)\s*{unit}s?", p)
    if m:
        try: return max(1, int(m.group(1)))
        except: pass
    if f"next {unit}" in p: return 1
    return default_n

@dataclass
class ChatResult:
    text: str
    chart: Optional[Dict[str, Any]] = None  # 'timeseries' | 'top_regions' | 'voltage' | 'forecast'

class RichEnergyAgent:
    """
    Persona-aware agent. Numbers/charts come from dbx; final text is rephrased by AOAI
    according to the selected persona style. UI charts remain unchanged.
    """
    def __init__(self):
        self._personas = load_personas()  # {id: {display_name, style}}

    def _personaize(self, text: str, persona_id: str) -> str:
        style = self._personas.get(persona_id, {}).get("style", "")
        sys_prompt = persona_system_prompt(style)
        return rewrite_with_persona(text, sys_prompt)

    def answer(self, prompt: str, ui_start: str, ui_end: str,
               ui_region: Optional[str], persona_id: str = "exec") -> ChatResult:
        today = date.today()
        p = prompt.lower()
        region = _best_region_from_prompt(prompt) or ui_region
        start, end = _parse_window(prompt, today)
        start = start or ui_start
        end = end or ui_end

        # ---- Forecasts ----
        if _want_forecast(p):
            if "month" in p:
                res = self._forecast_months(region, _horizon(p, "month", 1), today)
            else:
                res = self._forecast_days(region, _horizon(p, "day", 7), today)
            res.text = self._personaize(res.text, persona_id)
            return res

        # ---- Charts on demand (payload unchanged) ----
        if _want_plot(p):
            if _want_voltage(p):
                vdf = dbx.voltage_breakdown(start, end, region)
                msg = f"Voltage mix for {region or 'all regions'} from {start} to {end}."
                return ChatResult(self._personaize(msg, persona_id), {"type":"voltage","data":vdf}) if not vdf.empty \
                       else ChatResult(self._personaize("No voltage/connection-type data in that window.", persona_id))
            if _want_top_regions(p):
                tdf = dbx.totals_by_region(start, end, top_n=12, region=region)
                hdr = f"Top regions by kWh from {start} to {end}" if region is None else f"Totals for {region}"
                return ChatResult(self._personaize(hdr, persona_id), {"type":"top_regions","data":tdf}) if not tdf.empty \
                       else ChatResult(self._personaize("No region totals available.", persona_id))
            ddf = dbx.agg_daily(start, end, region)
            msg = f"Time series for {region or 'all regions'} from {start} to {end}."
            return ChatResult(self._personaize(msg, persona_id), {"type":"timeseries","data":ddf}) if not ddf.empty \
                   else ChatResult(self._personaize("No time-series data for the requested window.", persona_id))

        # ---- Numeric totals fallback ----
        if _has_any(p, ["kwh","consumption","total"]):
            ddf = dbx.agg_daily(start, end, region)
            total = float(ddf["total_kwh"].sum()) if not ddf.empty else 0.0
            scope = region or "All regions"
            msg = f"Total consumption for **{scope}** from **{start}** to **{end}** is **{total:,.0f} kWh**."
            return ChatResult(self._personaize(msg, persona_id))

        if _has_any(p, ["what regions","list regions"]):
            return ChatResult(self._personaize("Available regions: " + ", ".join(dbx.distinct_regions()), persona_id))

        # default
        ddf = dbx.agg_daily(start, end, region)
        total = float(ddf["total_kwh"].sum()) if not ddf.empty else 0.0
        scope = region or "All regions"
        msg = f"{scope}: {total:,.0f} kWh from {start} to {end}."
        return ChatResult(self._personaize(msg, persona_id))

    # ---------- Forecast helpers ----------
    def _forecast_months(self, region: Optional[str], n: int, today: date) -> ChatResult:
        hist_start = (today.replace(day=1) - timedelta(days=210)).isoformat()
        df = dbx.agg_daily(hist_start, today.isoformat(), region)
        if df.empty: return ChatResult("Not enough history to forecast monthly usage.")
        s = (df.assign(reading_date=pd.to_datetime(df["reading_date"]))
               .set_index("reading_date")["total_kwh"].resample("MS").sum()).dropna()
        if len(s) < 3: return ChatResult("Need ≥3 months of history to forecast.")
        y = s.tail(6).values; x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        last_m = s.index.max()
        hist_df = s.tail(6).rename("kwh").reset_index().rename(columns={"reading_date":"date"})
        vals = [float(max(0.0, slope*((len(y)-1)+i) + intercept)) for i in range(1, n+1)]
        idx = pd.date_range(last_m + pd.offsets.MonthBegin(1), periods=n, freq="MS")
        fcst_df = pd.DataFrame({"date": idx, "kwh": vals})
        scope = region or "All regions"
        msg = "Forecast for {} (monthly): ".format(scope) + ", ".join(
            [f"{d.strftime('%b %Y')}: {v:,.0f} kWh" for d, v in zip(idx, vals)]
        )
        return ChatResult(msg, {"type":"forecast","hist":hist_df, "forecast":fcst_df})

    def _forecast_days(self, region: Optional[str], n: int, today: date) -> ChatResult:
        hist_start = (today - timedelta(days=60)).isoformat()
        df = dbx.agg_daily(hist_start, today.isoformat(), region)
        if df.empty: return ChatResult("Not enough history to forecast daily usage.")
        d = (df.assign(reading_date=pd.to_datetime(df["reading_date"]))
               .groupby("reading_date", as_index=False)["total_kwh"].sum())
        d["weekday"] = d["reading_date"].dt.weekday
        recent = d[d["reading_date"] >= (pd.to_datetime(today) - pd.Timedelta(days=28))]
        prof = recent.groupby("weekday")["total_kwh"].mean().to_dict()
        start_next = pd.to_datetime(today) + pd.Timedelta(days=1)
        future = pd.date_range(start_next, periods=n, freq="D")
        vals = [float(max(0.0, prof.get(dt.weekday(), recent["total_kwh"].mean()))) for dt in future]
        hist_df = d.rename(columns={"reading_date":"date","total_kwh":"kwh"})[["date","kwh"]].tail(30)
        fcst_df = pd.DataFrame({"date": future, "kwh": vals})
        scope = region or "All regions"
        msg = f"Daily forecast for {scope} (next {n} days): total {sum(vals):,.0f} kWh."
        return ChatResult(msg, {"type":"forecast","hist":hist_df, "forecast":fcst_df})

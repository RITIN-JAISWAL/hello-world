import os
import json
import datetime as dt
from calendar import monthrange
from typing import Dict, Any

import pandas as pd
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI  # ✅ Azure-specific
from sklearn.linear_model import LinearRegression

from src.data_access import dbx


def _llm() -> AzureChatOpenAI:
    """Azure OpenAI chat model (no OPENAI_API_KEY required)."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0.2,
    )


# ------------------------- Tools -------------------------

@tool
def schema_map(_: Dict[str, Any]) -> str:
    """Return the auto-inferred semantic map (tables/columns) as JSON."""
    return dbx.get_semantic_map_json()


@tool
def daily_consumption(params: Dict[str, Any]) -> str:
    """
    Return daily totals between dates (region optional).
    Params:
      - start_date: YYYY-MM-DD
      - end_date:   YYYY-MM-DD
      - region:     (optional) region or GSP group
    """
    df = dbx.agg_daily(params["start_date"], params["end_date"], params.get("region"))
    return df.to_json(orient="records")


@tool
def peak_offpeak(params: Dict[str, Any]) -> str:
    """
    Peak/off-peak split (joins dim_halfhour if available).
    Params: start_date, end_date, region?
    """
    df = dbx.peak_offpeak(params["start_date"], params["end_date"], params.get("region"))
    return df.to_json(orient="records")


@tool
def predict_month(params: Dict[str, Any]) -> str:
    """
    Predict this and next month totals for an optional region, using last ~180 days.
    Params: region? (optional)
    """
    region = params.get("region")

    end = dt.date.today()
    start = end - dt.timedelta(days=180)
    hist = dbx.agg_daily(start.isoformat(), end.isoformat(), region)
    if hist.empty:
        return json.dumps({"error": "no data"})

    hist["reading_date"] = pd.to_datetime(hist["reading_date"])
    hist = hist.sort_values("reading_date")

    X = (hist["reading_date"] - hist["reading_date"].min()).dt.days.values.reshape(-1, 1)
    y = hist["total_kwh"].values
    model = LinearRegression().fit(X, y)

    def month_sum(y0: int, m0: int) -> float:
        d0 = pd.Timestamp(year=y0, month=m0, day=1)
        days = monthrange(y0, m0)[1]
        rng = pd.date_range(d0, periods=days, freq="D")
        Xf = ((rng - hist["reading_date"].min()).days.values).reshape(-1, 1)
        return float(model.predict(Xf).sum())

    today = pd.Timestamp.today()
    this_total = month_sum(today.year, today.month)
    nm = today + pd.offsets.MonthBegin(1)
    next_total = month_sum(nm.year, nm.month)

    return json.dumps(
        {
            "region": region,
            "this_month_est_kwh": round(this_total, 2),
            "next_month_est_kwh": round(next_total, 2),
            "trend": "up" if next_total > this_total else "down" if next_total < this_total else "flat",
        }
    )


TOOLS = [schema_map, daily_consumption, peak_offpeak, predict_month]


def build_agent():
    """LangGraph agent that plans + calls tools."""
    llm = _llm().bind_tools(TOOLS)
    graph = StateGraph(MessagesState)

    def call_llm(state: MessagesState):
        out = llm.invoke(state["messages"])
        return {"messages": [out]}

    from langgraph.prebuilt import ToolNode

    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")
    return graph.compile()
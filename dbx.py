import os, re, json, time
import pandas as pd
from databricks import sql
from typing import Optional, Dict, Any, List

HOST = os.getenv("DATABRICKS_HOST")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")

GOLD_CATALOG = os.getenv("GOLD_CATALOG_NAME", "sdr_gold_dev")
GOLD_SCHEMA  = os.getenv("GOLD_SCHEMA_NAME",  "meter_data")
SILVER_CATALOG = os.getenv("SILVER_CATALOG_NAME", "sdr_silver_dev")
SILVER_SCHEMA  = os.getenv("SILVER_SCHEMA_NAME",  "silver_meter_data")
SILVER_TABLE   = os.getenv("SILVER_TABLE_NAME",   "tbl_meterdata")

_cache: Dict[str, Any] = {"ts": 0, "semantic": None}
TTL = 25 * 60  # 25 min cache, aligns with ~30 min data refresh

# ------------------ Core SQL helpers ------------------

def _connect():
    if not (HOST and HTTP_PATH and TOKEN):
        raise RuntimeError("Databricks SQL credentials missing (DATABRICKS_HOST/HTTP_PATH/TOKEN).")
    return sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN)

def query_df(query: str) -> pd.DataFrame:
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as c:
            c.execute(query)
            rows = c.fetchall()
            cols = [d[0] for d in c.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        if conn: conn.close()

def discover_tables(catalog: str, schema: str) -> pd.DataFrame:
    q = f"""
    SELECT table_name
    FROM system.information_schema.tables
    WHERE table_catalog='{catalog}' AND table_schema='{schema}'
    ORDER BY table_name
    """
    return query_df(q)

def head_table(catalog: str, schema: str, table: str, n: int = 5) -> pd.DataFrame:
    return query_df(f"SELECT * FROM {catalog}.{schema}.{table} LIMIT {n}")

def _columns(catalog: str, schema: str, table: str) -> pd.DataFrame:
    q = f"""
    SELECT column_name, data_type
    FROM system.information_schema.columns
    WHERE table_catalog='{catalog}' AND table_schema='{schema}' AND table_name='{table}'
    """
    return query_df(q)

def _best_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for pat in patterns:
        r = re.compile(pat, re.I)
        for lc, orig in lower.items():
            if r.search(lc):
                return orig
    return None

# ------------------ Semantic inference ------------------

def infer_semantic_map() -> Dict[str, Any]:
    """Auto-detect tables/columns once, cache ~25 min."""
    now = time.time()
    if _cache["semantic"] and now - _cache["ts"] < TTL:
        return _cache["semantic"]

    gold_tables = discover_tables(GOLD_CATALOG, GOLD_SCHEMA)["table_name"].str.lower().tolist()
    gold_daily_mv = next((t for t in gold_tables if t.startswith("mv_agg_meterdata_daily")), None)
    dim_gsp = "dim_gspgroup" if "dim_gspgroup" in gold_tables else None
    dim_halfhour = "dim_halfhour" if "dim_halfhour" in gold_tables else None

    silver_cols = _columns(SILVER_CATALOG, SILVER_SCHEMA, SILVER_TABLE)
    colnames = silver_cols["column_name"].tolist()

    date_col   = _best_col(colnames, [r"UTCSettlementDate", r"settlement.*date", r".*date$"])
    value_col  = _best_col(colnames, [r"UTCPeriodConsumptionValue", r"kwh", r"consumption", r"^value$"])
    region_col = _best_col(colnames, [r"region", r"gsp.?group.*", r"^gspgroupid$"])
    mpan_col   = _best_col(colnames, [r"mpan", r"mpancore"])
    period_col = _best_col(colnames, [r"settlementperiod", r"^period$"])

    semantic = {
        "gold": {
            "catalog": GOLD_CATALOG, "schema": GOLD_SCHEMA,
            "daily_mv": gold_daily_mv,
            "dim_gspgroup": dim_gsp,
            "dim_halfhour": dim_halfhour,
        },
        "silver": {
            "catalog": SILVER_CATALOG, "schema": SILVER_SCHEMA, "table": SILVER_TABLE,
            "date_col": date_col, "value_col": value_col, "region_col": region_col,
            "mpan_col": mpan_col, "period_col": period_col,
        },
    }
    _cache["ts"], _cache["semantic"] = now, semantic
    return semantic

def get_semantic_map_json() -> str:
    return json.dumps(infer_semantic_map(), indent=2)

# ------------------ Query helpers (schema-agnostic) ------------------

def agg_daily(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()

    # Prefer Gold MV if present (and optionally join dim_gspgroup)
    if S["gold"]["daily_mv"]:
        mv = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['daily_mv']}"
        region_field = "f.GSPGroupID"
        join_txt = ""
        if S["gold"]["dim_gspgroup"]:
            g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_gspgroup']}"
            join_txt = f" LEFT JOIN {g} g ON f.GSPGroupID = g.GSPGroupID "
            region_field = "COALESCE(g.RegionName, f.GSPGroupID)"
        region_pred = f" AND {region_field} = '{region}'" if region else ""
        q = f"""
        SELECT f.UTCSettlementDate AS reading_date,
               {region_field} AS region,
               SUM(f.total_kwh) AS total_kwh
        FROM {mv} f
        {join_txt}
        WHERE f.UTCSettlementDate >= '{start_date}' AND f.UTCSettlementDate <= '{end_date}' {region_pred}
        GROUP BY f.UTCSettlementDate, {region_field}
        ORDER BY reading_date
        """
        try:
            return query_df(q)
        except Exception:
            pass  # fall back to Silver

    # Silver fallback
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r = s["date_col"], s["value_col"], s["region_col"]
    if not (d and v):
        raise RuntimeError("Could not infer date/value columns from silver table.")
    reg_sel = f"COALESCE({r}, 'Unknown')" if r else "'All'"
    region_pred = f" AND {r} = '{region}'" if (region and r) else ""
    q = f"""
    SELECT {d} AS reading_date,
           {reg_sel} AS region,
           SUM({v}) AS total_kwh
    FROM {fq}
    WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
    GROUP BY {d}, {reg_sel}
    ORDER BY reading_date
    """
    return query_df(q)

def import_export_breakdown(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r = s["date_col"], s["value_col"], s["region_col"]
    if not (d and v):
        raise RuntimeError("Could not infer date/value columns from silver table.")
    region_pred = f" AND {r} = '{region}'" if (region and r) else ""
    q = f"""
    SELECT CASE WHEN {v} >= 0 THEN 'Import' ELSE 'Export' END AS import_export_flag,
           SUM(ABS({v})) AS kwh_consumed
    FROM {fq}
    WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
    GROUP BY CASE WHEN {v} >= 0 THEN 'Import' ELSE 'Export' END
    """
    return query_df(q)

def peak_offpeak(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r, p = s["date_col"], s["value_col"], s["region_col"], s["period_col"]
    if S["gold"]["dim_halfhour"] and p:
        hh = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_halfhour']}"
        region_pred = f" AND {r} = '{region}'" if (region and r) else ""
        q = f"""
        SELECT s.{d} AS reading_date,
               SUM(CASE WHEN COALESCE(hh.IsPeak, false) THEN s.{v} ELSE 0 END) AS peak_kwh,
               SUM(CASE WHEN NOT COALESCE(hh.IsPeak, false) THEN s.{v} ELSE 0 END) AS offpeak_kwh
        FROM {fq} s
        LEFT JOIN {hh} hh ON s.{p} = hh.SettlementPeriod
        WHERE s.{d} >= '{start_date}' AND s.{d} <= '{end_date}' {region_pred}
        GROUP BY s.{d}
        ORDER BY reading_date
        """
        try:
            return query_df(q)
        except Exception:
            pass
    # fallback: no split available
    return agg_daily(start_date, end_date, region)[["reading_date"]].assign(peak_kwh=None, offpeak_kwh=None)

def distinct_regions(limit: int = 200) -> List[str]:
    """Get distinct region/GSP values from Silver (or Gold dim if needed)."""
    S = infer_semantic_map()
    s = S["silver"]
    r = s["region_col"]
    if r:
        fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
        d = s["date_col"] or "1=1"
        q = f"""
        SELECT DISTINCT {r} AS region
        FROM {fq}
        WHERE {d} IS NOT NULL
        ORDER BY region
        LIMIT {limit}
        """
        df = query_df(q)
        return [x for x in df["region"].astype(str).tolist() if x and x.lower() != "null"]
    # fallback to Gold dim
    if S["gold"]["dim_gspgroup"]:
        g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_gspgroup']}"
        try:
            df = query_df(f"SELECT DISTINCT COALESCE(RegionName, GSPGroupID) AS region FROM {g} ORDER BY region LIMIT {limit}")
            return df["region"].astype(str).tolist()
        except Exception:
            pass
    return []

def totals_by_region(start_date: str, end_date: str, top_n: int = 10) -> pd.DataFrame:
    df = agg_daily(start_date, end_date, None)
    if df.empty: return df
    out = df.groupby("region", as_index=False)["total_kwh"].sum().sort_values("total_kwh", ascending=False)
    return out.head(top_n)

def heatmap_day_period(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    """Aggregate by date × settlement period for heatmap (if period column exists)."""
    S = infer_semantic_map()
    s = S["silver"]
    d, v, r, p = s["date_col"], s["value_col"], s["region_col"], s["period_col"]
    if not (d and v and p):
        return pd.DataFrame(columns=["reading_date", "period", "kwh"])
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    region_pred = f" AND {r} = '{region}'" if (region and r) else ""
    q = f"""
    SELECT {d} AS reading_date, {p} AS period, SUM({v}) AS kwh
    FROM {fq}
    WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
    GROUP BY {d}, {p}
    ORDER BY reading_date, period
    """
    return query_df(q)

def distinct_mpan_count(start_date: str, end_date: str, region: Optional[str] = None) -> Optional[int]:
    S = infer_semantic_map()
    s = S["silver"]
    if not s["mpan_col"] or not s["date_col"]:
        return None
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, r, m = s["date_col"], s["region_col"], s["mpan_col"]
    region_pred = f" AND {r} = '{region}'" if (region and r) else ""
    q = f"""
    SELECT COUNT(DISTINCT {m}) AS mpan_cnt
    FROM {fq}
    WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
    """
    try:
        return int(query_df(q)["mpan_cnt"].iloc[0])
    except Exception:
        return None

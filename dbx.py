import os, re, json, time
import pandas as pd
from databricks import sql
from typing import Optional, Dict, Any, List, Tuple

# --------- ENV ----------
HOST = os.getenv("DATABRICKS_HOST")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")

GOLD_CATALOG = os.getenv("GOLD_CATALOG_NAME", "sdr_gold_dev")
GOLD_SCHEMA  = os.getenv("GOLD_SCHEMA_NAME",  "meter_data")
SILVER_CATALOG = os.getenv("SILVER_CATALOG_NAME", "sdr_silver_dev")
SILVER_SCHEMA  = os.getenv("SILVER_SCHEMA_NAME",  "silver_meter_data")
SILVER_TABLE   = os.getenv("SILVER_TABLE_NAME",   "tbl_meterdata")

_cache: Dict[str, Any] = {"ts": 0, "semantic": None, "region_maps": None, "gold_tables": None}
TTL = 25 * 60  # ~25 min (aligns with ~30 min refresh)

# --------- SQL helpers ----------
def _connect():
    if not (HOST and HTTP_PATH and TOKEN):
        raise RuntimeError("Databricks SQL credentials missing (DATABRICKS_HOST/HTTP_PATH/TOKEN).")
    return sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN)

def query_df(q: str) -> pd.DataFrame:
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as c:
            c.execute(q)
            rows = c.fetchall()
            cols = [d[0] for d in c.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        if conn: conn.close()

def discover_tables(catalog: str, schema: str) -> pd.DataFrame:
    return query_df(f"""
        SELECT table_name
        FROM system.information_schema.tables
        WHERE table_catalog='{catalog}' AND table_schema='{schema}'
        ORDER BY table_name
    """)

def _columns(catalog: str, schema: str, table: str) -> pd.DataFrame:
    return query_df(f"""
        SELECT column_name, data_type
        FROM system.information_schema.columns
        WHERE table_catalog='{catalog}' AND table_schema='{schema}' AND table_name='{table}'
        ORDER BY ordinal_position
    """)

def _best_col(cols: List[str], pats: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for pat in pats:
        r = re.compile(pat, re.I)
        for lc, orig in low.items():
            if r.search(lc):
                return orig
    return None

def _sql_in_list(values):
    """Return a SQL IN (...) list with each value single-quoted."""
    return ",".join([f"'{str(v)}'" for v in values])

# --------- Discovery ----------
def infer_semantic_map() -> Dict[str, Any]:
    now = time.time()
    if _cache["semantic"] and now - _cache["ts"] < TTL:
        return _cache["semantic"]

    gold_df = discover_tables(GOLD_CATALOG, GOLD_SCHEMA)
    _cache["gold_tables"] = gold_df
    gold = {t.lower() for t in gold_df["table_name"].tolist()}

    semantic: Dict[str, Any] = {
        "gold": {
            "catalog": GOLD_CATALOG,
            "schema": GOLD_SCHEMA,
            # daily MV if present
            "daily_mv": next((t for t in gold if t.startswith("mv_agg_meterdata_daily")), None),
            # region mapping sources (prefer the view seen in your screenshot)
            "vw_region_map": "vw_mpan_consumption_consent" if "vw_mpan_consumption_consent" in gold else None,
            "dim_gspgroup": "dim_gspgroup" if "dim_gspgroup" in gold else None,
            "dim_halfhour": "dim_halfhour" if "dim_halfhour" in gold else None,
            # voltage/connection type mapping
            "dim_connectiontype": "dim_connectiontype" if "dim_connectiontype" in gold else None,
        },
        "silver": {
            "catalog": SILVER_CATALOG,
            "schema": SILVER_SCHEMA,
            "table": SILVER_TABLE,
        },
    }

    # infer silver cols
    s_cols = _columns(SILVER_CATALOG, SILVER_SCHEMA, SILVER_TABLE)
    cn = s_cols["column_name"].tolist()
    semantic["silver"].update({
        "date_col":   _best_col(cn, [r"UTCSettlementDate", r"settlement.*date", r".*date$"]),
        "value_col":  _best_col(cn, [r"UTCPeriodConsumptionValue", r"kwh", r"consumption", r"^value$"]),
        "region_col": _best_col(cn, [r"region", r"gsp.?group.*", r"^gspgroupid$"]),
        "mpan_col":   _best_col(cn, [r"mpan", r"mpancore"]),
        "period_col": _best_col(cn, [r"settlementperiod", r"^period$"]),
        # connection/voltage indicator in silver (from your tbl_meterdata screenshot)
        "conn_type_col": _best_col(cn, [r"ConnectionTypeIndicator", r"connection.*type.*indicator", r"connection.*type", r"voltage"]),
    })

    _cache["semantic"], _cache["ts"] = semantic, now
    return semantic

def get_semantic_map_json() -> str:
    S = infer_semantic_map()
    tables = _cache["gold_tables"]
    return json.dumps(
        {"semantic": S, "gold_tables": tables["table_name"].tolist() if tables is not None else []},
        indent=2,
    )

# --------- Region mapping (use GOLD VIEW first) ----------
def _normalize_code(x: str) -> str:
    return re.sub(r"^_+", "", str(x or ""))

def _region_maps() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      code_to_name: map of all code variants ('_A','A') -> 'London'
      name_to_codes: 'London' -> ['_A','A']
    """
    now = time.time()
    if _cache["region_maps"] and now - _cache["ts"] < TTL:
        return _cache["region_maps"]

    S = infer_semantic_map()
    code_to_name: Dict[str, str] = {}
    name_to_codes: Dict[str, List[str]] = {}

    # 1) Preferred: the view `vw_mpan_consumption_consent` (has GSPGroupID + GSPGroupDescription)
    if S["gold"]["vw_region_map"]:
        vw = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['vw_region_map']}"
        try:
            df = query_df(f"""
                SELECT DISTINCT GSPGroupID, GSPGroupDescription
                FROM {vw}
                WHERE GSPGroupID IS NOT NULL
            """)
            for _, r in df.iterrows():
                raw = str(r["GSPGroupID"])
                desc = str(r["GSPGroupDescription"]) if pd.notna(r["GSPGroupDescription"]) else raw
                norm = _normalize_code(raw)
                for variant in {raw, norm, f"_{norm}"}:
                    code_to_name[variant] = desc
                name_to_codes.setdefault(desc, [])
                for variant in {raw, norm, f"_{norm}"}:
                    if variant not in name_to_codes[desc]:
                        name_to_codes[desc].append(variant)
        except Exception:
            pass

    # 2) Fallback: dim_gspgroup
    if not code_to_name and S["gold"]["dim_gspgroup"]:
        g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_gspgroup']}"
        try:
            df = query_df(f"SELECT GSPGroupID, RegionName FROM {g}")
            for _, r in df.iterrows():
                raw = str(r["GSPGroupID"])
                name = str(r["RegionName"]) if pd.notna(r["RegionName"]) else raw
                norm = _normalize_code(raw)
                for variant in {raw, norm, f"_{norm}"}:
                    code_to_name[variant] = name
                name_to_codes.setdefault(name, [])
                for variant in {raw, norm, f"_{norm}"}:
                    if variant not in name_to_codes[name]:
                        name_to_codes[name].append(variant)
        except Exception:
            pass

    _cache["region_maps"] = (code_to_name, name_to_codes)
    return _cache["region_maps"]

def _region_predicate(region_input: Optional[str], region_col: Optional[str]) -> str:
    """WHERE clause that accepts either human name or code; expands _A/A variants."""
    if not region_input or not region_col:
        return ""
    code_to_name, name_to_codes = _region_maps()
    # user selected a friendly name
    if region_input in name_to_codes:
        in_list = _sql_in_list(name_to_codes[region_input])
        return f" AND {region_col} IN ({in_list})"
    # treat as code; allow variants
    norm = _normalize_code(region_input)
    variants = [region_input, norm, f"_{norm}"]
    in_list = _sql_in_list(variants)
    return f" AND {region_col} IN ({in_list})"

def _apply_friendly(df: pd.DataFrame, col: str = "region") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    code_to_name, _ = _region_maps()
    if not code_to_name:
        return df
    return df.assign(region=df[col].map(lambda x: code_to_name.get(str(x), str(x))))

# --------- Queries ----------
def agg_daily(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()

    # Prefer Gold MV with friendly name join
    if S["gold"]["daily_mv"]:
        mv = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['daily_mv']}"
        region_field = "f.GSPGroupID"
        join_txt = ""
        if S["gold"]["vw_region_map"]:
            vw = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['vw_region_map']}"
            join_txt = (
                f" LEFT JOIN {vw} v "
                f" ON regexp_replace(f.GSPGroupID, '^_+', '') = regexp_replace(v.GSPGroupID, '^_+', '') "
            )
            region_field = "COALESCE(v.GSPGroupDescription, f.GSPGroupID)"
        elif S["gold"]["dim_gspgroup"]:
            g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_gspgroup']}"
            join_txt = (
                f" LEFT JOIN {g} g "
                f" ON regexp_replace(f.GSPGroupID, '^_+', '') = regexp_replace(g.GSPGroupID, '^_+', '') "
            )
            region_field = "COALESCE(g.RegionName, f.GSPGroupID)"

        region_pred = ""
        if region:
            if "COALESCE(" in region_field:
                region_pred = f" AND {region_field} = '{region}'"
            else:
                region_pred = f" AND f.GSPGroupID IN ('{region}','{_normalize_code(region)}','_{_normalize_code(region)}')"

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
            pass  # fall through to silver

    # Silver fallback
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r = s["date_col"], s["value_col"], s["region_col"]
    if not (d and v):
        raise RuntimeError("Could not infer date/value columns from silver.")
    region_pred = _region_predicate(region, r)
    reg_sel = f"COALESCE({r}, 'Unknown')" if r else "'All'"
    q = f"""
    SELECT {d} AS reading_date,
           {reg_sel} AS region,
           SUM({v}) AS total_kwh
    FROM {fq}
    WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
    GROUP BY {d}, {reg_sel}
    ORDER BY reading_date
    """
    df = query_df(q)
    return _apply_friendly(df, "region")

def import_export_breakdown(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    """(Kept for completeness) Import vs Export breakdown from silver."""
    S = infer_semantic_map()
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r = s["date_col"], s["value_col"], s["region_col"]
    if not (d and v):
        raise RuntimeError("Could not infer date/value columns from silver.")
    region_pred = _region_predicate(region, r)
    return query_df(f"""
        SELECT CASE WHEN {v} >= 0 THEN 'Import' ELSE 'Export' END AS import_export_flag,
               SUM(ABS({v})) AS kwh_consumed
        FROM {fq}
        WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
        GROUP BY CASE WHEN {v} >= 0 THEN 'Import' ELSE 'Export' END
    """)

def _conn_type_mapping_cols() -> Optional[Tuple[str, str]]:
    """
    Detect ID and description columns in GOLD dim_connectiontype.
    Returns (id_col, name_col) or None if not found.
    """
    S = infer_semantic_map()
    dim = S["gold"]["dim_connectiontype"]
    if not dim:
        return None
    cols = _columns(S["gold"]["catalog"], S["gold"]["schema"], dim)["column_name"].tolist()
    id_col = _best_col(cols, [r"ConnectionTypeIndicator", r"connection.*indicator", r"connection.*type.*id", r"^code$"])
    name_col = _best_col(cols, [r"Description", r"ConnectionType", r"Voltage", r"Name$"])
    if id_col and name_col:
        return (id_col, name_col)
    return None

def voltage_breakdown(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    """
    Breakdown of kWh by voltage/connection type (e.g., Unmetered, Low Voltage with CT, Low Voltage Whole Current).
    Uses silver connection-type indicator and maps to friendly names via GOLD dim_connectiontype when available.
    """
    S = infer_semantic_map()
    s = S["silver"]
    d, v, r, c = s["date_col"], s["value_col"], s["region_col"], s["conn_type_col"]
    if not (d and v) or not c:
        return pd.DataFrame(columns=["voltage_band", "kwh"])

    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    region_pred = _region_predicate(region, r)

    # if GOLD dimension exists, join to get friendly names
    if S["gold"]["dim_connectiontype"]:
        dim = S["gold"]["dim_connectiontype"]
        id_name = _conn_type_mapping_cols()
        if id_name:
            id_col, name_col = id_name
            g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{dim}"
            q = f"""
            SELECT COALESCE(g.{name_col}, s.{c}) AS voltage_band,
                   SUM(ABS(s.{v})) AS kwh
            FROM {fq} s
            LEFT JOIN {g} g
              ON s.{c} = g.{id_col}
            WHERE s.{d} >= '{start_date}' AND s.{d} <= '{end_date}' {region_pred}
            GROUP BY COALESCE(g.{name_col}, s.{c})
            ORDER BY kwh DESC
            """
            try:
                return query_df(q)
            except Exception:
                pass

    # fallback: group by the indicator in silver
    q2 = f"""
    SELECT s.{c} AS voltage_band,
           SUM(ABS(s.{v})) AS kwh
    FROM {fq} s
    WHERE s.{d} >= '{start_date}' AND s.{d} <= '{end_date}' {region_pred}
    GROUP BY s.{c}
    ORDER BY kwh DESC
    """
    return query_df(q2)

def peak_offpeak(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()
    s = S["silver"]
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, v, r, p = s["date_col"], s["value_col"], s["region_col"], s["period_col"]
    if S["gold"]["dim_halfhour"] and p:
        hh = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_halfhour']}"
        region_pred = _region_predicate(region, r)
        q = f"""
        SELECT s.{d} AS reading_date,
               SUM(CASE WHEN COALESCE(hh.IsPeak,false) THEN s.{v} ELSE 0 END) AS peak_kwh,
               SUM(CASE WHEN NOT COALESCE(hh.IsPeak,false) THEN s.{v} ELSE 0 END) AS offpeak_kwh
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
    # fallback (no split available)
    return agg_daily(start_date, end_date, region)[["reading_date"]].assign(peak_kwh=None, offpeak_kwh=None)

def distinct_regions(limit: int = 200) -> List[str]:
    """Return FRIENDLY names (via vw_mpan_consumption_consent or dim_gspgroup)."""
    S = infer_semantic_map()
    # Prefer the mapping view
    if S["gold"]["vw_region_map"]:
        vw = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['vw_region_map']}"
        try:
            df = query_df(f"""
                SELECT DISTINCT GSPGroupDescription AS region
                FROM {vw}
                WHERE GSPGroupDescription IS NOT NULL
                ORDER BY region LIMIT {limit}
            """)
            return df["region"].astype(str).tolist()
        except Exception:
            pass
    # Fallback: dim_gspgroup
    if S["gold"]["dim_gspgroup"]:
        g = f"{S['gold']['catalog']}.{S['gold']['schema']}.{S['gold']['dim_gspgroup']}"
        try:
            df = query_df(f"SELECT DISTINCT COALESCE(RegionName, GSPGroupID) AS region FROM {g} ORDER BY region LIMIT {limit}")
            return df["region"].astype(str).tolist()
        except Exception:
            pass
    # Last resort: distinct codes from silver, mapped
    s = S["silver"]; r = s["region_col"]
    if r:
        fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
        d = s["date_col"] or "1=1"
        df = query_df(f"SELECT DISTINCT {r} AS region FROM {fq} WHERE {d} IS NOT NULL ORDER BY region LIMIT {limit}")
        return _apply_friendly(df, "region")["region"].astype(str).tolist()
    return []

def totals_by_region(start_date: str, end_date: str, top_n: int = 10, region: Optional[str] = None) -> pd.DataFrame:
    df = agg_daily(start_date, end_date, region)
    if df.empty: return df
    out = df.groupby("region", as_index=False)["total_kwh"].sum().sort_values("total_kwh", ascending=False)
    return out.head(top_n) if region is None else out

def heatmap_day_period(start_date: str, end_date: str, region: Optional[str] = None) -> pd.DataFrame:
    S = infer_semantic_map()
    s = S["silver"]
    d, v, r, p = s["date_col"], s["value_col"], s["region_col"], s["period_col"]
    if not (d and v and p):
        return pd.DataFrame(columns=["reading_date","period","kwh"])
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    region_pred = _region_predicate(region, r)
    return query_df(f"""
        SELECT {d} AS reading_date, {p} AS period, SUM({v}) AS kwh
        FROM {fq}
        WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
        GROUP BY {d}, {p}
        ORDER BY reading_date, period
    """)

def distinct_mpan_count(start_date: str, end_date: str, region: Optional[str] = None) -> Optional[int]:
    S = infer_semantic_map()
    s = S["silver"]
    if not s["mpan_col"] or not s["date_col"]:
        return None
    fq = f"{s['catalog']}.{s['schema']}.{s['table']}"
    d, r, m = s["date_col"], s["region_col"], s["mpan_col"]
    region_pred = _region_predicate(region, r)
    try:
        return int(query_df(f"""
            SELECT COUNT(DISTINCT {m}) AS mpan_cnt
            FROM {fq}
            WHERE {d} >= '{start_date}' AND {d} <= '{end_date}' {region_pred}
        """)["mpan_cnt"].iloc[0])
    except Exception:
        return None

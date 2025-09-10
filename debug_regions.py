#!/usr/bin/env python3
"""
Databricks region mapping & discovery debug.

Reads env:
  DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
  GOLD_CATALOG_NAME, GOLD_SCHEMA_NAME
  SILVER_CATALOG_NAME, SILVER_SCHEMA_NAME, SILVER_TABLE_NAME
"""

import os, re, sys, argparse, datetime as dt
import pandas as pd
from databricks import sql

# ----------------- ENV -----------------
HOST = os.getenv("DATABRICKS_HOST")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")

G_CATALOG = os.getenv("GOLD_CATALOG_NAME", "sdr_gold_dev")
G_SCHEMA  = os.getenv("GOLD_SCHEMA_NAME",  "meter_data")
S_CATALOG = os.getenv("SILVER_CATALOG_NAME", "sdr_silver_dev")
S_SCHEMA  = os.getenv("SILVER_SCHEMA_NAME",  "silver_meter_data")
S_TABLE   = os.getenv("SILVER_TABLE_NAME",   "tbl_meterdata")

def mask(x, keep=4):
    if not x: return "<missing>"
    return x[:keep] + "…" + x[-keep:] if len(x) > keep*2 else "****"

def connect():
    if not (HOST and HTTP_PATH and TOKEN):
        print("❌ Missing Databricks env (HOST/HTTP_PATH/TOKEN).")
        sys.exit(1)
    return sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=TOKEN)

def qdf(q):
    with connect() as c:
        cur = c.cursor()
        cur.execute(q)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

# ----------------- helpers -----------------
def info_env():
    print("🔧 Databricks connection")
    print(f"• Host       : {HOST}")
    print(f"• HTTP path  : {HTTP_PATH}")
    print(f"• Token      : {mask(TOKEN)}")
    print(f"• Gold       : {G_CATALOG}.{G_SCHEMA}")
    print(f"• Silver     : {S_CATALOG}.{S_SCHEMA}.{S_TABLE}")
    print("-" * 80)

def discover_gold_tables():
    q = f"""
    SELECT table_name FROM system.information_schema.tables
    WHERE table_catalog='{G_CATALOG}' AND table_schema='{G_SCHEMA}'
    ORDER BY table_name
    """
    df = qdf(q)
    print(f"📚 Gold tables in {G_CATALOG}.{G_SCHEMA}: {len(df)} found")
    print(df.head(50).to_string(index=False))
    print("-" * 80)
    return df["table_name"].str.lower().tolist()

def columns_of(catalog, schema, table):
    q = f"""
    SELECT column_name, data_type
    FROM system.information_schema.columns
    WHERE table_catalog='{catalog}' AND table_schema='{schema}' AND table_name='{table}'
    ORDER BY ordinal_position
    """
    return qdf(q)

def best_col(cols, patterns):
    lower = {c.lower(): c for c in cols}
    for pat in patterns:
        r = re.compile(pat, re.I)
        for lc, orig in lower.items():
            if r.search(lc):
                return orig
    return None

def normalize_code(x: str) -> str:
    return re.sub(r"^_+", "", str(x or ""))

# ----------------- main checks -----------------
def main(limit_region=200, sample_days=14):
    info_env()

    gold_tables = discover_gold_tables()
    has_dim_gsp = "dim_gspgroup" in gold_tables

    # Preview dim_gspgroup
    code_to_name = {}
    if has_dim_gsp:
        g = f"{G_CATALOG}.{G_SCHEMA}.dim_gspgroup"
        try:
            dim = qdf(f"SELECT GSPGroupID, RegionName FROM {g} ORDER BY GSPGroupID")
            print("🗺️  dim_gspgroup (first 20):")
            print(dim.head(20).to_string(index=False))
            print("-" * 80)
            for _, r in dim.iterrows():
                raw = str(r["GSPGroupID"])
                name = str(r["RegionName"]) if pd.notna(r["RegionName"]) else raw
                norm = normalize_code(raw)
                for variant in {raw, norm, f"_{norm}"}:
                    code_to_name[variant] = name
        except Exception as e:
            print(f"⚠️  Could not read dim_gspgroup: {e}")

    # Silver columns & inference
    s_cols = columns_of(S_CATALOG, S_SCHEMA, S_TABLE)
    print("🧩 Silver columns:")
    print(s_cols.to_string(index=False))
    colnames = s_cols["column_name"].tolist()
    date_col   = best_col(colnames, [r"UTCSettlementDate", r"settlement.*date", r".*date$"])
    value_col  = best_col(colnames, [r"UTCPeriodConsumptionValue", r"kwh", r"consumption", r"^value$"])
    region_col = best_col(colnames, [r"region", r"gsp.?group.*", r"^gspgroupid$"])
    period_col = best_col(colnames, [r"settlementperiod", r"^period$"])

    print(f"\n🔎 Inferred columns → date: {date_col}, value: {value_col}, region: {region_col}, period: {period_col}")
    if not region_col:
        print("❌ Could not infer a region column from silver. Aborting.")
        sys.exit(2)

    fq_silver = f"{S_CATALOG}.{S_SCHEMA}.{S_TABLE}"

    # Distinct regions from silver (raw)
    raw_regions = qdf(f"""
        SELECT DISTINCT {region_col} AS region
        FROM {fq_silver}
        WHERE {region_col} IS NOT NULL
        ORDER BY region
        LIMIT {limit_region}
    """)
    print("\n📦 Distinct region values from silver (raw):")
    print(raw_regions.head(50).to_string(index=False))

    # Mapped names
    if code_to_name:
        mapped = raw_regions.assign(
            friendly=raw_regions["region"].astype(str).map(lambda x: code_to_name.get(x, code_to_name.get(normalize_code(x), x)))
        )
        print("\n✅ Mapping (silver value → friendly name) using dim_gspgroup:")
        print(mapped.head(50).to_string(index=False))

        unmapped = mapped[mapped["friendly"] == mapped["region"]]
        if not unmapped.empty:
            print("\n⚠️ Unmapped region codes (no friendly name found):")
            print(unmapped["region"].astype(str).head(50).to_string(index=False))
    else:
        print("\n⚠️ No dim_gspgroup available; cannot map codes to names. You will see raw codes like _A/_B.")

    # Join check (underscore-insensitive) on a recent window if date_col exists
    where_date = ""
    if date_col:
        try:
            mx = qdf(f"SELECT MAX({date_col}) AS mx FROM {fq_silver}")["mx"].iloc[0]
            if pd.notna(mx):
                end = pd.to_datetime(mx)
                start = end - pd.Timedelta(days=sample_days)
                where_date = f" AND s.{date_col} >= '{start.date()}' AND s.{date_col} <= '{end.date()}' "
                print(f"\n🕒 Sampling window: {start.date()} → {end.date()}")
        except Exception as e:
            print(f"⚠️ Could not compute date window: {e}")

    print("\n🔗 Left join silver → dim_gspgroup (underscore-insensitive):")
    if has_dim_gsp:
        g = f"{G_CATALOG}.{G_SCHEMA}.dim_gspgroup"
        join_q = f"""
        SELECT
          s.{region_col}      AS silver_code,
          COALESCE(g.RegionName, '<NO MATCH>') AS region_name,
          COUNT(1)            AS rows
        FROM {fq_silver} s
        LEFT JOIN {g} g
          ON regexp_replace(s.{region_col}, '^_+', '') = regexp_replace(g.GSPGroupID, '^_+', '')
        WHERE 1=1 {where_date}
        GROUP BY s.{region_col}, COALESCE(g.RegionName, '<NO MATCH>')
        ORDER BY rows DESC
        LIMIT 200
        """
        joined = qdf(join_q)
        print(joined.head(50).to_string(index=False))

        no_match = joined[joined["region_name"] == "<NO MATCH>"]
        if not no_match.empty:
            total = int(joined["rows"].sum())
            bad = int(no_match["rows"].sum())
            pct = 100.0 * bad / total if total else 0.0
            print(f"\n❗ Unmatched rows: {bad:,} / {total:,} ({pct:.2f}%). Example codes:")
            print(no_match.head(20).to_string(index=False))
        else:
            print("\n✅ All sampled rows matched a region name.")
    else:
        print("⚠️ dim_gspgroup not found in gold; cannot resolve friendly names at source.")

    print("\nDone.")
    print("-" * 80)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit-region", type=int, default=200, help="How many distinct regions to fetch from silver")
    p.add_argument("--sample-days", type=int, default=14, help="Days to sample for the join check")
    args = p.parse_args()
    main(limit_region=args.limit_region, sample_days=args.sample_days)

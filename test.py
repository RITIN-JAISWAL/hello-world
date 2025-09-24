# =========================
# Retail Attribute Coding EDA (Stages 1-4)
# Author intent: senior DS + retail labeling domain expert
# =========================

import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import chain

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# ---- 0) Setup ----
df = merged_df.copy()

# Helper: safe display head with many cols
def show(df_, n=5, title=None):
    if title: print(f"\n=== {title} ===")
    display(df_.head(n))

# Helper: remove accents (brand/category normalization)
def strip_accents(s: str) -> str:
    if s is None or pd.isna(s): return s
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

# Helper: simple barh plot
def barh(series, title="", top=20, figsize=(8,6)):
    s = series.dropna().head(top) if isinstance(series, pd.Series) else pd.Series(series).head(top)
    plt.figure(figsize=figsize)
    plt.barh(range(len(s))[::-1], list(s.values)[::-1])
    plt.yticks(range(len(s))[::-1], list(s.index)[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Helper: hist plot
def hist(series, bins=50, title="", xlim=None, figsize=(8,4)):
    plt.figure(figsize=figsize)
    plt.hist(series.dropna(), bins=bins)
    if xlim: plt.xlim(*xlim)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Identify attribute columns
attr_cols       = [c for c in df.columns if str(c).lower().startswith("attribute ") and "value" not in str(c).lower()]
attr_value_cols = [c for c in df.columns if str(c).lower().startswith("attribute_value ")]

# =========================
# 1) Dataset Overview
# =========================
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
show(df, 5, "Sample rows")

info_df = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "null_%": df.isna().mean().round(4)*100,
    "nunique": df.nunique(dropna=True)
}).sort_values(["null_%", "nunique"], ascending=[False, True])
show(info_df, 50, "Column profile (dtype / %null / nunique)")

print("Duplicate product_id rows:", int(df.duplicated(subset=["product_id"]).sum()))

# Null % bar
null_series = (df.isna().mean()*100).sort_values(ascending=False)
barh(null_series, "Null % by column", top=40, figsize=(10,10))

# =========================
# 2) Description, Category, Brand quality
# =========================
# Normalize description whitespace
df["desc"] = (df["1"].astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.replace("\s+", " ", regex=True)
                .str.strip())

df["desc_len"] = df["desc"].str.len()
hist(df["desc_len"], bins=60, title="Description length distribution")

# Brand cleanliness
df["brand_raw"]   = df["brand"].astype(str)
df["brand_clean"] = (df["brand_raw"].str.lower().str.strip()
                                    .map(strip_accents)
                                    .str.replace(r"[^a-z0-9 &\-\./]", "", regex=True)
                                    .str.replace(r"\s+", " ", regex=True)
                                    .str.strip())
brand_counts = df["brand_clean"].replace({"nan": np.nan}).value_counts(dropna=True)

barh(brand_counts, "Top 20 brands (cleaned)", top=20)

# Category cleanliness
if "category" in df.columns:
    df["category_clean"] = (df["category"].astype(str).str.lower().str.strip()
                             .map(strip_accents).str.replace(r"\s+", " ", regex=True))
    cat_counts = df["category_clean"].replace({"nan": np.nan}).value_counts(dropna=True)
    barh(cat_counts, "Top 20 categories (cleaned)", top=20)

# Brand duplication examples (same brand with many spellings)
brand_variants = (
    df.groupby(["brand_clean"])["brand_raw"]
      .apply(lambda s: list(pd.Series(s.unique()).head(5)))
      .reset_index()
      .rename(columns={"brand_raw":"sample_variants"})
)
show(brand_variants.head(20), title="Brand spelling variants (sample)")

# =========================
# 3) Quantity & Unit analysis (+ normalization readiness)
# =========================
# Coerce numeric Quantity and use Unit as-is; also consider qty_value/qty_unit if present
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
if "qty_value" in df.columns:
    df["qty_value"] = pd.to_numeric(df["qty_value"], errors="coerce")
if "qty_unit" in df.columns:
    df["qty_unit"] = df["qty_unit"].astype(str).str.lower()

# Unit distribution
unit_counts = df["Unit"].astype(str).str.lower().value_counts(dropna=True)
barh(unit_counts, "Unit distribution", top=15)

# Quantity stats & distribution
show(df["Quantity"].describe(percentiles=[.25,.5,.75,.95,.99]).to_frame("Quantity stats"), title="Quantity stats")
hist(df["Quantity"], bins=60, title="Quantity histogram (all)")
hist(df.loc[df["Quantity"] < 2000, "Quantity"], bins=60, title="Quantity histogram (<2000)")

# Outliers
qty_out_high = df.loc[df["Quantity"] >= 10000, ["product_id","desc","Quantity","Unit"]].head(20)
qty_out_low  = df.loc[(df["Quantity"]>0) & (df["Quantity"] < 1), ["product_id","desc","Quantity","Unit"]].head(20)
show(qty_out_high, title="Potential outliers (Quantity >= 10,000)")
show(qty_out_low,  title="Potential outliers (0 < Quantity < 1)")

# Consistency check with parsed qty_value/qty_unit when available
if {"qty_value","qty_unit"}.issubset(df.columns):
    mismatch_qty = df.loc[
        (df["qty_value"].notna()) & (df["Quantity"].notna()) &
        (np.round(df["qty_value"], 3) != np.round(df["Quantity"], 3)),
        ["product_id","desc","Quantity","Unit","qty_value","qty_unit"]
    ]
    print("Qty mismatches (Quantity vs qty_value):", len(mismatch_qty))
    show(mismatch_qty.sample(min(20, len(mismatch_qty))) if len(mismatch_qty) else mismatch_qty, title="Sample quantity mismatches")

# =========================
# 4) Attribute coverage & shape
# =========================
print(f"Detected {len(attr_cols)} attribute slots and {len(attr_value_cols)} attribute_value slots")

# Coverage (# attributes per product)
df["num_attributes"] = df[attr_cols].notna().sum(axis=1) if attr_cols else 0
hist(df["num_attributes"], bins=30, title="# Attributes per product")

# Fill % per slot (safe calculation)
if attr_cols:
    cov = (df[attr_cols].isna().sum().div(len(df)).rsub(1).mul(100)
           ).sort_values(ascending=False).round(2)
    barh(cov, "Coverage % by attribute slot", top=len(attr_cols), figsize=(7,10))

if attr_value_cols:
    cov_val = (df[attr_value_cols].isna().sum().div(len(df)).rsub(1).mul(100)
               ).sort_values(ascending=False).round(2)
    barh(cov_val, "Coverage % by attribute_value slot", top=len(attr_value_cols), figsize=(7,10))

# Most common attribute codes (A####:v) across slots
if attr_cols:
    flattened_attrs = pd.Series(list(chain.from_iterable(df[c].dropna().astype(str).tolist() for c in attr_cols)))
    flattened_attrs = flattened_attrs[flattened_attrs.str.match(r'^[A-Za-z]\d{4}(:.*)?$')]
    top_attr_codes = flattened_attrs.value_counts().head(30)
    barh(top_attr_codes, "Top attribute codes across all slots", top=30, figsize=(8,10))

# Parse attribute_value "Label:Value" into columns (left/right) for analysis
def split_attr_val(s):
    if pd.isna(s): return pd.Series([np.nan, np.nan])
    parts = str(s).split(":", 1)
    if len(parts)==1: return pd.Series([parts[0].strip(), np.nan])
    return pd.Series([parts[0].strip(), parts[1].strip()])

if attr_value_cols:
    long_rows = []
    for col in attr_value_cols:
        temp = df[[col]].copy()
        temp[["left","right"]] = temp[col].apply(split_attr_val)
        temp["slot"] = col
        long_rows.append(temp[["slot","left","right"]])
    long_df = pd.concat(long_rows, ignore_index=True)
    left_counts = long_df["left"].dropna().value_counts().head(30)
    barh(left_counts, "Top attribute_value 'left' labels (semantic attributes)", top=30, figsize=(8,10))
    show(long_df.dropna().sample(15, random_state=42), title="Sample parsed attribute_value rows")

# Size consistency checks same as before...

# =========================
# 5) Long-tail & class balance (Sector/Category/Brand)
# =========================
# (unchanged logic)

# =========================
# 6) Leakage risk checks (near-dup description signatures)
# =========================
# (unchanged logic)

# =========================
# 7) Executive Summary (auto-generated bullets)
# =========================
summary = {}

summary["rows"] = int(df.shape[0])
summary["cols"] = int(df.shape[1])

# Null % — safe calculation
n_rows = len(df)
nulls = df.isna().sum().div(n_rows).mul(100).round(2)
summary["nulliest_cols"] = nulls.sort_values(ascending=False).head(10).to_dict()

summary["unit_top"] = unit_counts.head(10).to_dict()
summary["brand_top"] = brand_counts.head(10).to_dict()
summary["brand_classes"] = int(brand_counts.size)
summary["brand_long_tail_80pct"] = headcount_to_cover(brand_counts, 0.80) if len(brand_counts) else 0
summary["quantity_outliers_ge_10000"] = int((df["Quantity"]>=10000).sum())
summary["quantity_missing_pct"] = float(df["Quantity"].isna().mean()*100)

if attr_cols:
    summary["avg_attributes_per_product"] = float(df["num_attributes"].mean())
if attr_value_cols:
    summary["attr_value_coverage_median_%"] = float((df[attr_value_cols].notna().mean()*100).median())

if "mismatch_qty" in locals():
    summary["qty_mismatch_count"] = int(len(mismatch_qty))

if len(sector_counts):
    summary["sectors_classes"] = int(sector_counts.size)
    summary["sectors_cover_80"] = headcount_to_cover(sector_counts, 0.80)
if len(category_counts):
    summary["categories_classes"] = int(category_counts.size)
    summary["categories_cover_80"] = headcount_to_cover(category_counts, 0.80)

print("\n===== EXECUTIVE SUMMARY (auto) =====")
for k,v in summary.items():
    print(f"- {k}: {v}")





























# ============================================
# Retail Product Labelling — Extended EDA
# Senior DS + Retail Domain Expert Edition
# ============================================

import pandas as pd
import numpy as np
import re, unicodedata
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# -------------------------
# 0) Setup & helpers
# -------------------------
df = merged_df.copy()

def strip_accents(s):
    if pd.isna(s): return s
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def clean_text(s):
    if pd.isna(s): return s
    s = str(s).replace("\u00A0"," ")
    s = strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9 x]*", " ", s)  # keep x for pack patterns like 6x500
    s = re.sub(r"\s+", " ", s).strip()
    return s

def barh(series, title="", top=20, figsize=(9,6)):
    s = series.dropna().head(top)
    plt.figure(figsize=figsize)
    plt.barh(range(len(s))[::-1], list(s.values)[::-1])
    plt.yticks(range(len(s))[::-1], list(s.index)[::-1])
    plt.title(title)
    plt.tight_layout(); plt.show()

def hist(series, bins=50, title="", xlim=None, figsize=(8,4)):
    plt.figure(figsize=figsize)
    plt.hist(series.dropna(), bins=bins)
    if xlim: plt.xlim(*xlim)
    plt.title(title)
    plt.tight_layout(); plt.show()

# Column guards
assert "product_id" in df.columns, "Expected column 'product_id'"
desc_col = "1" if "1" in df.columns else "description"
assert desc_col in df.columns, f"Expected description col '{desc_col}'"

# Cleaned views
df["desc"] = df[desc_col].astype(str).map(clean_text)
df["brand_clean"] = df.get("brand", pd.Series(index=df.index, dtype=object)).astype(str).map(clean_text)
df["category_clean"] = df.get("category", pd.Series(index=df.index, dtype=object)).astype(str).map(clean_text)

# Identify attribute slots
attr_cols        = [c for c in df.columns if str(c).lower().startswith("attribute ") and "value" not in str(c).lower()]
attr_value_cols  = [c for c in df.columns if str(c).lower().startswith("attribute_value ")]

print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print(f"Attribute slots: {len(attr_cols)} | Attribute_value slots: {len(attr_value_cols)}")

# -------------------------
# 1) Pack size × multiplier detection
# -------------------------
# Patterns: "6x500 ml", "500ml x 6", "pack de 6", "6 pack", "x6"
# We’ll extract: multiplier N and single-unit quantity Q, and compute N*Q for expected total.
pack_patterns = [
    r'(?P<N>\d{1,3})\s*[x×]\s*(?P<Q>\d+(?:[.,]\d+)?)\s*(?P<U>ml|l|lt|g|gr|kg)\b',
    r'(?P<Q>\d+(?:[.,]\d+)?)\s*(?P<U>ml|l|lt|g|gr|kg)\s*[x×]\s*(?P<N>\d{1,3})\b',
    r'(?:pack\s*de|pack)\s*(?P<N>\d{1,3})\b'
]
pack_regex = re.compile("|".join(pack_patterns), re.IGNORECASE)

def parse_pack(row_text):
    s = str(row_text)
    m = pack_regex.search(s)
    if not m: return pd.Series([np.nan, np.nan, np.nan, np.nan], index=["pack_n","unit_qty","unit_u","total_expected"])
    gd = m.groupdict(default="")
    # N
    try: N = int(gd.get("N") or "")
    except: N = np.nan
    # Q and U
    q_raw = gd.get("Q")
    U = gd.get("U", "").lower()
    Q = float(str(q_raw).replace(",", ".")) if q_raw else np.nan
    # Normalize to base: grams / milliliters for total_expected
    MASS = {"g":1,"gr":1,"kg":1000}
    VOL  = {"ml":1,"l":1000,"lt":1000}
    total = np.nan
    if not np.isnan(N) and not np.isnan(Q) and U:
        if U in MASS:
            total = N * Q * MASS[U]
        elif U in VOL:
            total = N * Q * VOL[U]
    return pd.Series([N, Q, U, total], index=["pack_n","unit_qty","unit_u","total_expected"])

pack_df = df["desc"].apply(parse_pack)
df = pd.concat([df, pack_df], axis=1)

# Compare with recorded Quantity/Unit if present
df["Quantity"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce")
df["Unit"] = df.get("Unit", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()

def normalize_to_base(q, u):
    MASS = {"g":1,"gr":1,"kg":1000}
    VOL  = {"ml":1,"l":1000,"lt":1000}
    if pd.isna(q) or pd.isna(u): return np.nan, ""
    u = u.lower()
    if u in MASS: return q * MASS[u], "g"
    if u in VOL:  return q * VOL[u], "ml"
    return np.nan, ""

df["qty_base"], df["unit_base"] = zip(*df.apply(lambda r: normalize_to_base(r["Quantity"], r["Unit"]), axis=1))

# Flag where we have pack info and a recorded Quantity, but totals disagree materially
pack_mismatch = df[
    df["total_expected"].notna() & df["qty_base"].notna() &
    (np.abs(df["total_expected"] - df["qty_base"]) > 1e-6)
][["product_id","desc","pack_n","unit_qty","unit_u","total_expected","Quantity","Unit","qty_base","unit_base"]]
print(f"[Pack-size] Potential pack vs recorded quantity mismatches: {len(pack_mismatch):,}")

# -------------------------
# 2) Brand × Category sanity (anomaly surfacing)
# -------------------------
# Heuristic signals: brand tokens should appear in categories rarely (and vice versa).
# We’ll list top brand-category pairs by count but with “unexpectedness”:
bc = (
    df.groupby(["brand_clean","category_clean"])
      .size().rename("cnt").reset_index()
      .sort_values("cnt", ascending=False)
)
# Show top cross pairs for review
print("\nTop brand × category pairs (manual scan for anomalies):")
display(bc.head(30))

# -------------------------
# 3) Unit × Category rule checks
# -------------------------
# Define expected units per category family (adjust to your taxonomy terms)
expected_units = {
    "bebida": {"ml","l","lt"},
    "agua": {"ml","l","lt"},
    "cerveza": {"ml","l","lt"},
    "leche": {"ml","l","lt"},
    "aceite": {"ml","l","lt"},
    "arroz": {"g","gr","kg"},
    "azucar": {"g","gr","kg"},
    "harina": {"g","gr","kg"},
    "detergente": {"ml","l","lt","g","gr","kg"},
    "pan": {"g","gr","kg","un"},
    "galleta": {"g","gr","kg","un"},
    "chocolate": {"g","gr","kg","un"},
}

def infer_cat_family(cat):
    if pd.isna(cat): return ""
    for k in expected_units:
        if k in cat:
            return k
    return ""

df["unit_lc"] = df["Unit"].astype(str).str.lower()
df["cat_family"] = df["category_clean"].map(infer_cat_family)

unit_cat_viol = df[
    (df["cat_family"] != "") &
    (df["unit_lc"].notna()) &
    (~df.apply(lambda r: r["unit_lc"] in expected_units.get(r["cat_family"], set()), axis=1))
][["product_id","desc","category","Unit","cat_family"]]

print(f"[Unit×Category] Suspect unit-category combinations: {len(unit_cat_viol):,}")

# -------------------------
# 4) Duplicate SKU detection (near-duplicate descriptions)
# -------------------------
# Sample to keep it scalable if huge
sample_n = min(8000, len(df))
dup_sample = df.sample(sample_n, random_state=42).copy()
corpus = dup_sample["desc"].fillna("")
vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
X = vect.fit_transform(corpus)

# For efficiency, compute pairwise similarities row-wise with a cutoff
def top_similarities(X, top_k=3, threshold=0.9):
    sims = cosine_similarity(X, dense_output=False)
    pairs = []
    # only upper triangle to avoid duplicates
    coo = sims.tocoo()
    for i,j,v in zip(coo.row, coo.col, coo.data):
        if i < j and v >= threshold:
            pairs.append((i,j,v))
    return pairs

pairs = top_similarities(X, threshold=0.92)
dup_rows = []
for i,j,sim in pairs[:1000]:  # cap to display
    r1 = dup_sample.iloc[i]
    r2 = dup_sample.iloc[j]
    if r1["product_id"] != r2["product_id"]:
        dup_rows.append({
            "pid_1": r1["product_id"], "desc_1": r1["desc"], "brand_1": r1["brand_clean"], "cat_1": r1["category_clean"],
            "pid_2": r2["product_id"], "desc_2": r2["desc"], "brand_2": r2["brand_clean"], "cat_2": r2["category_clean"],
            "sim": round(float(sim), 4)
        })
dup_df = pd.DataFrame(dup_rows)
print(f"[Duplicates] Highly similar description pairs (sample): {len(dup_df):,}")
display(dup_df.head(20))

# -------------------------
# 5) Attribute hierarchy consistency (Category → Sector)
# -------------------------
# We’ll try to infer (Sector, Category) pairs from attribute_value columns if present.
# Logic: find slots whose left label includes "sector" and "categ".
def split_left_right(s):
    if pd.isna(s): return pd.Series([np.nan, np.nan])
    parts = str(s).split(":", 1)
    if len(parts) == 1: return pd.Series([parts[0].strip(), np.nan])
    return pd.Series([parts[0].strip(), parts[1].strip()])

def first_attr_match(prefix):
    if not attr_value_cols: return pd.Series([pd.NA]*len(df))
    acc = pd.Series([pd.NA]*len(df), index=df.index)
    for col in attr_value_cols:
        left = df[col].str.split(":", n=1).str[0].str.lower()
        if prefix == "sector":
            mask = left.str.contains("sector", na=False)
        else:
            mask = left.str.contains("categ", na=False)
        acc = acc.fillna(df[col].where(mask))
    return acc

df["attr_sector"]   = first_attr_match("sector")
df["attr_category"] = first_attr_match("category")

# Parse out right labels (the actual values)
df[["sector_label","_"]] = df["attr_sector"].apply(split_left_right)
df[["category_label","__"]] = df["attr_category"].apply(split_left_right)

# Build empirical mapping Category -> most common Sector
cat_sec_map = (
    df.dropna(subset=["sector_label","category_label"])
      .groupby("category_label")["sector_label"]
      .agg(lambda s: s.value_counts().index[0])
      .to_dict()
)

# Flag violations where observed sector != expected sector for that category (by majority)
hier_viol = df.dropna(subset=["sector_label","category_label"]).copy()
hier_viol["expected_sector"] = hier_viol["category_label"].map(cat_sec_map)
hier_viol = hier_viol[hier_viol["expected_sector"].notna() & (hier_viol["sector_label"] != hier_viol["expected_sector"])]
hier_viol = hier_viol[["product_id","desc","category_label","sector_label","expected_sector"]]
print(f"[Hierarchy] Category→Sector inconsistencies (vs majority mapping): {len(hier_viol):,}")
display(hier_viol.head(20))

# -------------------------
# 6) Attribute coverage heatmap (sample)
# -------------------------
if attr_cols:
    # Binary presence matrix
    pres = df[attr_cols].notna().astype(int)
    # Sample to 500 rows to keep plot readable
    pres_s = pres.sample(min(500, len(pres)), random_state=42)
    # Aggregate by (optional) sector/category to show coverage by group
    # If you'd like by sector:
    if "sector_label" in df.columns and df["sector_label"].notna().any():
        cov_by_sector = (
            pd.concat([df.loc[pres_s.index, ["sector_label"]], pres_s], axis=1)
              .groupby("sector_label")
              .mean().sort_index()
        )
        plt.figure(figsize=(12, max(4, 0.4*cov_by_sector.shape[0])))
        plt.imshow(cov_by_sector, aspect="auto")
        plt.colorbar(label="Fill ratio")
        plt.yticks(range(cov_by_sector.shape[0]), cov_by_sector.index)
        plt.xticks(range(cov_by_sector.shape[1]), cov_by_sector.columns, rotation=90)
        plt.title("Attribute slot coverage by Sector (sample)")
        plt.tight_layout(); plt.show()

# -------------------------
# 7) Executive addendum (for slides)
# -------------------------
print("\n===== ADDENDUM INSIGHTS (Retail) =====")
print(f"- Pack-size mismatches (pack × unit vs recorded qty): {len(pack_mismatch):,} rows → fix before pricing/promotions.")
print(f"- Unit×Category anomalies: {len(unit_cat_viol):,} rows → indicates taxonomy mis-coding.")
print(f"- High-similarity duplicate SKU pairs (sample): {len(dup_df):,} → dedupe or merge families before training.")
print(f"- Category→Sector hierarchy violations: {len(hier_viol):,} rows → fix taxonomy edges (business rules or mapping).")
print("- Brand & Category cleaning: normalize accents/case; collapse variants (e.g., 'coca cola' vs 'coca-cola').")
print("- Recommend gating PoC on attributes with strong support (Sector, Category, Brand, Size) and clear business value.")
















































# =========================
# Imports & small helpers
# =========================
import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib.pyplot as plt
from itertools import chain
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

def strip_accents(s):
    if pd.isna(s): return s
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def hist(series, bins=60, title="", xlim=None):
    plt.figure(figsize=(10,4))
    plt.hist(series.dropna(), bins=bins)
    if xlim: plt.xlim(*xlim)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def barh(series, title="", top=None, figsize=(10,6)):
    s = series.copy()
    if top is not None:
        s = s.head(top)
    plt.figure(figsize=figsize)
    s.plot.barh()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show(obj, title=None):
    if title:
        print(f"\n=== {title} ===")
    display(obj)

def headcount_to_cover(series, pct=0.80):
    if len(series) == 0: return 0
    cum = series.cumsum() / series.sum()
    return int((cum <= pct).sum())

# -------------------------------------------------------------------
# Assumes your DataFrame is in variable `df`
# -------------------------------------------------------------------
assert isinstance(df, pd.DataFrame), "Expected your DataFrame in variable `df`"

# =========================
# 2) Description, Category, Brand quality
# =========================
# Normalize description whitespace
df["desc"] = (df["1"].astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())

df["desc_len"] = df["desc"].str.len()
hist(df["desc_len"], bins=60, title="Description length distribution")

# Brand cleanliness
df["brand_raw"]   = df["brand"].astype(str)
df["brand_clean"] = (df["brand_raw"].str.lower().str.strip()
                                    .map(strip_accents)
                                    .str.replace(r"[^a-z0-9 &\-\./]", "", regex=True)
                                    .str.replace(r"\s+", " ", regex=True)
                                    .str.strip())
brand_counts = df["brand_clean"].replace({"nan": np.nan}).value_counts(dropna=True)

barh(brand_counts, "Top 20 brands (cleaned)", top=20)

# Category cleanliness
if "category" in df.columns:
    df["category_clean"] = (df["category"].astype(str)
                            .str.lower().str.strip()
                            .map(strip_accents)
                            .str.replace(r"\s+", " ", regex=True))
    cat_counts = df["category_clean"].replace({"nan": np.nan}).value_counts(dropna=True)
    barh(cat_counts, "Top 20 categories (cleaned)", top=20)
else:
    cat_counts = pd.Series(dtype=int)

# Brand duplication examples (same brand with many spellings)
brand_variants = (
    df.groupby(["brand_clean"])["brand_raw"]
      .apply(lambda s: list(pd.Series(s.unique()).head(5)))
      .reset_index()
      .rename(columns={"brand_raw":"sample_variants"})
)
show(brand_variants.head(20), title="Brand spelling variants (sample)")

# =========================
# 3) Quantity & Unit analysis (+ normalization readiness)
# =========================
# Coerce numeric Quantity and use Unit as-is; also consider qty_value/qty_unit if present
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
if "qty_value" in df.columns:
    df["qty_value"] = pd.to_numeric(df["qty_value"], errors="coerce")
if "qty_unit" in df.columns:
    df["qty_unit"] = df["qty_unit"].astype(str).str.lower()

# Unit distribution
unit_counts = df["Unit"].astype(str).str.lower().value_counts(dropna=True)
barh(unit_counts, "Unit distribution", top=15)

# Quantity stats & distribution
show(df["Quantity"].describe(percentiles=[.25,.5,.75,.95,.99]).to_frame("Quantity stats"),
     title="Quantity stats")
hist(df["Quantity"], bins=60, title="Quantity histogram (all)")
hist(df.loc[df["Quantity"] < 2000, "Quantity"], bins=60, title="Quantity histogram (<2000)")

# Outliers
qty_out_high = df.loc[df["Quantity"] >= 10000, ["product_id","desc","Quantity","Unit"]].head(20)
qty_out_low  = df.loc[(df["Quantity"]>0) & (df["Quantity"] < 1), ["product_id","desc","Quantity","Unit"]].head(20)
show(qty_out_high, title="Potential outliers (Quantity >= 10,000)")
show(qty_out_low,  title="Potential outliers (0 < Quantity < 1)")

# Consistency check with parsed qty_value/qty_unit when available
if {"qty_value","qty_unit"}.issubset(df.columns):
    mismatch_qty = df.loc[
        (df["qty_value"].notna()) & (df["Quantity"].notna()) &
        (np.round(df["qty_value"], 3) != np.round(df["Quantity"], 3)),
        ["product_id","desc","Quantity","Unit","qty_value","qty_unit"]
    ]
    print("Qty mismatches (Quantity vs qty_value):", len(mismatch_qty))
    show(mismatch_qty.sample(min(20, len(mismatch_qty))) if len(mismatch_qty) else mismatch_qty,
         title="Sample quantity mismatches")

# =========================
# 4) Attribute coverage & shape
# =========================
attr_cols = [c for c in df.columns if str(c).lower().startswith("attribute ") and "value" not in str(c).lower()]
attr_value_cols = [c for c in df.columns if str(c).lower().startswith("attribute_value ")]

print(f"Detected {len(attr_cols)} attribute slots and {len(attr_value_cols)} attribute_value slots")

# Coverage (# attributes per product)
df["num_attributes"] = df[attr_cols].notna().sum(axis=1) if attr_cols else 0
hist(df["num_attributes"], bins=30, title="# Attributes per product")

# Fill % per slot (SAFE calculation)
n_rows = len(df)
if attr_cols:
    cov = (df[attr_cols].notna().mean().mul(100)  # filled %
           ).sort_values(ascending=False).round(2)
    barh(cov, "Coverage % by attribute slot", top=len(attr_cols), figsize=(7,10))

if attr_value_cols:
    cov_val = (df[attr_value_cols].notna().mean().mul(100)  # filled %
               ).sort_values(ascending=False).round(2)
    barh(cov_val, "Coverage % by attribute_value slot", top=len(attr_value_cols), figsize=(7,10))

# Most common attribute codes (A####:v) across slots
if attr_cols:
    flattened_attrs = pd.Series(list(chain.from_iterable(df[c].dropna().astype(str).tolist() for c in attr_cols)))
    flattened_attrs = flattened_attrs[flattened_attrs.str.match(r'^[A-Za-z]\d{4}(:.*)?$')]
    top_attr_codes = flattened_attrs.value_counts().head(30)
    barh(top_attr_codes, "Top attribute codes across all slots", top=30, figsize=(8,10))

# Parse attribute_value "Label:Value" into columns (left/right) for analysis
def split_attr_val(s):
    if pd.isna(s): return pd.Series([np.nan, np.nan])
    parts = str(s).split(":", 1)
    if len(parts)==1: return pd.Series([parts[0].strip(), np.nan])
    return pd.Series([parts[0].strip(), parts[1].strip()])

if attr_value_cols:
    long_rows = []
    for col in attr_value_cols:
        temp = df[[col]].copy()
        temp[["left","right"]] = temp[col].apply(split_attr_val)
        temp["slot"] = col
        long_rows.append(temp[["slot","left","right"]])
    long_df = pd.concat(long_rows, ignore_index=True)
    left_counts = long_df["left"].dropna().value_counts().head(30)
    barh(left_counts, "Top attribute_value 'left' labels (semantic attributes)", top=30, figsize=(8,10))
    show(long_df.dropna().sample(15, random_state=42), title="Sample parsed attribute_value rows")

# =========================
# 5) (Optional) Proxies for Sector/Category counts so summary won't error
#     If you already computed these elsewhere, feel free to remove this section.
# =========================
# Sector proxy (from attribute_value if any contains 'sector')
try:
    sector_counts = pd.Series(dtype=int)
    if attr_value_cols:
        for col in attr_value_cols:
            left = df[col].str.split(":", n=1).str[0].str.lower()
            mask = left.str.contains("sector", na=False)
            vals = df[col].where(mask).dropna().str.split(":", n=1).str[-1]
            sector_counts = pd.concat([sector_counts, vals]).value_counts()
except Exception:
    sector_counts = pd.Series(dtype=int)

# Category counts already built as cat_counts above

# =========================
# 7) Executive Summary (auto-generated bullets)
# =========================
summary = {}

summary["rows"] = int(df.shape[0])
summary["cols"] = int(df.shape[1])

# Null % — safe calculation
nulls = df.isna().sum().div(len(df)).mul(100).round(2)
summary["nulliest_cols"] = nulls.sort_values(ascending=False).head(10).to_dict()

summary["unit_top"] = unit_counts.head(10).to_dict()
summary["brand_top"] = brand_counts.head(10).to_dict()
summary["brand_classes"] = int(brand_counts.size)
summary["brand_long_tail_80pct"] = headcount_to_cover(brand_counts, 0.80) if len(brand_counts) else 0
summary["quantity_outliers_ge_10000"] = int((df["Quantity"]>=10000).sum())
summary["quantity_missing_pct"] = float(df["Quantity"].isna().mean()*100)

if len(attr_cols):
    summary["avg_attributes_per_product"] = float(df["num_attributes"].mean())
if len(attr_value_cols):
    summary["attr_value_coverage_median_%"] = float((df[attr_value_cols].notna().mean()*100).median())

if "mismatch_qty" in locals():
    summary["qty_mismatch_count"] = int(len(mismatch_qty))

if len(sector_counts):
    summary["sectors_classes"] = int(sector_counts.size)
    summary["sectors_cover_80"] = headcount_to_cover(sector_counts, 0.80)
if len(cat_counts):
    summary["categories_classes"] = int(cat_counts.size)
    summary["categories_cover_80"] = headcount_to_cover(cat_counts, 0.80)

print("\n===== EXECUTIVE SUMMARY (auto) =====")
for k,v in summary.items():
    print(f"- {k}: {v}")




























Perfect — now I’ve got a full view of your EDA results and the executive summary.
Here’s a **presentation-ready summary with key insights and important facts** for tomorrow, structured the way a senior data scientist and retail domain expert would present it:

---

# 📊 Data Quality & EDA Summary (Retail Product Labeling)

### 🔹 Dataset Overview

* **Total records (products):** 310,224
* **Total columns:** 53
* **Attribute slots:** 19 (attribute codes)
* **Attribute value slots:** 19 (semantic descriptors)

---

### 🔹 Missing Data (Nulls)

* High null presence in some fields:

  * **qty\_value & qty\_unit:** \~75% missing → quantity standardization not consistently captured.
  * **Category & Brand:** \~60% missing → indicates weak metadata assignment.
  * **Unit:** only \~2.3% missing → relatively well captured.
* **Action needed:**

  * Enforce stricter capture rules for *brand* and *category*.
  * Mandatory filling of *qty\_value/unit* fields to support consistency checks.

---

### 🔹 Brand Analysis

* **Unique brands detected:** 7,024
* Extremely **long-tail distribution**:

  * Top **153 brands cover 80%** of products.
  * Remaining \~6,800 brands have very sparse representation.
* **Common brands observed:** Rendy, Hit, Latti, Natura, Alquería, Alpina, Yanbal.
* **Issue:** >186k rows have **brand missing** → more than half the dataset!
* **Business risk:** weak brand attribution undermines search, assortment planning, and brand-level reporting.

---

### 🔹 Category Analysis

* **Unique categories detected:** 105
* **Category concentration:**

  * Top **36 categories cover 80%** of products.
* Examples of dominant categories:

  * *Papel higiénico, Snacks, Pan industrializado, Galletas, Perfumes/Fragancias, Yogurt/Kumis.*
* **Category missing:** \~60% of rows.
* **Risk:** poor categorization → misplacement in e-commerce navigation & analytics blind spots.

---

### 🔹 Quantity & Unit Consistency

* **Quantity distribution:**

  * Most products <2,000 units, but extreme outliers up to **100,000+** detected.
* **Outliers:**

  * 583 products with **Quantity ≥ 10,000** (likely packaging or entry errors).
  * \~31k products show **Quantity mismatch vs qty\_value**.
* **Unit distribution:**

  * Major units: *ml, gr, un*.
  * But noise categories like *none, “nan”* exist.
* **Critical inconsistencies:**

  * **38,969 mismatches (Gramos vs Quantity)**
  * **86,227 mismatches (Mililitros vs Quantity)**
* **Risk:** these inconsistencies will break pricing-per-unit, pack-size comparisons, and regulatory labeling.

---

### 🔹 Attributes Coverage

* Average attributes per product: **14.1**
* **Attribute coverage median:** 100% (but only for fields that exist → actual semantics often missing).
* **Frequent semantic attributes (“left” labels):** Categoría, Envase, Mililitros, Gramos, Presentación, Marca, Sector.
* Long-tail: niche attributes (Tipo de Fragancia, Tipo de Piel, Nivel Calórico) sparsely filled.
* **Risk:** incomplete attributes limit filtering (e.g., “low calorie” or “for sensitive skin” queries).

---

### 🔹 Duplicates & Data Leakage

* Near-duplicate product descriptions detected (same product listed multiple times with slight text variation).
* **Risk:** Duplicate SKUs distort demand forecasting, recommendation systems, and customer search.

---

### 🔹 Executive Summary (Key Stats)

* Rows: **310,224**
* Nulls (brand, category, qty): **60–75%** missing in critical fields.
* Brand classes: **7,024**, but **80% covered by only 153 brands**.
* Category classes: **105**, but **80% covered by only 36 categories**.
* Quantity mismatches: **\~120k total** (grams + ml inconsistencies).
* Outliers: **583 products ≥10,000 units**.
* Duplicate product\_id rows: **\~92k**

---

# 📌 Recommendations

1. **Data Capture Standards**

   * Make *brand* and *category* mandatory fields.
   * Standardize *qty\_value + qty\_unit* for all new entries.
2. **Brand & Category Master Alignment**

   * Consolidate brand variants (e.g., "Colgate" vs "colgate") into master brand list.
   * Map missing categories via NLP-based description parsing.
3. **Quantity Normalization**

   * Create business rules to cap or flag outliers (>10,000).
   * Reconcile **qty\_value vs Quantity vs Unit** automatically.
4. **Attribute Enrichment**

   * Drive attribute completion for *Sector, Marca, Envase, Tamaño*.
   * Use ML/NLP to infer missing attributes from descriptions.
5. **Duplicate Resolution**

   * De-duplicate using product\_id + normalized description.
   * Ensure each SKU is uniquely represented in catalog.

---

👉 This narrative positions your EDA findings in **business terms** (impact on e-commerce, assortment, pricing, reporting) — not just stats.

Would you like me to also **draft presentation slides (PowerPoint-style with visuals + bullet points)** summarizing these findings, so you can walk through them tomorrow?
























# =============================
# STAGES 5 → 8: Labeling POC
# =============================
import pandas as pd
import numpy as np
import re
import os
import io
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import sparse
import joblib

# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def drop_near_dups(df, text_col="desc", pid_col="product_id"):
    # very light de-dupe: normalize text and keep first instance
    norm = df[text_col].astype(str).str.lower().str.replace(r"[^a-z0-9 ]"," ", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()
    deduped = df.loc[~norm.duplicated(keep="first")].copy()
    return deduped

def topk(series: pd.Series, k=10):
    return series.value_counts().head(k)

def filter_min_support(df, y_col, min_count=200):
    vc = df[y_col].value_counts()
    keep = vc[vc >= min_count].index
    return df[df[y_col].isin(keep)].copy(), keep

def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# ------------------------------------------------------------------
# Stage 5: Target curation & split
# ------------------------------------------------------------------
# Build targets
df_stage = df.copy()

# Targets (choose one label source each; tweak if you want different proxies)
df_stage["y_brand"]    = df_stage["brand_clean"].replace({"nan": np.nan})
df_stage["y_category"] = df_stage["y_category_raw"].fillna(df_stage.get("category_clean"))
# y_sector_raw looks like "Sector: Bebidas" or similar → keep right side if present
def right_after_colon(x):
    if pd.isna(x): return np.nan
    parts = str(x).split(":", 1)
    return parts[-1].strip() if len(parts) == 2 else str(x).strip()

df_stage["y_sector"]   = df_stage["y_sector_raw"].apply(right_after_colon)

# Minimal text cleaning
df_stage["desc"] = df_stage["desc"].apply(clean_text)

# Optional: de-duplicate by normalized desc to avoid leakage
df_stage = drop_near_dups(df_stage, text_col="desc", pid_col="product_id")

# Choose which targets to train in this POC
TARGETS = {
    "sector":   "y_sector",
    "category": "y_category",
    "brand":    "y_brand",
}

# Stage 5 – per-target filtering, label encoding, and stratified split
MIN_SUPPORT = {
    "sector":   200,   # tune by coverage
    "category": 300,
    "brand":    500,
}

splits = {}   # store all objects per target

for name, ycol in TARGETS.items():
    print_header(f"Stage 5: Preparing target = {name} ({ycol})")
    sub = df_stage.loc[df_stage[ycol].notna(), ["product_id","desc","Quantity","Unit", ycol]].copy()
    # quantity to numeric
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce")
    sub["Unit"] = sub["Unit"].astype(str).str.lower()

    # filter low-support classes
    sub_filt, kept_labels = filter_min_support(sub, ycol, min_count=MIN_SUPPORT[name])
    print(f"Kept {len(kept_labels)} classes for {name} (min_count={MIN_SUPPORT[name]}). "
          f"Dropped {(sub[ycol].nunique()-len(kept_labels))} rare classes.")

    # label encode
    le = LabelEncoder()
    y = le.fit_transform(sub_filt[ycol])
    X = sub_filt.drop(columns=[ycol])

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    splits[name] = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "label_encoder": le,
        "ycol": ycol
    }
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------------
# Stage 6: Features (TF-IDF desc + scaled Quantity + one-hot Unit)
# ------------------------------------------------------------------
# Column selectors
TEXT_COL = "desc"
NUM_COLS = ["Quantity"]
CAT_COLS = ["Unit"]

text_vect = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9,
    strip_accents="unicode",
    sublinear_tf=True
)

preprocess = ColumnTransformer(
    transformers=[
        ("tfidf", text_vect, TEXT_COL),
        ("num",   Pipeline([("scaler", StandardScaler(with_mean=False))]), NUM_COLS),
        ("unit",  OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

# Two alternative classifiers; pick one or evaluate both
clf_logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
    C=2.0,
    solver="saga"
)

clf_linsvc = LinearSVC(
    class_weight="balanced",
    C=1.0
)

def build_pipeline(base_clf="svm"):
    clf = clf_linsvc if base_clf == "svm" else clf_logreg
    return Pipeline([
        ("prep", preprocess),
        ("clf",  clf)
    ])

# ------------------------------------------------------------------
# Stage 7: Train + evaluate (CV & holdout)
# ------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support

def evaluate_target(target_name, base_clf="svm", cv_folds=5, prob=False):
    print_header(f"Stage 7: Train & Evaluate — target={target_name}, model={base_clf}")
    pack = splits[target_name]
    X_train, X_test = pack["X_train"], pack["X_test"]
    y_train, y_test = pack["y_train"], pack["y_test"]
    le = pack["label_encoder"]

    pipe = build_pipeline(base_clf=base_clf)

    # CV on train
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1)
    print(f"CV weighted-F1 ({cv_folds} folds): mean={cv_f1.mean():.4f}  std={cv_f1.std():.4f}")

    # Fit on full train
    pipe.fit(X_train, y_train)

    # Holdout test
    y_pred = pipe.predict(X_test)
    print("\n-- HOLDOUT RESULTS --")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Macro F1:    {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nTop confusion pairs (most confused labels):")
    cm = confusion_matrix(y_test, y_pred)
    # show a few biggest off-diagonal errors
    errs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i,j] > 0:
                errs.append((cm[i,j], le.classes_[i], le.classes_[j]))
    errs = sorted(errs, reverse=True)[:10]
    for c, a, b in errs:
        print(f"{c:5d}  {a}  →  {b}")

    # Build predictions frame with confidence (if available)
    Xh = X_test.copy()
    if hasattr(pipe.named_steps["clf"], "decision_function"):
        dec = pipe.decision_function(X_test)
        if dec.ndim == 1:  # binary
            conf = np.abs(dec)
            top_idx = (dec > 0).astype(int)
        else:
            top_idx = dec.argmax(axis=1)
            conf = dec.max(axis=1)
    elif hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)
        top_idx = proba.argmax(axis=1)
        conf = proba.max(axis=1)
    else:
        top_idx = y_pred
        conf = np.ones_like(y_pred, dtype=float)

    pred_labels = le.inverse_transform(top_idx)
    true_labels = le.inverse_transform(y_test)

    out = Xh[["product_id"]].copy()
    out["true"] = true_labels
    out["pred"] = pred_labels
    out["confidence"] = conf
    out["correct"] = (out["true"] == out["pred"]).astype(int)

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"pipeline": pipe, "label_encoder": le}, f"artifacts/{target_name}_model.pkl")
    out.to_csv(f"artifacts/{target_name}_predictions.csv", index=False)

    # hardest errors for SME review
    err_df = out.loc[out["correct"] == 0].sort_values("confidence", ascending=False)
    err_join = err_df.merge(X_test[["product_id","desc","Quantity","Unit"]], on="product_id", how="left")
    err_join.head(500).to_csv(f"artifacts/{target_name}_errors_top500.csv", index=False)

    print(f"\nSaved: artifacts/{target_name}_model.pkl, "
          f"artifacts/{target_name}_predictions.csv, "
          f"artifacts/{target_name}_errors_top500.csv")

    return {"preds": out, "pipe": pipe, "label_encoder": le}

# Train & evaluate the three targets (you can switch to "logreg" to compare)
res_sector   = evaluate_target("sector",   base_clf="svm", cv_folds=5)
res_category = evaluate_target("category", base_clf="svm", cv_folds=5)
res_brand    = evaluate_target("brand",    base_clf="svm", cv_folds=5)

# ------------------------------------------------------------------
# Stage 8: Inference wrapper (top-k & thresholding)
# ------------------------------------------------------------------
def predict_with_threshold(model_pack, df_in, k=3, threshold=None):
    """
    Returns top-k predictions with scores; if threshold is set,
    labels below threshold are replaced with 'OTHER'.
    """
    pipe = model_pack["pipeline"]
    le   = model_pack["label_encoder"]

    # minimal preproc for new data
    data = df_in.copy()
    data["desc"] = data["desc"].apply(clean_text)
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Unit"] = data["Unit"].astype(str).str.lower()

    # decision scores or proba
    if hasattr(pipe.named_steps["clf"], "decision_function"):
        D = pipe.decision_function(data)
        if D.ndim == 1:
            D = np.vstack([-D, D]).T  # binary compat
        scores = np.sort(D, axis=1)[:, ::-1]
        idxs   = np.argsort(D, axis=1)[:, ::-1]
    elif hasattr(pipe.named_steps["clf"], "predict_proba"):
        P = pipe.predict_proba(data)
        scores = np.sort(P, axis=1)[:, ::-1]
        idxs   = np.argsort(P, axis=1)[:, ::-1]
    else:
        # fall back to hard predictions
        yhat = pipe.predict(data)
        idxs = np.vstack([yhat]).T
        scores = np.ones((len(yhat), 1))

    topk_labels = np.column_stack([le.classes_[idxs[:,i]] for i in range(min(k, idxs.shape[1]))])
    topk_scores = scores[:, :min(k, scores.shape[1])]

    # apply threshold → 'OTHER'
    if threshold is not None:
        top1 = topk_labels[:,0].astype(object)
        top1_score = topk_scores[:,0]
        top1[top1_score < threshold] = "OTHER"
        topk_labels[:,0] = top1

    return topk_labels, topk_scores

# Example inference on the category model
cat_pack = joblib.load("artifacts/category_model.pkl")
topk_labels, topk_scores = predict_with_threshold(cat_pack, splits["category"]["X_test"], k=3, threshold=None)

pred_df = pd.DataFrame({
    "product_id": splits["category"]["X_test"]["product_id"].values,
    "pred_top1": topk_labels[:,0],
    "score_top1": topk_scores[:,0]
})
pred_df.head()

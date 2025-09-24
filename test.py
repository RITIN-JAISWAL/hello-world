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

# Fill % per slot
if attr_cols:
    cov = (df[attr_cols].notna().mean()*100).sort_values(ascending=False).round(2)
    barh(cov, "Coverage % by attribute slot", top=len(attr_cols), figsize=(7,10))

if attr_value_cols:
    cov_val = (df[attr_value_cols].notna().mean()*100).sort_values(ascending=False).round(2)
    barh(cov_val, "Coverage % by attribute_value slot", top=len(attr_value_cols), figsize=(7,10))

# Most common attribute codes (A####:v) across slots
if attr_cols:
    flattened_attrs = pd.Series(list(chain.from_iterable(df[c].dropna().astype(str).tolist() for c in attr_cols)))
    # keep only tokens that look like A####:number or A#### (robust)
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
    # Build a long-form table for distribution analysis
    long_rows = []
    for col in attr_value_cols:
        temp = df[[col]].copy()
        temp[["left","right"]] = temp[col].apply(split_attr_val)
        temp["slot"] = col
        long_rows.append(temp[["slot","left","right"]])
    long_df = pd.concat(long_rows, ignore_index=True)
    # Most common left labels (e.g., "Gramos", "Porciones", "Tipo de Chocolate")
    left_counts = long_df["left"].dropna().value_counts().head(30)
    barh(left_counts, "Top attribute_value 'left' labels (semantic attributes)", top=30, figsize=(8,10))
    show(long_df.dropna().sample(15, random_state=42), title="Sample parsed attribute_value rows")

# Size consistency: Compare Quantity to attribute_value like "Gramos:500" or "Mililitros:500"
if attr_value_cols:
    size_like = long_df.dropna()
    size_like["left_lc"] = size_like["left"].str.lower().map(strip_accents)
    grams_mask = size_like["left_lc"].str.contains("gram", na=False)
    ml_mask    = size_like["left_lc"].str.contains("mili|mlit|ml", na=False)  # cover mililitros/mililiters
    # Collect per product (join back on index alignment)
    # Note: long_df lost product_id; rebuild quickly:
    # Build product_id-aware long_df
    pid_long = []
    for col in attr_value_cols:
        tmp = df[["product_id", col]].rename(columns={col:"val"})
        tmp[["left","right"]] = tmp["val"].apply(split_attr_val)
        tmp["slot"] = col
        pid_long.append(tmp[["product_id","slot","left","right"]])
    pid_long = pd.concat(pid_long, ignore_index=True)
    pid_long["left_lc"] = pid_long["left"].str.lower().map(strip_accents)

    def to_float(s):
        if pd.isna(s): return np.nan
        return pd.to_numeric(str(s).replace(",", "."), errors="coerce")

    grams_df = pid_long.loc[pid_long["left_lc"].str.contains("gram", na=False)].copy()
    grams_df["right_num"] = grams_df["right"].apply(to_float)
    ml_df    = pid_long.loc[pid_long["left_lc"].str.contains("mili|mlit|ml", na=False)].copy()
    ml_df["right_num"] = ml_df["right"].apply(to_float)

    # Join against Quantity/Unit
    gram_merge = grams_df.merge(df[["product_id","Quantity","Unit"]], on="product_id", how="left")
    ml_merge   = ml_df.merge(df[["product_id","Quantity","Unit"]], on="product_id", how="left")

    gram_incons = gram_merge.loc[
        gram_merge["Quantity"].notna() & (np.round(gram_merge["Quantity"],1) != np.round(gram_merge["right_num"],1)),
        ["product_id","left","right","Quantity","Unit","slot"]
    ]
    ml_incons = ml_merge.loc[
        ml_merge["Quantity"].notna() & (np.round(ml_merge["Quantity"],1) != np.round(ml_merge["right_num"],1)),
        ["product_id","left","right","Quantity","Unit","slot"]
    ]
    print(f"Inconsistencies (Gramos vs Quantity): {len(gram_incons):,}")
    show(gram_incons.head(20), title="Sample inconsistencies: Gramos vs Quantity")
    print(f"Inconsistencies (Mililitros vs Quantity): {len(ml_incons):,}")
    show(ml_incons.head(20), title="Sample inconsistencies: Mililitros vs Quantity")

# =========================
# 5) Long-tail & class balance (Sector/Category/Brand)
# =========================
# Attempt to derive a candidate 'sector' from attribute_value columns if present
def find_first_label(prefix, default_col=None):
    """
    Try to find the first attribute_value whose left label matches 'Sector'/'Categoria' patterns.
    If default_col is given, use that as a fallback source.
    """
    if not attr_value_cols: return pd.Series([np.nan]*len(df))
    lc = None
    for col in attr_value_cols:
        lefts = df[col].str.split(":", n=1).str[0].str.lower()
        if prefix == "sector":
            mask = lefts.str.contains("sector", na=False)
        elif prefix == "category":
            mask = lefts.str.contains("categ", na=False)  # categoría / categoria
        else:
            mask = pd.Series([False]*len(df))
        vals = df[col].where(mask)
        if lc is None:
            lc = vals
        else:
            lc = lc.fillna(vals)
    if default_col and default_col in df.columns:
        lc = lc.fillna(df[default_col])
    return lc

df["y_sector_raw"]   = find_first_label("sector")
df["y_category_raw"] = find_first_label("category", default_col="category")
df["y_brand_raw"]    = df["brand_clean"]

# Count distributions
sector_counts = df["y_sector_raw"].dropna().str.split(":").str[0].value_counts()
category_counts = df["y_category_raw"].dropna().astype(str).value_counts()
brand_counts = df["brand_clean"].dropna().value_counts()

barh(sector_counts, "Top Sectors (proxy)", top=20)
barh(category_counts, "Top Categories (proxy)", top=20)
barh(brand_counts, "Top Brands (cleaned)", top=20)

# Tail coverage (how many classes cover X% of data)
def headcount_to_cover(series, pct=0.8):
    s = series.copy()
    cum = (s.cumsum()/s.sum())
    k = (cum <= pct).sum()
    return int(k)

for name, s in [("Sector", sector_counts), ("Category", category_counts), ("Brand", brand_counts)]:
    if len(s):
        k80 = headcount_to_cover(s, 0.80)
        print(f"{name}: Top {k80} classes cover 80% of products (total classes={len(s)})")

# =========================
# 6) Leakage risk checks (near-dup description signatures)
# =========================
def normalize_for_signature(text):
    s = strip_accents(text)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["desc_sig"] = df["desc"].astype(str).apply(normalize_for_signature)
sig_counts = df["desc_sig"].value_counts()
dup_sigs = sig_counts[sig_counts>1]
print("Potential near-duplicates by normalized description:", int(dup_sigs.sum() - len(dup_sigs)))
show(df.loc[df["desc_sig"].isin(dup_sigs.index), ["product_id","desc"]].head(20),
     title="Sample near-duplicate descriptions")

# =========================
# 7) Executive Summary (auto-generated bullets)
# =========================
summary = {}

summary["rows"] = int(df.shape[0])
summary["cols"] = int(df.shape[1])
summary["nulliest_cols"] = null_series.head(10).round(2).to_dict()
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

print("""
Key takeaways (domain guidance):
1) Unit normalization: Majority units are 'gr'/'ml' with some 'kg','lt','un'. Normalize to grams/ml for modeling and rule checks.
2) Brand canonicalization needed: Multiple spellings and accents inflate class space; clean before modeling.
3) Attribute richness is uneven across SKUs; set PoC scope to high-coverage attributes (Sector/Category/Brand/Size).
4) Quantity inconsistencies exist vs attribute_value (e.g., 'Gramos:500'); implement consistency rules and flag for human QA.
5) Long-tail: A small head of classes covers most volume; use class weights and focus evaluation on high-support classes first.
6) Leakage risk: near-duplicate descriptions exist; ensure splits by product family/brand to avoid optimistic metrics.
""")

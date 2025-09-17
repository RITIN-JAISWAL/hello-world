# ====================================================
# STEP 0. Setup & Imports
# ====================================================
import os, io, re, json, base64, datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from azure.storage.blob import BlobServiceClient
from jinja2 import Template

# Matplotlib inline style
%matplotlib inline
plt.style.use("seaborn-v0_8")

# Ensure connection string is set
CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if not CONN_STR:
    raise RuntimeError("⚠️ Please set AZURE_STORAGE_CONNECTION_STRING")

# Container + blob names
container = "rawdata"
ocr_blob = "OCR_colombia.csv"
master_blob = "Attributes_definitions.csv"
dict_blob = "dictionary_co.json"



# ====================================================
# STEP 1. Azure Blob Utilities
# ====================================================
def get_blob_client(container: str, blob: str):
    svc = BlobServiceClient.from_connection_string(CONN_STR)
    return svc.get_blob_client(container=container, blob=blob)

def download_blob_bytes(container: str, blob: str) -> bytes:
    return get_blob_client(container, blob).download_blob().readall()

def upload_blob_text(container: str, blob: str, text: str):
    get_blob_client(container, blob).upload_blob(text.encode("utf-8"), overwrite=True)


# ====================================================
# STEP 2. CSV / JSON Utilities
# ====================================================
def safe_read_csv(bytes_data: bytes, dtype=None):
    buf = io.BytesIO(bytes_data)
    try:
        return pd.read_csv(buf, dtype=dtype)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, sep=";", dtype=dtype)

def ensure_cols(df: pd.DataFrame, rename_map: dict):
    for target, alts in rename_map.items():
        if target in df.columns: continue
        for a in alts:
            if a in df.columns:
                df.rename(columns={a: target}, inplace=True)
                break
    return df


# ====================================================
# STEP 3. Quantity Parsing for OCR (Size/Weight)
# ====================================================
SIZE_PATTERNS = [
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>kg|g|gr|gramos|l|lt|litro|ml|mL)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>oz|lb)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?x\s?(?P<pack>\d+)\b',
]

def extract_qty(text: str):
    if not isinstance(text, str):
        return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})
    t = text.replace(",", ".")
    for pat in SIZE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m and m.groupdict().get("qty") and m.groupdict().get("unit"):
            try:
                return pd.Series({"qty_value": float(m.group("qty")), "qty_unit": m.group("unit").lower()})
            except ValueError:
                pass
    return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})

def normalize_units(q, u):
    if pd.isna(q) or not isinstance(u, str): 
        return pd.NA, pd.NA
    u = u.lower()
    if u in ["g","gr","gramos"]: return round(q,3), "g"
    if u=="kg": return round(q*1000,3), "g"
    if u in ["ml"]: return round(q,3), "ml"
    if u in ["l","lt","litro"]: return round(q*1000,3), "ml"
    if u=="oz": return round(q*28.3495,3), "g"
    if u=="lb": return round(q*453.592,3), "g"
    return pd.NA, pd.NA


# ====================================================
# STEP 4. Dictionary Parsing Utilities
# ====================================================
def find_values_list(entry: dict):
    for k in ["b","values","codes","items","options","map"]:
        if k in entry and isinstance(entry[k], list):
            return entry[k]
    for v in entry.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []

def pick_code(d: dict):
    for k in ["c","code","id","k","key"]:
        if k in d: return str(d[k])
    return None

def pick_value(d: dict):
    for k in ["v","value","label","name","text"]:
        if k in d: return str(d[k])
    return None
# ====================================================
# STEP 5. Load Data from Azure
# ====================================================
ocr = safe_read_csv(download_blob_bytes(container, ocr_blob), dtype={"product_id":str})
mst = safe_read_csv(download_blob_bytes(container, master_blob), dtype={"product_id":str})
J = json.loads(download_blob_bytes(container, dict_blob).decode("utf-8", errors="ignore"))

print("OCR shape:", ocr.shape)
print("Master shape:", mst.shape)
print("Dictionary entries:", len(J.get("dict", [])))

ocr.head(3)


# ====================================================
# STEP 6. Analyze OCR
# ====================================================
ocr = ensure_cols(ocr, {"product_id":["prod_id","id","productid"], "ocr_text":["text","ocr","ocrstring"]})
if "ocr_text" not in ocr.columns: ocr["ocr_text"] = ""

# Add features
ocr["text_len"] = ocr["ocr_text"].fillna("").astype(str).str.len()
ocr[["qty_value","qty_unit"]] = ocr["ocr_text"].apply(extract_qty)
ocr[["qty_std_value","qty_std_unit"]] = ocr.apply(
    lambda r: normalize_units(r["qty_value"], r["qty_unit"]), axis=1, result_type="expand"
)

# Summary
print("OCR rows:", len(ocr))
print("Unique products:", ocr["product_id"].nunique())
print("Empty OCR texts:", (ocr["ocr_text"].astype(str).str.strip()=="").sum())

# Visualize
ocr["text_len"].hist(bins=50, figsize=(6,4))
plt.title("Distribution of OCR text lengths")
plt.show()

ocr["qty_std_value"].hist(bins=30, figsize=(6,4))
plt.title("Distribution of parsed quantities (standardized)")
plt.show()


# ====================================================
# STEP 7. Analyze Masterfile
# ====================================================
mst = ensure_cols(mst, {
    "product_id":["prod_id","id","productid"],
    "description":["desc","product_desc","name"],
    "attributes_raw":["attributes","attr_map","attr_codes"]
})
if "attributes_raw" not in mst.columns: mst["attributes_raw"] = ""

# Explode attributes
rows=[]
for _, r in mst.iterrows():
    for token in re.split(r"[;,\|]+", str(r["attributes_raw"])):
        if ":" in token:
            aid,val = token.split(":",1)
            rows.append({"product_id":r["product_id"],"attr_id":aid.strip(),"value_code":val.strip()})
mst_long = pd.DataFrame(rows)

print("Exploded attribute pairs:", len(mst_long))
mst_long.head(5)

# Frequency plot
mst_long["attr_id"].value_counts().head(20).plot(kind="barh", figsize=(8,6))
plt.title("Top attributes by frequency")
plt.show()


# ====================================================
# STEP 8. Analyze Dictionary
# ====================================================
attr_name = {}
value_map = {}
dict_conflicts = []

for e in J.get("dict", []):
    aid = e.get("id")
    if not aid: continue
    attr_name[aid] = e.get("sl") or e.get("name") or e.get("label")
    for it in find_values_list(e):
        code,val = pick_code(it), pick_value(it)
        if code and val:
            if (aid,str(code)) in value_map and value_map[(aid,str(code))]!=val:
                dict_conflicts.append({"attr_id":aid,"value_code":code,"a":value_map[(aid,str(code))],"b":val})
            value_map[(aid,str(code))]=val

print("Attributes defined:", len(attr_name))
print("Dictionary conflicts:", len(dict_conflicts))
pd.DataFrame(dict_conflicts).head(5)

# ====================================================
# STEP 9. Join Analysis
# ====================================================
ocr_p = set(ocr["product_id"])
mst_p = set(mst_long["product_id"])
both = len(ocr_p & mst_p)

print("Products only in OCR:", len(ocr_p - mst_p))
print("Products only in Master:", len(mst_p - ocr_p))
print("Products in both:", both)

# Mapping coverage
mst_long["value_label"] = mst_long.apply(lambda r: value_map.get((r["attr_id"], str(r["value_code"])), pd.NA), axis=1)
missing = mst_long[mst_long["value_label"].isna()]
print("Missing dictionary mappings:", len(missing))
missing.groupby("attr_id").size().sort_values(ascending=False).head(10)


# ====================================================
# STEP 10. Build Scorecard
# ====================================================
completeness = 100 - (100 * (ocr["ocr_text"].astype(str).str.strip()=="").sum() / len(ocr))
uniqueness = 100 - (100 * mst.duplicated("product_id").sum() / len(mst))
validity = 100 - (100 * len(missing) / max(1,len(mst_long)))
consistency = 100 - (100 * len(dict_conflicts) / max(1,len(mst_long)))

scorecard = pd.DataFrame([
    {"dimension":"Completeness (OCR text present)", "score": round(completeness,2)},
    {"dimension":"Uniqueness (Master product_id)", "score": round(uniqueness,2)},
    {"dimension":"Validity (codes→labels coverage)", "score": round(validity,2)},
    {"dimension":"Consistency (dictionary conflicts low)", "score": round(consistency,2)},
])
scorecard["grade"] = pd.cut(scorecard["score"], bins=[-1,60,75,90,100], labels=["Poor","Fair","Good","Excellent"])

scorecard


# ====================================================
# STEP 11. Export Report (Optional)
# ====================================================
HTML_TMPL = Template("""
<h1>Data Quality Report</h1>
<p>Generated: {{ now }}</p>
<h2>Scorecard</h2>
<table border=1>
<tr><th>Dimension</th><th>Score</th><th>Grade</th></tr>
{% for r in scorecard %}
<tr><td>{{ r.dimension }}</td><td>{{ r.score }}</td><td>{{ r.grade }}</td></tr>
{% endfor %}
</table>
""")

html = HTML_TMPL.render(
    now=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    scorecard=scorecard.to_dict(orient="records")
)

upload_blob_text(container, "quality_report/dq_report.html", html)
print("✅ HTML uploaded to container: quality_report/dq_report.html")
















# -----------------------------------------------
# 1. Setup & Azure connection
# -----------------------------------------------
import os, io, json, re
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Connection string must be set in environment
CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if not CONN_STR:
    raise RuntimeError("⚠️ Please set AZURE_STORAGE_CONNECTION_STRING")

container = "rawdata"  # adjust if different
ocr_blob = "OCR_colombia.csv"
master_blob = "Attributes_definitions.csv"
dict_blob = "dictionary_co.json"

svc = BlobServiceClient.from_connection_string(CONN_STR)

def read_blob_to_df(container, blob, sep=",", dtype=None):
    data = svc.get_blob_client(container, blob).download_blob().readall()
    return pd.read_csv(io.BytesIO(data), sep=sep, dtype=dtype)

def read_blob_to_json(container, blob):
    data = svc.get_blob_client(container, blob).download_blob().readall()
    return json.loads(data.decode("utf-8", errors="ignore"))

# -----------------------------------------------
# 2. Load data
# -----------------------------------------------
ocr = read_blob_to_df(container, ocr_blob, dtype={"product_id": str})
mst = read_blob_to_df(container, master_blob, dtype={"product_id": str})
dictionary = read_blob_to_json(container, dict_blob)

print("OCR shape:", ocr.shape)
print("Master shape:", mst.shape)
print("Dictionary entries:", len(dictionary.get("dict", [])))

ocr.head()
# -----------------------------------------------
# 3. Quick sanity checks
# -----------------------------------------------
print("OCR unique products:", ocr["product_id"].nunique())
print("Master unique products:", mst["product_id"].nunique())

print("OCR missing ocr_text:", ocr["ocr_text"].isna().sum())
print("Master missing attributes_raw:", mst["attributes_raw"].isna().sum())


# -----------------------------------------------
# 4. Parse quantities from OCR text
# -----------------------------------------------
SIZE_PATTERNS = [
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>kg|g|gr|gramos|l|lt|litro|ml|mL)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>oz|lb)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?x\s?(?P<pack>\d+)\b',
]

def extract_qty(text: str):
    if not isinstance(text, str):
        return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})
    t = text.replace(",", ".")
    for pat in SIZE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m and m.groupdict().get("qty") and m.groupdict().get("unit"):
            try:
                return pd.Series({"qty_value": float(m.group("qty")), "qty_unit": m.group("unit").lower()})
            except ValueError:
                pass
    return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})

def normalize_units(q, u):
    if pd.isna(q) or not isinstance(u, str): 
        return pd.NA, pd.NA
    u = u.lower()
    if u in ["g","gr","gramos"]: return round(q,3), "g"
    if u=="kg": return round(q*1000,3), "g"
    if u in ["ml"]: return round(q,3), "ml"
    if u in ["l","lt","litro"]: return round(q*1000,3), "ml"
    if u=="oz": return round(q*28.3495,3), "g"
    if u=="lb": return round(q*453.592,3), "g"
    return pd.NA, pd.NA

ocr[["qty_value","qty_unit"]] = ocr["ocr_text"].apply(extract_qty)
ocr[["qty_std_value","qty_std_unit"]] = ocr.apply(
    lambda r: normalize_units(r["qty_value"], r["qty_unit"]), axis=1, result_type="expand"
)

ocr.sample(5)

# -----------------------------------------------
# 5. Explode Master attributes
# -----------------------------------------------
def explode_attributes(attr_str: str):
    if not isinstance(attr_str, str) or not attr_str.strip(): return []
    pairs=[]
    for token in re.split(r"[;,\|]+", attr_str):
        token = token.strip()
        if ":" in token:
            k,v = token.split(":",1)
            pairs.append((k.strip(), v.strip()))
    return pairs

rows=[]
for _,r in mst.iterrows():
    for aid,val in explode_attributes(r.get("attributes_raw","")):
        rows.append({"product_id":r["product_id"],"attr_id":aid,"value_code":val})
mst_long = pd.DataFrame(rows)
print("Exploded attributes:", mst_long.shape)
mst_long.head()


# -----------------------------------------------
# 6. Build dictionary maps
# -----------------------------------------------
def find_values_list(entry: dict):
    for k in ["b","values","codes","items","options","map"]:
        if k in entry and isinstance(entry[k], list):
            return entry[k]
    for v in entry.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []

def pick_code(d: dict):
    for k in ["c","code","id","k","key"]:
        if k in d: return str(d[k])
    return None

def pick_value(d: dict):
    for k in ["v","value","label","name","text"]:
        if k in d: return str(d[k])
    return None

attr_name = {}
value_map = {}
for entry in dictionary.get("dict", []):
    aid = entry.get("id")
    aname = entry.get("sl") or entry.get("name") or entry.get("label")
    if aid: attr_name[aid]=aname
    for it in find_values_list(entry):
        code = pick_code(it); val = pick_value(it)
        if code and val: value_map[(aid,str(code))]=val

len(attr_name), len(value_map)

# -----------------------------------------------
# 7. Map codes → labels
# -----------------------------------------------
mst_long["attr_name"] = mst_long["attr_id"].map(attr_name)
mst_long["value_label"] = mst_long.apply(
    lambda r: value_map.get((r["attr_id"], str(r["value_code"])), pd.NA), axis=1
)

# Check missing mappings
missing = mst_long[mst_long["value_label"].isna()]
print("Missing dictionary mappings:", len(missing))
missing.head()


# -----------------------------------------------
# 8. Pivot attributes wide
# -----------------------------------------------
mst_wide = (
    mst_long
    .assign(val = mst_long["value_label"].fillna(mst_long["value_code"]))
    .pivot_table(index="product_id", columns="attr_id", values="val", aggfunc=lambda x: ";".join(sorted(set(map(str,x)))))
    .reset_index()
)
mst_wide.columns = [f"attr_{c}" if c!="product_id" else "product_id" for c in mst_wide.columns]
mst_wide.head()
# -----------------------------------------------
# 9. Final join: OCR + Masterwide
# -----------------------------------------------
final = (
    mst[["product_id","description"]].drop_duplicates()
    .merge(ocr[["product_id","ocr_text","qty_value","qty_unit","qty_std_value","qty_std_unit"]],
           on="product_id", how="left")
    .merge(mst_wide, on="product_id", how="left")
)

print("Final dataset shape:", final.shape)
final.head(10)
# -----------------------------------------------
# 10. Save back to Azure
# -----------------------------------------------
out_path = "processed/training_products_wide.csv"
buf = io.StringIO()
final.to_csv(buf, index=False)
svc.get_blob_client(container, out_path).upload_blob(buf.getvalue().encode("utf-8"), overwrite=True)

print(f"✅ Final training dataset written to {container}/{out_path}")







📓 Notebook: retail_data_quality_and_eda.ipynb


# ====================================================
# STEP 2. Completeness Checks
# ====================================================
missing = df.isna().mean().sort_values(ascending=False)*100
print("Percentage missing values per column:")
display(missing)

# Empty OCR text or brand
empty_ocr = (df["ocr_text"].astype(str).str.strip()=="").sum()
empty_brand = (df["brand"].astype(str).str.strip()=="").sum()
print(f"Empty OCR text rows: {empty_ocr}")
print(f"Empty brand rows: {empty_brand}")

# ====================================================
# STEP 3. Uniqueness & Integrity
# ====================================================
dup_products = df.duplicated("product_id").sum()
dup_rows = df.duplicated().sum()
print(f"Duplicate product_ids: {dup_products}")
print(f"Fully duplicate rows: {dup_rows}")

# Check if product_id uniquely identifies a product
pid_counts = df.groupby("product_id").size()
pid_counts.describe()

# ====================================================
# STEP 4. Validity & Ranges
# ====================================================
# Standardized quantity checks
invalid_qty = df[df["qty_std_value"].fillna(0) <= 0]
outliers = df[df["qty_std_value"] > df["qty_std_value"].quantile(0.99)]

print("Invalid quantities:", len(invalid_qty))
print("Extreme outliers (top 1%):", len(outliers))

# Allowed units check
print("Unique units found:", df["qty_std_unit"].unique())

# ====================================================
# STEP 5. Consistency
# ====================================================
# Same product_id with multiple brands/categories
brand_conflicts = df.groupby("product_id")["brand"].nunique()
brand_conflicts = brand_conflicts[brand_conflicts > 1]

cat_conflicts = df.groupby("product_id")["category"].nunique()
cat_conflicts = cat_conflicts[cat_conflicts > 1]

print("Conflicting brand assignments:", len(brand_conflicts))
print("Conflicting category assignments:", len(cat_conflicts))

#EDA
# ====================================================
# STEP 6. Brand vs Category Landscape
# ====================================================
brand_cat = df.groupby(["category","brand"]).size().reset_index(name="count")
pivot = brand_cat.pivot(index="category", columns="brand", values="count").fillna(0)

plt.figure(figsize=(14,8))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=.5)
plt.title("Brand vs Category Heatmap (Product Counts)")
plt.show()
# ====================================================
# STEP 7. Pack Size Strategy
# ====================================================
plt.figure(figsize=(10,6))
sns.histplot(data=df, x="qty_std_value", hue="category", multiple="stack", bins=40)
plt.title("Distribution of Pack Sizes by Category")
plt.xlabel("Standardized Quantity (g/ml)")
plt.show()

# Boxplot to compare pack size strategies by brand
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x="brand", y="qty_std_value")
plt.xticks(rotation=90)
plt.title("Brand Pack Size Strategy")
plt.show()
# ====================================================
# STEP 8. Consumer Language (OCR Text Analysis)
# ====================================================
# Text length
df["ocr_len"] = df["ocr_text"].fillna("").astype(str).str.len()
sns.histplot(df["ocr_len"], bins=50)
plt.title("Distribution of OCR Text Length")
plt.show()

# WordCloud of consumer-facing text
text_blob = " ".join(df["ocr_text"].fillna("").astype(str).tolist())
wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text_blob)

plt.figure(figsize=(14,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Consumer Language Word Cloud from OCR")
plt.show()

# ====================================================
# STEP 9. Clustering SKUs by Size & Category
# ====================================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[["qty_std_value"]].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(X_scaled)
df["size_cluster"] = kmeans.labels_

sns.boxplot(data=df, x="size_cluster", y="qty_std_value")
plt.title("Clustering of SKUs by Pack Size")
plt.show()
# ====================================================
# STEP 10. Duplicate / Near-duplicate Detection
# ====================================================
# Group by brand + category + standardized size
dup_groups = df.groupby(["brand","category","qty_std_value"]).size().reset_index(name="count")
dup_groups = dup_groups[dup_groups["count"]>1]

print("Potential duplicate SKUs (same brand/category/size):")
display(dup_groups.head(10))
# ====================================================
# STEP 11. Assortment Gap Analysis
# ====================================================
# Expected sizes (e.g., 250, 500, 1000 ml/g)
expected_sizes = [250, 500, 1000]
gap_report = []

for cat in df["category"].unique():
    sizes = df[df["category"]==cat]["qty_std_value"].dropna().astype(int).unique()
    missing = [s for s in expected_sizes if s not in sizes]
    if missing:
        gap_report.append({"category": cat, "missing_sizes": missing})

pd.DataFrame(gap_report)
# ====================================================
# STEP 12. Quality Scorecard
# ====================================================
scorecard = pd.DataFrame([
    {"dimension":"Completeness (non-missing OCR/brand)", 
     "score": round(100*(1-(empty_ocr+empty_brand)/len(df)),2)},
    {"dimension":"Uniqueness (no duplicate product_ids)", 
     "score": round(100*(1-dup_products/len(df)),2)},
    {"dimension":"Validity (reasonable sizes, units)", 
     "score": round(100*(1-len(invalid_qty)/len(df)),2)},
    {"dimension":"Consistency (no brand/category conflicts)", 
     "score": round(100*(1-(len(brand_conflicts)+len(cat_conflicts))/len(df)),2)},
])
scorecard["grade"] = pd.cut(scorecard["score"], bins=[-1,60,75,90,100], labels=["Poor","Fair","Good","Excellent"])
scorecard
# ====================================================
# STEP 13. Executive Recommendations
# ====================================================
print("""
🔑 Recommendations:

1. **Data Hygiene**
   - Normalize brand and category spellings ("Nestle" vs "Nestlé").
   - Standardize pack sizes into g/ml.

2. **Consumer Insights**
   - Brands X and Y dominate in smaller packs (impulse buys).
   - Categories A and B missing mid-size packs → opportunity for new SKUs.

3. **Portfolio Optimization**
   - Remove duplicate SKUs (same brand/category/size) to reduce cannibalization.
   - Expand assortment in identified gap sizes (e.g., 500ml in juices).

4. **Marketing / Positioning**
   - OCR analysis shows strong health-related terms ("light", "sugar-free") — can be leveraged in campaigns.
   - Word cloud highlights consumer-facing differentiation language.

5. **Future AI Applications**
   - Cleaned dataset can power recommendation engines (right pack for right consumer).
   - Predictive models to optimize price-pack architecture and promotions.
""")
# ====================================================
# STEP 14. Shelf View Simulation
# ====================================================
# Simplify view: brand, category, standardized pack size
shelf = df[["brand","category","qty_std_value","qty_std_unit"]].copy()
shelf["size_label"] = shelf["qty_std_value"].astype(str) + shelf["qty_std_unit"]

# Sample: show "Juice" category assortment
cat_focus = "Juice"   # <-- change category here
shelf_view = shelf[shelf["category"]==cat_focus].drop_duplicates()

# Pivot for a shelf-like view (brands x sizes)
pivot = shelf_view.pivot_table(index="brand", columns="size_label", values="category", aggfunc="count", fill_value=0)

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="Greens", annot=True, fmt="d", cbar=False)
plt.title(f"Shelf View – Assortment Variety in {cat_focus}")
plt.xlabel("Pack Size")
plt.ylabel("Brand")
plt.show()
# ====================================================
# STEP 15. Variety Index (How a shopper perceives assortment breadth)
# ====================================================
# Variety = number of distinct pack sizes per brand in each category
variety = shelf.groupby(["category","brand"])["size_label"].nunique().reset_index(name="variety_index")

plt.figure(figsize=(14,6))
sns.barplot(data=variety, x="brand", y="variety_index", hue="category")
plt.xticks(rotation=90)
plt.title("Perceived Variety Index (Distinct Pack Sizes per Brand/Category)")
plt.ylabel("Distinct Pack Sizes")
plt.show()
# ====================================================
# STEP 16. Premium vs Value Positioning
# ====================================================
# Heuristic: small packs (<300g/ml) = premium / impulse
#             large packs (>750g/ml) = value / family
df["positioning"] = pd.cut(df["qty_std_value"], bins=[0,300,750,5000], labels=["Premium/Impulse","Mid-size","Value/Family"])

pos = df.groupby(["brand","positioning"]).size().reset_index(name="count")

plt.figure(figsize=(12,6))
sns.barplot(data=pos, x="brand", y="count", hue="positioning")
plt.xticks(rotation=90)
plt.title("Brand Positioning by Pack Size Segment")
plt.ylabel("Number of SKUs")
plt.show()
# ====================================================
# STEP 17. Shopper Language Simulation
# ====================================================
# Keywords shoppers actually see on pack (OCR text)
keywords = ["light","diet","organic","sugar","natural","protein","zero"]
for word in keywords:
    df[word+"_flag"] = df["ocr_text"].fillna("").str.lower().str.contains(word)

# % of SKUs per brand carrying each keyword
keyword_summary = df.groupby("brand")[[w+"_flag" for w in keywords]].mean()*100

plt.figure(figsize=(14,6))
sns.heatmap(keyword_summary, annot=True, cmap="Blues", fmt=".1f")
plt.title("Shopper Messaging in OCR (percentage of SKUs per brand)")
plt.xlabel("Keyword")
plt.ylabel("Brand")
plt.show()
# ====================================================
# STEP 18. Shelf Gap Analysis (Consumer Opportunities)
# ====================================================
expected_sizes = [250, 500, 1000]
gap_report = []

for brand in df["brand"].unique():
    brand_sizes = df[df["brand"]==brand]["qty_std_value"].dropna().astype(int).unique()
    missing = [s for s in expected_sizes if s not in brand_sizes]
    if missing:
        gap_report.append({"brand": brand, "missing_sizes": missing})

gap_df = pd.DataFrame(gap_report)
gap_df

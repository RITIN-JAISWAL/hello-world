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

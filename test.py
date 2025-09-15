# ==== JSON: flatten ALL types + build label lookups (read-only) ====
import json, pandas as pd
from azure.storage.blob import BlobServiceClient

CONN_STR  = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
CONTAINER = "rawdata"
JSON_BLOB = "dictionary_co_fmcg_cross.json"

svc = BlobServiceClient.from_connection_string(CONN_STR)
container = svc.get_container_client(CONTAINER)
js = json.loads(container.get_blob_client(JSON_BLOB).download_blob().readall().decode("utf-8","replace"))

# Full dict (ALL types present)
df_full  = pd.json_normalize(js.get("dict", []), max_level=1)

# Header table (nice for context)
df_hdr = pd.DataFrame([js.get("hdr", {})]).T.reset_index()
df_hdr.columns = ["key","value"]

# Type counts (see what types exist)
df_types = df_full["ty"].value_counts(dropna=False).rename_axis("type").reset_index(name="count")

# ---- Code -> label (works for ANY type id) ----
pref_label = df_full.get("ll").fillna(df_full.get("sl")).fillna(df_full.get("id"))
code_label_map = dict(zip(df_full.get("id"), pref_label))

# ---- Parent(code) + ValueId -> ValueLabel via child list 'b' (if present) ----
# Many dictionaries embed permissible values under 'b' for a parent id (often A*-type).
def explode_list(df, list_col, parent_keys=("id","ty")):
    if list_col not in df.columns: return pd.DataFrame()
    t = df[parent_keys + [list_col]].copy()
    t[list_col] = t[list_col].apply(lambda x: x if isinstance(x, list) else [])
    ex = t.explode(list_col, ignore_index=True)
    if ex.empty: return ex
    child = pd.json_normalize(ex[list_col]).add_prefix(f"{list_col}.")
    out = pd.concat([ex[parent_keys], child], axis=1)
    return out.drop(columns=[list_col], errors="ignore")

df_b = explode_list(df_full, "b", parent_keys=("id","ty"))
# child columns look like: b.id, b.sl, b.ll, …
if not df_b.empty:
    val_pref = df_b.get("b.ll").fillna(df_b.get("b.sl")).fillna(df_b.get("b.id"))
    # map: (parent_code, value_id_as_str) -> value_label
    value_label_map = {
        (str(pid), str(vid)): str(vlab)
        for pid, vid, vlab in zip(df_b["id"], df_b["b.id"], val_pref)
    }
else:
    value_label_map = {}

# Quick peeks
print("Header rows:", len(df_hdr))
print("Dict rows:", len(df_full))
print("Type counts:\n", df_types)
df_hdr.head(10), df_full.head(10)




























# ===== CSV PARSER (read-only; no uploads) =====
# - Reads the CSV text from Azure Blob
# - Does NOT try to find a header row (your sample lines are data-only)
# - Extracts:
#     * product_weight (float) + product_weight_unit (e.g., Gr, Ml, Kg)
#     * attributes_dict: {"A0001": "24", "A0002": "2", "C0001": "500", ...}
#     * attributes_text_raw: the full original line for verification
# - Expands the most frequent codes (top N) into *_value columns
#
# Tip: If you later learn the exact schema, you can split the line by the detected delimiter
#      and pick specific fields (e.g., a product name field). For now we keep everything in raw form.

import re
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobServiceClient
import csv

# ---- config ----
CONN_STR   = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
CONTAINER  = "rawdata"
CSV_BLOB   = "codification_co_fmcg.csv"
TOP_CODES  = 50  # expand columns for the most frequent N codes

# ---- connect & read raw text ----
svc = BlobServiceClient.from_connection_string(CONN_STR)
container = svc.get_container_client(CONTAINER)
csv_bytes = container.get_blob_client(CSV_BLOB).download_blob().readall()
csv_text  = csv_bytes.decode("utf-8", errors="replace")
lines     = [ln for ln in csv_text.splitlines() if ln.strip()]

# (Optional) detect delimiter just for your reference; we won’t rely on it
try:
    dialect = csv.Sniffer().sniff("\n".join(lines[:200]))
    delim = dialect.delimiter
    print("Detected delimiter (FYI):", repr(delim))
except Exception:
    delim = None
    print("Delimiter sniffing failed; proceeding as raw-line parser.")

# ---- regex helpers ----
# Codes like A0001:24, A0001-24, V0123:3, P0456-2, C0001:500, or just A0007
CODE_VAL_RE = re.compile(r"\b([A-Z]\d{4})\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)?\b")

# weight patterns: "500 Gr", "Gr 500", "150 Ml", "1,5 Kg", etc.
def parse_weight(text: str):
    if not isinstance(text, str):
        return None, None
    # number + unit
    m1 = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*(Gr|GR|g|G|Kg|KG|kg|MI|Ml|ML|mL|L|lt|Lt|LT)\b", text)
    # unit + number
    m2 = re.search(r"\b(Gr|GR|g|G|Kg|KG|kg|MI|Ml|ML|mL|L|lt|Lt|LT)\s*([0-9]+(?:[.,][0-9]+)?)\b", text)
    if m1:
        num, unit = m1.group(1), m1.group(2)
    elif m2:
        unit, num = m2.group(1), m2.group(2)
    else:
        return None, None
    num_norm = num.replace(".", "").replace(",", ".")
    try:
        return float(num_norm), unit
    except Exception:
        return None, unit

def parse_codes(text: str):
    if not isinstance(text, str):
        return []
    return CODE_VAL_RE.findall(text)

# ---- build a dataframe with one row per line ----
df = pd.DataFrame({"attributes_text_raw": lines})

# parse per line
weights, units, dicts, all_codes = [], [], [], []
for s in df["attributes_text_raw"]:
    w, u = parse_weight(s)
    weights.append(w); units.append(u)
    d = {}
    for code, val in parse_codes(s):
        d[code] = (val.replace(",", ".") if val else None)
        all_codes.append(code)
    dicts.append(d)

df["product_weight"] = weights
df["product_weight_unit"] = units
df["attributes_dict"] = dicts

# expand top-N frequent codes into columns with the numeric/text value
top_codes = pd.Series(all_codes).value_counts().head(TOP_CODES).index.tolist()
for code in top_codes:
    df[f"{code}_value"] = [d.get(code) for d in dicts]

print("Rows parsed:", len(df))
print("Columns now:", len(df.columns))
df.head(10)





























# ==== CSV: attach labels for ANY code type (read-only) ====
import re
import pandas as pd

# df  -> your parsed CSV DataFrame from the earlier read-only CSV block
# code_label_map  -> from JSON block above (ANY code -> label)
# value_label_map -> from JSON block above ((parent_code, value_id)-> value_label)

# 1) detect all flattened value columns (works for A/V/P/C/…)
value_cols = [c for c in df.columns if c.endswith("_value") and re.match(r"^[A-Z]\d{4}_value$", c)]
codes = [c.replace("_value","") for c in value_cols]

# 2) add attribute labels (same for the whole column)
for code in codes:
    df[f"{code}_attr_label"] = code_label_map.get(code, code)  # fallback to the code

# 3) add value labels row-wise (if we have them in the JSON dictionary)
def map_value_label(code, ser):
    # ser is the {code}_value column (strings like "402", "2", …)
    if not value_label_map:
        return pd.Series([None]*len(ser), index=ser.index)
    return ser.astype(str).map(lambda v: value_label_map.get((code, v)))

for code in codes:
    df[f"{code}_value_label"] = map_value_label(code, df[f"{code}_value"])

# 4) (optional) also expand labels for ALL codes found in attributes_dict (even if not in top-N)
#     This creates a long table you can pivot, covering every code/value seen per row.
def long_view_from_dict(df, dict_col="attributes_dict", row_id_col=None):
    if row_id_col is None:
        row_id_col = "_row_id"
        if row_id_col not in df: df[row_id_col] = range(len(df))
    rows = []
    for rid, d in zip(df[row_id_col], df[dict_col]):
        if isinstance(d, dict):
            for code, val in d.items():
                rows.append((rid, code, None if val in (None,"") else str(val)))
    if not rows: 
        return pd.DataFrame(columns=[row_id_col,"code","value","attr_label","value_label"])
    long_df = pd.DataFrame(rows, columns=[row_id_col,"code","value"])
    long_df["attr_label"]  = long_df["code"].map(lambda c: code_label_map.get(c, c))
    long_df["value_label"] = long_df.apply(lambda r: value_label_map.get((r["code"], str(r["value"]))) if r["value"] is not None else None, axis=1)
    return long_df

df_long = long_view_from_dict(df, dict_col="attributes_dict")

print("Wide columns added for:", codes[:10], "…")
print("df shape:", df.shape, "| long view rows:", len(df_long))
df.head(5), df_long.head(10)

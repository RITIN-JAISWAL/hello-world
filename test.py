# ===== JSON FLATTENER =====
# - Flattens your dictionary JSON to tabular CSVs
# - Outputs header (key/value), full dict (all types), type counts
# - Builds a generic code->label map (prefers long label 'll' then short 'sl')

import pandas as pd, json
from azure.storage.blob import BlobServiceClient
from io import StringIO

# ---- config ----
CONN_STR   = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
CONTAINER  = "rawdata"
JSON_BLOB  = "dictionary_co_fmcg_cross.json"

OUT_JSON_HDR_CSV     = "dictionary_hdr.csv"
OUT_JSON_FULL_CSV    = "dictionary_full.csv"
OUT_JSON_TYPES_CSV   = "dictionary_type_counts.csv"
OUT_JSON_LABELS_CSV  = "dictionary_code_label_map.csv"  # (optional convenience sheet)

svc = BlobServiceClient.from_connection_string(CONN_STR)
container = svc.get_container_client(CONTAINER)

def upload_csv(df: pd.DataFrame, name: str):
    container.get_blob_client(name).upload_blob(df.to_csv(index=False), overwrite=True)

# 1) load
js_bytes = container.get_blob_client(JSON_BLOB).download_blob().readall()
data_js  = json.loads(js_bytes.decode("utf-8", errors="replace"))

# 2) header (key/value)
df_hdr = pd.DataFrame([data_js.get("hdr", {})]).T.reset_index()
df_hdr.columns = ["key", "value"]
upload_csv(df_hdr, OUT_JSON_HDR_CSV)

# 3) full dict (all types)
df_full = pd.json_normalize(data_js.get("dict", []), max_level=1)
upload_csv(df_full, OUT_JSON_FULL_CSV)

# 4) type counts (to see A/V/P/… composition)
type_counts = df_full["ty"].value_counts(dropna=False).rename_axis("type").reset_index(name="count")
upload_csv(type_counts, OUT_JSON_TYPES_CSV)

# 5) convenient code->label map (prefer ll, then sl, then id)
if {"id","ll","sl"}.issubset(df_full.columns):
    label = df_full["ll"].fillna(df_full["sl"]).fillna(df_full["id"])
    df_map = pd.DataFrame({"code": df_full["id"], "label": label})
    upload_csv(df_map, OUT_JSON_LABELS_CSV)

print(" JSON done. Wrote:",
      f"{OUT_JSON_HDR_CSV}, {OUT_JSON_FULL_CSV}, {OUT_JSON_TYPES_CSV}" +
      (f", {OUT_JSON_LABELS_CSV}" if 'df_map' in locals() else ""))
























# ===== CSV ENRICHER =====
# - Detects & captures a CSV "header block" (metadata) above the real table header
# - Reads the table from the detected header row
# - Extracts product weight + unit from the Linea text
# - Parses ALL attribute codes (A/V/P/…) like A0001:24 or A0001-24
# - Keeps originals for verification
# - Saves: codification_csv_header_block.csv, codification_csv_data_only.csv, codification_co_fmcg_enriched.csv

import csv, re, json
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobServiceClient

# ---- config ----
CONN_STR   = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
CONTAINER  = "rawdata"
CSV_BLOB   = "codification_co_fmcg.csv"
LINEA_COL  = "Linea"          # change if your column name differs
TOP_CODES  = 50               # how many most-frequent codes to expand into columns

svc = BlobServiceClient.from_connection_string(CONN_STR)
container = svc.get_container_client(CONTAINER)

def upload_csv(df: pd.DataFrame, name: str):
    container.get_blob_client(name).upload_blob(df.to_csv(index=False), overwrite=True)

# 1) download raw text
csv_bytes = container.get_blob_client(CSV_BLOB).download_blob().readall()
csv_text  = csv_bytes.decode("utf-8", errors="replace")
lines     = csv_text.splitlines()

# 2) sniff delimiter + find real header row (contains "Linea" and likely id columns)
dialect = csv.Sniffer().sniff("\n".join(lines[:200]))
delim   = dialect.delimiter

def is_table_header(line: str) -> bool:
    toks = [t.strip() for t in line.split(delim)]
    low  = line.lower()
    return ("linea" in low) and (len(toks) >= 3) and any(k in low for k in ["dcode","idart","idarticulo","codigo"])

header_idx = next((i for i,l in enumerate(lines[:300]) if is_table_header(l)), None)
if header_idx is None:  # fallback: first line that has 'linea'
    header_idx = next((i for i,l in enumerate(lines[:300]) if "linea" in l.lower() and len(l.split(delim))>=3), None)
if header_idx is None:
    raise RuntimeError("Couldn't detect the real table header row—specify LINEA_COL or provide an anchor word.")

# 3) capture CSV header block (metadata) above the table header
meta_lines = lines[:header_idx]
meta_rows  = []
for raw in meta_lines:
    parts = [p.strip() for p in raw.split(delim)]
    if len(parts) >= 2:
        meta_rows.append({"key": parts[0], "value": delim.join(parts[1:])})
    elif raw.strip():
        meta_rows.append({"key": f"raw_{len(meta_rows)+1}", "value": raw})

df_csv_hdr = pd.DataFrame(meta_rows)
upload_csv(df_csv_hdr, "codification_csv_header_block.csv")

# 4) read the actual table
df_csv = pd.read_csv(
    StringIO(csv_text),
    engine="python",
    sep=delim,
    header=header_idx,
    on_bad_lines="skip",
    dtype=str
)
upload_csv(df_csv, "codification_csv_data_only.csv")

# 5) parse weight + attributes (ALL code types)
CODE_VAL_RE = re.compile(r"\b([A-Z]\d{4})\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)?\b")
def parse_codes(text: str):
    if not isinstance(text, str): return []
    return CODE_VAL_RE.findall(text)

def parse_weight(text: str):
    if not isinstance(text, str): return None, None
    # number + unit OR unit + number
    m = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*(Gr|GR|g|G|Kg|KG|kg|MI|Ml|ML|mL|L|lt|Lt|LT)\b", text)
    if not m:
        m2 = re.search(r"\b(Gr|GR|g|G|Kg|KG|kg|MI|Ml|ML|mL|L|lt|Lt|LT)\s*([0-9]+(?:[.,][0-9]+)?)\b", text)
        if not m2: return None, None
        unit, num = m2.group(1), m2.group(2)
    else:
        num, unit = m.group(1), m.group(2)
    num_norm = num.replace(".", "").replace(",", ".")
    try: return float(num_norm), unit
    except: return None, unit

weights, units, attr_dicts, all_codes = [], [], [], []
for s in df_csv[LINEA_COL].fillna(""):
    w,u = parse_weight(s); weights.append(w); units.append(u)
    d = {}
    for code,val in parse_codes(s):
        d[code] = val.replace(",", ".") if val else None
        all_codes.append(code)
    attr_dicts.append(d)

df_csv["product_weight"]       = weights
df_csv["product_weight_unit"]  = units
df_csv["attributes_text_raw"]  = df_csv[LINEA_COL]   # keep original for verification
df_csv["attributes_dict"]      = attr_dicts          # per-row JSON dict of codes

# 6) expand most frequent codes to columns (values only; labels added from JSON block below)
top_codes = pd.Series(all_codes).value_counts().head(TOP_CODES).index.tolist()
for code in top_codes:
    df_csv[f"{code}_value"] = [d.get(code) for d in attr_dicts]

upload_csv(df_csv, "codification_co_fmcg_enriched.csv")
print(" CSV done. Wrote:",
      "codification_csv_header_block.csv, codification_csv_data_only.csv, codification_co_fmcg_enriched.csv")


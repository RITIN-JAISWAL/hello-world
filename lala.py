import pandas as pd

# Split column 4 into separate columns
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)

# Rename the new columns as attribute 1, attribute 2, ...
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]

# Concatenate back to mst
mst = pd.concat([mst, split_cols], axis=1)




import pandas as pd
import numpy as np
import re

# ---------- 1) Find the right columns in `attribute` ----------
# handles naming variants like: IdAtributo / idAtributo / id_atributo
def _pick(colnames, candidates):
    cand = {c.lower() for c in candidates}
    for c in colnames:
        if c.lower() in cand:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(colnames)}")

id_col   = _pick(attribute.columns, ["IdAtributo","idAtributo","id_atributo","idatributo"])
desc_col = _pick(attribute.columns, ["Descricao","Descrição","Descripcion","descricao","descrição"])

# ---------- 2) Build prefix -> label map, e.g. 'A0000' -> 'Sector' ----------
attr_tmp = attribute[[id_col, desc_col]].copy()
attr_tmp[id_col] = pd.to_numeric(attr_tmp[id_col], errors="coerce").astype("Int64")
attr_tmp = attr_tmp.dropna(subset=[id_col])

attr_tmp["prefix"] = "A" + attr_tmp[id_col].astype(int).astype(str).str.zfill(4)
attr_map = dict(zip(attr_tmp["prefix"], attr_tmp[desc_col].astype(str).str.strip()))

# ---------- 3) Function to convert "A0000:1" -> "Sector:1" ----------
def _map_token(token: str):
    if token is None or (isinstance(token, float) and np.isnan(token)):
        return np.nan
    s = str(token).strip()
    if s in ("", "None", "nan"):
        return np.nan
    # split by ":" once
    if ":" in s:
        code, val = s.split(":", 1)
        val = val.strip()
    else:
        code, val = s, ""
    code = code.strip().upper()
    prefix = code[:5]  # 'A0000', 'A0017', ...
    label = attr_map.get(prefix, prefix)
    return f"{label}:{val}" if val != "" else label

# ---------- 4) Apply to all "attribute *" columns ----------
# ensure column names are strings (avoids the 'int has no attribute startswith' error)
mst.columns = mst.columns.map(str)

attr_cols = [c for c in mst.columns if c.lower().startswith("attribute")]
for c in attr_cols:
    mst[c] = mst[c].apply(_map_token)







































import pandas as pd
import numpy as np
import json

# --- Load the JSON dictionary (adjust path/variable if you already have it in memory) ---
# If you already have it as a dict named `attr_dict`, skip the file load and set:
#   data = attr_dict
with open("dictionary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# The JSON appears to have a top-level key 'dict' which is a list of attribute nodes
nodes = data["dict"] if isinstance(data, dict) and "dict" in data else data

# --- 1) Build A#### -> Descricao from the `attribute` dataframe ---
def _pick(colnames, candidates):
    cand = {c.lower() for c in candidates}
    for c in colnames:
        if c.lower() in cand:
            return c
    raise KeyError(f"Could not find any of {candidates} in {list(colnames)}")

id_col   = _pick(attribute.columns, ["IdAtributo","idAtributo","id_atributo","idatributo"])
desc_col = _pick(attribute.columns, ["Descricao","Descripción","Descripcion","descrição","descripcion","descricao"])

attr_map = (
    attribute[[id_col, desc_col]]
      .assign(_id = pd.to_numeric(attribute[id_col], errors="coerce").astype("Int64"))
      .dropna(subset=["_id"])
      .assign(prefix = lambda d: "A" + d["_id"].astype(int).astype(str).str.zfill(4))
      .set_index("prefix")[desc_col]
      .astype(str).str.strip()
      .to_dict()
)

# --- 2) Build A#### -> { value_id(int) -> sl(str) } from the JSON ---
value_map = {}
for node in nodes:
    code = str(node.get("A", "")).upper().strip()          # e.g., "A0000"
    children = node.get("b", [])                            # list of value entries
    id_to_sl = {}
    for ch in children:
        try:
            vid = int(str(ch.get("id", "")).strip())
        except Exception:
            continue
        sl = str(ch.get("sl", "")).strip()
        if sl:
            id_to_sl[vid] = sl
    if code and id_to_sl:
        value_map[code] = id_to_sl

# --- 3) Rewriter: "A0000:1" -> "Sector: Bebidas" ---
def rewrite_token(token):
    if pd.isna(token) or str(token).strip() in ("", "None", "nan"):
        return np.nan
    s = str(token).strip()
    if ":" in s:
        code, v = s.split(":", 1)
        code = code.strip().upper()
        try:
            vid = int(v.strip())
        except Exception:
            vid = None
    else:
        code, vid = s.strip().upper(), None

    # left label from attribute df (Descricao); fallback to code prefix
    left = attr_map.get(code[:5], code[:5])

    if vid is None:
        return left

    # right label from JSON (sl)
    right = value_map.get(code[:5], {}).get(vid, str(vid))
    return f"{left}: {right}"

# --- 4) Apply to all split columns that start with 'attribute' ---
mst.columns = mst.columns.map(str)  # avoid 'int has no attribute startswith'
attr_cols = [c for c in mst.columns if c.lower().startswith("attribute")]
for c in attr_cols:
    mst[c] = mst[c].apply(rewrite_token)

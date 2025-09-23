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

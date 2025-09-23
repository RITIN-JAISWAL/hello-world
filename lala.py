import pandas as pd

# Split column 4 into separate columns
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)

# Rename the new columns as attribute 1, attribute 2, ...
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]

# Concatenate back to mst
mst = pd.concat([mst, split_cols], axis=1)



import pandas as pd
import numpy as np
import json

# --- 0) Split column 4 (0-based) into 'attribute i' columns
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]
mst = pd.concat([mst, split_cols], axis=1)

# --- 1) Load/prepare dictionary JSON ---
# If already loaded to a variable called `data`, remove the file-read.
# with open("dictionary.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

nodes = data["dict"] if isinstance(data, dict) and "dict" in data else data

# Build: code -> attribute label (sl), and code -> {value_id -> value_label (sl)}
attr_label = {}           # e.g., {'A0000': 'Sector', 'A0001': 'Categoría', ...}
value_label_map = {}      # e.g., {'A0000': {'1':'Bebidas','4':'Alimentos', ...}, ...}

for node in nodes:
    code = str(node.get("id", "")).strip().upper()   # 'A0000', 'A0001', ...
    if not code:
        continue
    # attribute label
    sl_attr = str(node.get("sl", "")).strip()
    if sl_attr:
        # title-case nicely (optional)
        attr_label[code] = sl_attr.title() if sl_attr.isupper() else sl_attr

    # children values
    vals = {}
    for ch in node.get("b", []) or []:
        vid = str(ch.get("id", "")).strip()          # keep as string to avoid type mismatch
        slv = str(ch.get("sl", "")).strip()
        if vid and slv:
            vals[vid] = slv
    if vals:
        value_label_map[code] = vals

# --- 2) Converter: "A0000:1" -> "Sector: Bebidas"
def token_to_readable(tok):
    if pd.isna(tok):
        return np.nan
    s = str(tok).strip()
    if not s or s.lower() in {"nan", "none"}:
        return np.nan

    # split into code and value id
    if ":" in s:
        code, val = s.split(":", 1)
        code, val = code.strip().upper(), val.strip()
    else:
        code, val = s.strip().upper(), ""

    # attribute label (left)
    left = attr_label.get(code[:5], code[:5])

    # value label (right) – look up by the same code
    right = value_label_map.get(code[:5], {}).get(val, val)

    return f"{left}: {right}" if right else left

# --- 3) Create 'attribute_value i' columns in one shot
attr_cols = [c for c in mst.columns if str(c).lower().startswith("attribute ")]
values_df = mst[attr_cols].applymap(token_to_readable)
values_df.columns = [f'attribute_value {i+1}' for i in range(values_df.shape[1])]

# Attach to mst
mst = pd.concat([mst, values_df], axis=1)





import pandas as pd
import numpy as np
import re

# --- 0) Split column 4 into 'attribute i' columns (you already have this) ---
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]
mst = pd.concat([mst, split_cols], axis=1)

# --- 1) Build A#### -> label map using `attribute` + `dreamy` ---
# We expect:
#   attribute: columns ['IdAtributo', 'IdTipoAtributo', ...]
#   dreamy:    columns ['IdTipoAtributo', 'Descricao', ...]  (Descricao = human label)
# This produces: 'A0000' -> 'Sector', 'A0001' -> 'Category', etc.

# Make a safe copy with normalized column names (case-insensitive)
attr_df = attribute.copy()
dreamy_df = dreamy.copy()

# Coerce numeric ids
attr_df['IdAtributo'] = pd.to_numeric(attr_df['IdAtributo'], errors='coerce').astype('Int64')
attr_df['IdTipoAtributo'] = pd.to_numeric(attr_df['IdTipoAtributo'], errors='coerce').astype('Int64')
dreamy_df['IdTipoAtributo'] = pd.to_numeric(dreamy_df['IdTipoAtributo'], errors='coerce').astype('Int64')

# Join to fetch the human label from dreamy
attr_joined = (
    attr_df[['IdAtributo','IdTipoAtributo']]
    .merge(dreamy_df[['IdTipoAtributo','Descricao']], on='IdTipoAtributo', how='left')
)

# Build A-prefix and map to label
attr_joined['Acode'] = 'A' + attr_joined['IdAtributo'].astype(int).astype(str).str.zfill(4)
Acode_to_label = dict(zip(attr_joined['Acode'], attr_joined['Descricao'].astype(str)))

# --- 2) Converter: "A0000:1" -> "Sector: 1" using the map above ---
def to_attr_value(token: str):
    if pd.isna(token):
        return np.nan
    s = str(token).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return np.nan

    if ':' in s:
        code, val = s.split(':', 1)
        code, val = code.strip().upper(), val.strip()
    else:
        code, val = s.strip().upper(), ''   # sometimes there may be no value part

    label = Acode_to_label.get(code[:5], code[:5])   # fall back to prefix if missing
    return f"{label}: {val}" if val != '' else label

# --- 3) Create the attribute_values i columns in one shot ---
mst.columns = mst.columns.map(str)  # ensure string names
attr_cols = [c for c in mst.columns if c.lower().startswith('attribute ')]
attr_values = mst[attr_cols].applymap(to_attr_value)
attr_values.columns = [f'attribute_values {i+1}' for i in range(attr_values.shape[1])]

# Attach to mst
mst = pd.concat([mst, attr_values], axis=1)

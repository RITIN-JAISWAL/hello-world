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

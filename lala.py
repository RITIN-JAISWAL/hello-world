import pandas as pd
import numpy as np

# --- 1) Build a code -> label map from `attribute` ---
# Adjust the column names if yours differ
# Example columns (from your screenshot): idAtributo, Descripcion
attr_map = (
    attribute
      .assign(idAtributo=lambda d: d['idAtributo'].astype(str).str.upper().str.strip())
      .set_index('idAtributo')['Descripcion']
      .astype(str)
      .str.strip()
      .to_dict()
)

# --- 2) Helper to parse one mst cell like "A00001:1,A00022:3" into {label: value} ---
def parse_and_map(cell):
    if pd.isna(cell) or not str(cell).strip():
        return {}
    out = {}
    for token in str(cell).split(','):
        token = token.strip()
        if not token:
            continue
        # split into code and value
        if ':' in token:
            code, val = token.split(':', 1)
            code = code.upper().strip()
            val = val.strip()
        else:
            code, val = token.upper().strip(), np.nan

        # map code -> human label; if not found, keep the code
        label = attr_map.get(code, code)
        out[label] = val
    return out

# --- 3) Apply to the 4th column of `mst` and expand ---
# If you know the name, use mst['<your_col_name>'] instead of iloc[:, 3]
mapped_dicts = mst.iloc[:, 3].apply(parse_and_map)

# A) If you want the *replaced tokens* still as comma text like "Sector:1, Brand:2, ...":
def dict_to_str(d):
    return ', '.join(f'{k}:{v}' for k, v in d.items()) if d else np.nan

mst['attributes_readable'] = mapped_dicts.apply(dict_to_str)

# B) If you prefer *proper columns* (one col per attribute, holding the values):
expanded = pd.json_normalize(mapped_dicts).astype('object')  # keeps strings like "1"
# Optional: prefix columns if you like
# expanded = expanded.add_prefix('attr_')

# Attach to mst (and drop the original raw column if desired)
mst = pd.concat([mst.drop(mst.columns[3], axis=1), expanded], axis=1)

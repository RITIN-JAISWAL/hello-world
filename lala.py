import pandas as pd
import numpy as np
import re

# --- Build prefix -> label map from `attribute` ---
# idAtributo might be float/str; coerce to int and zero-pad to make A####.
attr = attribute[['idAtributo','Descripcion']].copy()
attr['idAtributo'] = pd.to_numeric(attr['idAtributo'], errors='coerce').dropna().astype(int)
attr['prefix'] = 'A' + attr['idAtributo'].astype(str).str.zfill(4)
attr_map = dict(zip(attr['prefix'], attr['Descripcion'].astype(str).str.strip()))

# --- Parser for one cell like "A00001:1,A00012:3, A00022:9" ---
def map_cell(cell):
    if pd.isna(cell) or not str(cell).strip():
        return {}
    out = {}
    for tok in re.split(r'\s*,\s*', str(cell).strip()):
        if not tok:
            continue
        # split into code and value
        if ':' in tok:
            code, val = tok.split(':', 1)
            val = val.strip()
        else:
            code, val = tok, np.nan
        code = code.strip().upper()

        # use the A#### prefix to look up the human label
        prefix = code[:5]  # e.g. 'A0000', 'A0001', ...
        label = attr_map.get(prefix, prefix)  # fall back to the prefix if unknown
        out[label] = val
    return out

# --- Apply to mst column 4 (0-based) and expand ---
mapped = mst.iloc[:, 4].apply(map_cell)

# If you want a readable text version:
mst['attributes_readable'] = mapped.apply(lambda d: ', '.join(f'{k}:{v}' for k,v in d.items()) if d else np.nan)

# If you want proper columns (Sector, Category, Company, Brand, ...):
expanded = pd.json_normalize(mapped).astype('object')  # keep values as strings like "1"
# Attach and drop the raw column-4 if you don’t need it
mst = pd.concat([mst.drop(mst.columns[4], axis=1), expanded], axis=1)

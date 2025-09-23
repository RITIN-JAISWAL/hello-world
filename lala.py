import pandas as pd

# Split column 4 into separate columns
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)

# Rename the new columns as attribute 1, attribute 2, ...
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]

# Concatenate back to mst
mst = pd.concat([mst, split_cols], axis=1)






import pandas as pd
import numpy as np

# --- 1. Build a mapping from A#### prefix to Descricao ---
attr_map = {}
for _, row in attribute.iterrows():
    try:
        id_val = int(row['idAtributo'])
        prefix = "A" + str(id_val).zfill(4)   # e.g. 0 -> A0000, 1 -> A0001
        attr_map[prefix] = str(row['Descricao']).strip()
    except:
        continue

# --- 2. Function to map one token like "A0000:1" ---
def replace_code(token):
    if pd.isna(token) or not str(token).strip():
        return np.nan
    token = str(token).strip()
    if ":" in token:
        code, val = token.split(":", 1)
        code = code.strip().upper()
        val = val.strip()
        label = attr_map.get(code, code)  # fallback: keep original code
        return f"{label}:{val}"
    else:
        code = token.strip().upper()
        label = attr_map.get(code, code)
        return label

# --- 3. Apply the replacement to your split columns ---
for col in [c for c in mst.columns if c.startswith("attribute")]:
    mst[col] = mst[col].apply(replace_code)

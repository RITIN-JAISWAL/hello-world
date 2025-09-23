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
import json

# --- 0) Split column 4 into 'attribute i' columns (you already had this)
split_cols = mst.iloc[:, 4].astype(str).str.split(',', expand=True)
split_cols.columns = [f'attribute {i+1}' for i in range(split_cols.shape[1])]
mst = pd.concat([mst, split_cols], axis=1)

# --- 1) Build maps from the JSON dictionary you loaded into `data`
nodes = data["dict"] if isinstance(data, dict) and "dict" in data else data

attr_label = {}        # 'A0000' -> attribute label from JSON, e.g. "Sector"
value_label_map = {}   # 'A0000' -> {"1": "Bebidas", "4": "Alimentos", ...}

for node in nodes:
    code = str(node.get("id", "")).strip().upper()
    if not code:
        continue
    sl_attr = str(node.get("sl", "")).strip()
    if sl_attr:
        attr_label[code] = sl_attr.title() if sl_attr.isupper() else sl_attr

    vals = {}
    for ch in node.get("b", []) or []:
        vid = str(ch.get("id", "")).strip()
        slv = str(ch.get("sl", "")).strip()
        if vid and slv:
            vals[vid] = slv
    if vals:
        value_label_map[code] = vals

# --- 2) Build the "middle" label from attribute + dreamy
# attribute: columns include 'IdAtributo', 'IdTipoAtributo'
# dreamy:    columns include 'IdTipoAtributo', 'Descricao'
attr_df = attribute[['IdAtributo','IdTipoAtributo']].copy()
dreamy_df = dreamy[['IdTipoAtributo','Descricao']].copy()

attr_df['IdAtributo'] = pd.to_numeric(attr_df['IdAtributo'], errors='coerce').astype('Int64')
attr_df['IdTipoAtributo'] = pd.to_numeric(attr_df['IdTipoAtributo'], errors='coerce').astype('Int64')
dreamy_df['IdTipoAtributo'] = pd.to_numeric(dreamy_df['IdTipoAtributo'], errors='coerce').astype('Int64')

attr_join = (
    attr_df.merge(dreamy_df, on='IdTipoAtributo', how='left')  # adds dreamy.Descricao
)
attr_join['Acode'] = 'A' + attr_join['IdAtributo'].astype(int).astype(str).str.zfill(4)

# Map: 'A0000' -> dreamy.Descricao   (the “middle” label you want)
Acode_to_dreamy_desc = dict(zip(attr_join['Acode'], attr_join['Descricao'].astype(str)))

# --- 3) Converter: "A0000:1" -> "Sector:<dreamy.Descricao>:Bebidas"
def token_to_full(tok):
    if pd.isna(tok):
        return np.nan
    s = str(tok).strip()
    if not s or s.lower() in {"nan", "none"}:
        return np.nan

    if ":" in s:
        code, val = s.split(":", 1)
        code, val = code.strip().upper(), val.strip()
    else:
        code, val = s.strip().upper(), ""

    prefix = code[:5]

    left   = attr_label.get(prefix, prefix)                       # from JSON (e.g., "Sector")
    middle = Acode_to_dreamy_desc.get(prefix, "")                 # from attribute+dreamy (e.g., "SECTOR")
    right  = value_label_map.get(prefix, {}).get(val, val)        # from JSON values (e.g., "Bebidas")

    if middle:
        return f"{left}:{middle}:{right}" if right else f"{left}:{middle}"
    else:
        return f"{left}:{right}" if right else left

# --- 4) Create 'attribute_value i' columns
mst.columns = mst.columns.map(str)
attr_cols = [c for c in mst.columns if c.lower().startswith("attribute ")]
values_df = mst[attr_cols].applymap(token_to_full)
values_df.columns = [f'attribute_value {i+1}' for i in range(values_df.shape[1])]

# Attach back
mst = pd.concat([mst, values_df], axis=1)






import pandas as pd

# --- 1) Build dataframe from JSON ---
rows = []
for node in J["dict"]:   # assuming your JSON is in variable J
    code = node.get("id", "").strip().upper()   # e.g., "A0000"
    sl   = str(node.get("sl", "")).strip()
    if code.startswith("A"):
        try:
            num_id = int(code[1:])   # "A0000" -> 0
        except:
            continue
        rows.append({"ID": num_id, "SL": sl})

json_df = pd.DataFrame(rows)

# --- 2) Normalise attribute dataframe ---
attr_df = attribute.copy()
attr_df = attr_df.rename(columns={"IdAtributo": "ID", "Descricao": "Descricao"})
attr_df["ID"] = pd.to_numeric(attr_df["ID"], errors="coerce").astype("Int64")

# --- 3) Join on ID ---
merged = json_df.merge(attr_df, on="ID", how="inner")

# --- 4) Find mismatches ---
mismatches = merged[merged["SL"].str.strip().str.lower() != merged["Descricao"].str.strip().str.lower()]

print(mismatches)



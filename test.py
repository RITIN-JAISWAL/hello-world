# View CSV/XLSX/JSON from blob without registering as a Data asset
from azureml.core import Workspace
import pandas as pd
import json
from pathlib import Path

# Auto-load workspace (no hardcoding)
ws = Workspace.from_config()
ds = ws.get_default_datastore()   # or Datastore.get(ws, "workspaceblobstore")

# Folder to place local copies
local_dir = Path("blob_preview")
local_dir.mkdir(exist_ok=True)

# --- filenames (adjust paths if they’re inside a folder) ---
csv_blob  = "codification_co_fmcg.csv"
json_blob = "dictionary_co_fmcg_cross.json"
xlsx_blob = "Attributes definition and types4.xlsx"

# Download each file locally (prefix can be a full path in the container)
ds.download(target_path=str(local_dir), prefix=csv_blob, overwrite=True, show_progress=True)
ds.download(target_path=str(local_dir), prefix=json_blob, overwrite=True, show_progress=True)
ds.download(target_path=str(local_dir), prefix=xlsx_blob, overwrite=True, show_progress=True)

csv_path  = local_dir / csv_blob
json_path = local_dir / json_blob
xlsx_path = local_dir / xlsx_blob

# --- View CSV ---
df_csv = pd.read_csv(csv_path)
display(df_csv.head())

# --- View JSON ---
with open(json_path, "r", encoding="utf-8") as f:
    data_json = json.load(f)
print(type(data_json))
print(str(data_json)[:1000])  # preview first ~1000 chars

# If JSON is records/table-like:
# df_json = pd.read_json(json_path, lines=False)  # adjust orient if needed
# display(df_json.head())

# --- View Excel (first sheet or specify sheet_name=...) ---
df_xlsx = pd.read_excel(xlsx_path, sheet_name=0)
display(df_xlsx.head())






from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
import pandas as pd, json, io

storage_account = "<your-storage-account>"         # e.g. "mystorageacct"
container       = "<your-container>"               # e.g. "mycontainer"

credential = DefaultAzureCredential()

def read_csv_blob(blob_name: str) -> pd.DataFrame:
    url = f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}"
    blob = BlobClient.from_blob_url(url, credential=credential)
    stream = io.BytesIO(blob.download_blob().readall())
    return pd.read_csv(stream)

def read_xlsx_blob(blob_name: str, **read_excel_kwargs) -> pd.DataFrame:
    url = f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}"
    blob = BlobClient.from_blob_url(url, credential=credential)
    stream = io.BytesIO(blob.download_blob().readall())
    return pd.read_excel(stream, **read_excel_kwargs)

def read_json_blob(blob_name: str):
    url = f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}"
    blob = BlobClient.from_blob_url(url, credential=credential)
    return json.loads(blob.download_blob().readall())

# Usage
df_csv  = read_csv_blob("codification_co_fmcg.csv")
df_xlsx = read_xlsx_blob("Attributes definition and types4.xlsx", sheet_name=0)
js      = read_json_blob("dictionary_co_fmcg_cross.json")

display(df_csv.head())
display(df_xlsx.head())
print(str(js)[:1000])

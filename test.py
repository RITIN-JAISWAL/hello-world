from azureml.core import Workspace, Datastore
import pandas as pd, json, io, requests

# 1. Load AML workspace
ws = Workspace.from_config()

# 2. Get datastore (default or by name)
ds = Datastore.get(ws, "productcodingstorage")

# 3. Build URLs for each blob in rawdata/
csv_url  = ds.path("rawdata/codification_co_fmcg.csv").as_download_url()
json_url = ds.path("rawdata/dictionary_co_fmcg_cross.json").as_download_url()
xlsx_url = ds.path("rawdata/Attributes definition and types4.xlsx").as_download_url()

print("Secure URLs generated (with SAS):")
print(csv_url)

# 4. Stream directly into Pandas / JSON
df_csv = pd.read_csv(csv_url)
print("CSV Preview:")
print(df_csv.head())

df_xlsx = pd.read_excel(xlsx_url)
print("Excel Preview:")
print(df_xlsx.head())

# For JSON, fetch into memory
r = requests.get(json_url)
data_json = r.json()
print("JSON Preview (first 300 chars):")
print(str(data_json)[:300])

from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import pandas as pd
import json

# ---------------------------
# 1. Connect to Blob Storage
# ---------------------------
conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"

svc = BlobServiceClient.from_connection_string(conn_str)
container = svc.get_container_client(container_name)

# ---------------------------
# 2. Read Excel file
# ---------------------------
excel_blob = "Attributes definition and types.xlsx"
excel_client = container.get_blob_client(excel_blob)
excel_bytes = excel_client.download_blob().readall()

df_excel = pd.read_excel(BytesIO(excel_bytes), engine="openpyxl")
print("✅ Excel file loaded")
print(df_excel.head())

# ---------------------------
# 3. Read CSV file
# ---------------------------
csv_blob = "codification_co_fmcg.csv"
csv_client = container.get_blob_client(csv_blob)
csv_bytes = csv_client.download_blob().readall()

csv_text = csv_bytes.decode("utf-8", errors="replace")
df_csv = pd.read_csv(StringIO(csv_text))
print("\n✅ CSV file loaded")
print(df_csv.head())

# ---------------------------
# 4. Read JSON file
# ---------------------------
json_blob = "dictionary_co_fmcg_cross.json"
json_client = container.get_blob_client(json_blob)
json_bytes = json_client.download_blob().readall()

data_json = json.loads(json_bytes.decode("utf-8", errors="replace"))
print("\n✅ JSON file loaded")
# Safely preview depending on structure
if isinstance(data_json, dict):
    print({k: data_json[k] for k in list(data_json)[:5]})
elif isinstance(data_json, list):
    print(data_json[:3])
else:
    print(type(data_json))

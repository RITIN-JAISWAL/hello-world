from azure.storage.blob import BlobServiceClient
import pandas as pd
import json

# Replace with your connection string
conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"

# Connect
blob_service = BlobServiceClient.from_connection_string(conn_str)
container_client = blob_service.get_container_client(container_name)

# Example: read CSV
blob_client = container_client.get_blob_client("codification_co_fmcg.csv")
stream = blob_client.download_blob().readall()
df = pd.read_csv(pd.compat.StringIO(stream.decode("utf-8")))
print(df.head())

# Example: read JSON
blob_client = container_client.get_blob_client("dictionary_co_fmcg_cross.json")
stream = blob_client.download_blob().readall()
data = json.loads(stream)
print(data)


















from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import pandas as pd
import json

# --- connect ---
conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"
svc = BlobServiceClient.from_connection_string(conn_str)
container = svc.get_container_client(container_name)

# ---------- CSV ----------
csv_blob = container.get_blob_client("codification_co_fmcg.csv")
csv_bytes = csv_blob.download_blob().readall()
csv_text = csv_bytes.decode("utf-8", errors="replace")
df_csv = pd.read_csv(StringIO(csv_text))
df_csv.head()   # keep it small to avoid IOPub overflow

# ---------- JSON ----------
json_blob = container.get_blob_client("dictionary_co_fmcg_cross.json")
json_bytes = json_blob.download_blob().readall()
data_json = json.loads(json_bytes.decode("utf-8", errors="replace"))
# Peek safely
list(data_json)[:5] if isinstance(data_json, dict) else data_json[:3]

# ---------- Excel (XLSX) ----------
xlsx_blob = container.get_blob_client("Attributes definition and types.xlsx")
xlsx_bytes = xlsx_blob.download_blob().readall()
df_xlsx = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=0, engine="openpyxl")  # engine optional if installed
df_xlsx.head()

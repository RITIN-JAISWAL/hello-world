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













from azure.storage.blob import BlobServiceClient
import reprlib

conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"

svc = BlobServiceClient.from_connection_string(conn_str)
container = svc.get_container_client(container_name)

print("Account URL:", svc.url)  # sanity check (no spaces, lowercase account)
print("Container URL:", container.url)

for b in container.list_blobs():
    # repr() shows hidden chars like \n, trailing spaces, double spaces, etc.
    print("->", repr(b.name))






from io import BytesIO, StringIO
import pandas as pd, json

def read_excel_blob(container, blob_name):
    blob = container.get_blob_client(blob_name)
    data = blob.download_blob().readall()
    return pd.read_excel(BytesIO(data), engine="openpyxl")

def read_csv_blob(container, blob_name):
    blob = container.get_blob_client(blob_name)
    data = blob.download_blob().readall().decode("utf-8", errors="replace")
    return pd.read_csv(StringIO(data))

def read_json_blob(container, blob_name):
    blob = container.get_blob_client(blob_name)
    data = blob.download_blob().readall().decode("utf-8", errors="replace")
    return json.loads(data)

# Replace with the EXACT names printed in step 1:
excel_name = "Attributes definition and types.xlsx"     # or whatever repr showed
csv_name   = "codification_co_fmcg.csv"
json_name  = "dictionary_co_fmcg_cross.json"

df_xlsx = read_excel_blob(container, excel_name)
df_csv  = read_csv_blob(container, csv_name)
data_js = read_json_blob(container, json_name)

print(df_xlsx.head(), df_csv.head(), type(data_js))




from urllib.parse import quote
from azure.storage.blob import BlobClient

account_name = svc.account_name
blob_path = quote(excel_name, safe="~()*!.'")  # encode spaces & specials in path
blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}"

# Use the connection string’s creds automatically:
excel_client = BlobClient.from_blob_url(blob_url, credential=svc.credential)
xlsx_bytes = excel_client.download_blob().readall()
pd.read_excel(BytesIO(xlsx_bytes), engine="openpyxl").head()

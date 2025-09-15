from azure.storage.blob import BlobServiceClient
import re

conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"

svc = BlobServiceClient.from_connection_string(conn_str)
container = svc.get_container_client(container_name)

print("Account name (parsed):", svc.account_name)   # must be lowercase, 3–24 chars, letters+digits only
print("Service URL repr:    ", repr(svc.url))       # look for spaces/newlines
print("Container URL repr:  ", repr(container.url)) # look for spaces/newlines

# quick validators
acct_ok = bool(re.fullmatch(r"[a-z0-9]{3,24}", svc.account_name or ""))
print("Account name valid?  ", acct_ok)





from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential

account_name = "<youraccount>"          # all lowercase, no spaces/underscores
account_key  = "<yourkey=="             # from Access keys
credential   = AzureNamedKeyCredential(account_name, account_key)

account_url = f"https://{account_name}.blob.core.windows.net"
svc = BlobServiceClient(account_url=account_url, credential=credential)
container = svc.get_container_client("rawdata")

print("Sanity list:")
for b in container.list_blobs():
    print("-", b.name)





from io import BytesIO, StringIO
import pandas as pd, json

# CSV
csv_bytes = container.get_blob_client("codification_co_fmcg.csv").download_blob().readall()
df_csv = pd.read_csv(StringIO(csv_bytes.decode("utf-8", errors="replace")))

# Excel (note the spaces; pass the name exactly as it appears)
xls_bytes = container.get_blob_client("Attributes definition and types.xlsx").download_blob().readall()
df_xls = pd.read_excel(BytesIO(xls_bytes), engine="openpyxl")

# JSON
js_bytes = container.get_blob_client("dictionary_co_fmcg_cross.json").download_blob().readall()
data_js = json.loads(js_bytes.decode("utf-8", errors="replace"))

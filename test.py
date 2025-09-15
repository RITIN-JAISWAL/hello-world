from azure.identity import InteractiveBrowserCredential
from azure.storage.blob import BlobClient
import pandas as pd, io, json

account   = "productcodingstorage"
container = "rawdata"
cred = InteractiveBrowserCredential()   # will prompt you to sign in

def read_csv(name):
    url = f"https://{account}.blob.core.windows.net/{container}/{name}"
    data = BlobClient.from_blob_url(url, credential=cred).download_blob().readall()
    return pd.read_csv(io.BytesIO(data))

def read_excel(name, **kw):
    url = f"https://{account}.blob.core.windows.net/{container}/{name}"
    data = BlobClient.from_blob_url(url, credential=cred).download_blob().readall()
    return pd.read_excel(io.BytesIO(data), **kw)

def read_json(name):
    url = f"https://{account}.blob.core.windows.net/{container}/{name}"
    data = BlobClient.from_blob_url(url, credential=cred).download_blob().readall()
    return json.loads(data)

df_csv  = read_csv("codification_co_fmcg.csv")
df_xlsx = read_excel("Attributes definition and types4.xlsx")
jobj    = read_json("dictionary_co_fmcg_cross.json")

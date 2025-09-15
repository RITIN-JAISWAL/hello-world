from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd, io, json

account   = "productcodingstorage"
container = "rawdata"
cred = ManagedIdentityCredential()

# List blobs (quick permission test)
svc = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=cred)
print([b.name for b in svc.get_container_client(container).list_blobs()][:10])

# Read one file into memory (no local save)
url = f"https://{account}.blob.core.windows.net/{container}/codification_co_fmcg.csv"
blob = BlobClient.from_blob_url(url, credential=cred)
df = pd.read_csv(io.BytesIO(blob.download_blob().readall()))
print(df.head())

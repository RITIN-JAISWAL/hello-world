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

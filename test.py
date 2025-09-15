from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd
import io, json, os

# Storage details
storage_account = "productcodingstorage"
container = "rawdata"

# Authenticate (uses AML compute's managed identity automatically)
credential = DefaultAzureCredential()
service_client = BlobServiceClient(
    f"https://{storage_account}.blob.core.windows.net",
    credential=credential
)
container_client = service_client.get_container_client(container)

def load_blob_file(blob_name: str):
    """Detect file type by extension and load from blob directly into memory."""
    blob = container_client.get_blob_client(blob_name)
    data = blob.download_blob().readall()
    
    ext = os.path.splitext(blob_name)[1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(data))
        return ("csv", df)
    
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(io.BytesIO(data))
        return ("excel", df)
    
    elif ext == ".json":
        js = json.loads(data)
        return ("json", js)
    
    else:
        return ("unknown", data)

# --- Auto-list all blobs in the rawdata/ folder ---
files = [b.name for b in container_client.list_blobs() if b.name.startswith("rawdata/")]

print("Files found in rawdata/:")
for f in files:
    print(" -", f)

# --- Auto-load everything ---
results = {}
for f in files:
    fname = f.split("/")[-1]  # strip 'rawdata/' prefix for dict key
    ftype, content = load_blob_file(f)
    results[fname] = (ftype, content)
    print(f"✔ Loaded {fname} as {ftype}")

# --- Preview some outputs ---
for fname, (ftype, content) in results.items():
    print(f"\nPreview of {fname} ({ftype}):")
    if ftype in ["csv", "excel"]:
        print(content.head())
    elif ftype == "json":
        print(str(content)[:300])

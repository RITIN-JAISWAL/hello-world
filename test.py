# --- Register Blob Files as Data Assets (auto-detect AML workspace) ---

# Imports
from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from urllib.parse import quote
from pathlib import Path
import shutil

# 1) Auto-detect AML workspace (no hardcoding needed)
ws = Workspace.from_config()
subscription_id = ws.subscription_id
resource_group = ws.resource_group
workspace_name = ws.name

print("Subscription ID:", subscription_id)
print("Resource Group :", resource_group)
print("Workspace Name :", workspace_name)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# 2) CONFIG — edit these two if your datastore/path differs
DATASTORE_NAME = "workspaceblobstore"   # <- change if your files are in a different AML datastore
PATH_PREFIX   = ""                      # e.g. "product-coding/" if your files are under a folder

FILES = [
    "codification_co_fmcg.csv",
    "dictionary_co_fmcg_cross.json",
    "Attributes definition and types4.xlsx",
]

# 3) Helpers
def datastore_uri(path: str) -> str:
    """Build azureml://datastores/<ds>/paths/<path> with proper quoting (spaces etc.)."""
    return f"azureml://datastores/{DATASTORE_NAME}/paths/" + quote(path, safe="/._-() ")

def make_mltable_spec(asset_name: str, target_uri: str) -> Path:
    """
    Create a temporary folder containing an MLTable file that references target_uri.
    Return the folder path.
    """
    base = Path("/mnt/data/_mltable_specs") / asset_name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    # Minimal MLTable: just point to the file path
    (base / "MLTable").write_text(f"paths:\n  - file: {target_uri}\n", encoding="utf-8")
    return base

# 4) Register assets
registered = []
for fname in FILES:
    path_in_container = f"{PATH_PREFIX}{fname}" if PATH_PREFIX else fname
    uri = datastore_uri(path_in_container)
    name = fname.lower().replace(" ", "_").replace(".", "_")

    if fname.lower().endswith((".csv", ".json")):
        # Use MLTable for tabular formats so you can Preview in AML
        spec_dir = make_mltable_spec(name, uri)
        asset = Data(
            name=name,
            description=f"Auto-registered MLTable from {uri}",
            type="mltable",
            path=str(spec_dir)
        )
    else:
        # Excel: register as uri_file (you can open in Data Wrangler or download)
        asset = Data(
            name=name,
            description=f"Auto-registered file asset from {uri}",
            type="uri_file",
            path=uri
        )

    created = ml_client.data.create_or_update(asset)
    registered.append((created.name, created.version, created.type))
    print(f"✔ Registered: {created.name} (v{created.version}, type {created.type})")

registered



import pandas as pd
from azure.ai.ml import dsl

# Replace with the printed name if different
asset_name = "codification_co_fmcg_csv"

# Resolve the latest version
asset = ml_client.data.get(name=asset_name)
print(asset.path)

# Directly read via pandas when the path is an MLTable spec folder you just created:
df = pd.read_csv(ml_client.path_to_url(asset.path).replace("MLTable", "").rstrip("/") + "/*", on_bad_lines="skip")
df.head()

from io import StringIO
import pandas as pd

csv_bytes = container.get_blob_client("codification_co_fmcg.csv").download_blob().readall()

# Try robust parsing (lets pandas sniff the delimiter/quoting and skip bad rows)
csv_text = csv_bytes.decode("utf-8", errors="replace")
df_csv = pd.read_csv(
    StringIO(csv_text),
    engine="python",          # more forgiving than the C engine
    sep=None,                 # auto-detect delimiter
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip',      # or "warn" if you want to see which lines were skipped
    dtype=str                 # read all as strings first
)
print(df_csv.shape)
df_csv.head()



# in a notebook cell
%pip install --upgrade openpyxl

import openpyxl, sys
print(openpyxl.__version__, sys.executable)  # confirms version and env


from io import BytesIO
xlsx_bytes = container.get_blob_client("Attributes definition and types.xlsx").download_blob().readall()
df_xlsx = pd.read_excel(BytesIO(xlsx_bytes), engine="openpyxl")
df_xlsx.head()




from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import pandas as pd, json

conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container_name = "rawdata"
svc = BlobServiceClient.from_connection_string(conn_str)
container = svc.get_container_client(container_name)

# CSV (robust)
csv_text = container.get_blob_client("codification_co_fmcg.csv").download_blob().readall().decode("utf-8", "replace")
df_csv = pd.read_csv(StringIO(csv_text), engine="python", sep=None, quotechar='"', escapechar='\\', on_bad_lines='skip', dtype=str)

# Excel
xlsx_bytes = container.get_blob_client("Attributes definition and types.xlsx").download_blob().readall()
df_xlsx = pd.read_excel(BytesIO(xlsx_bytes), engine="openpyxl")

# JSON
json_bytes = container.get_blob_client("dictionary_co_fmcg_cross.json").download_blob().readall()
data_json = json.loads(json_bytes.decode("utf-8", "replace"))

print("CSV shape:", df_csv.shape)
print("Excel shape:", df_xlsx.shape)
print("JSON type:", type(data_json).__name__)

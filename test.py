import os, io, re, json, argparse, textwrap
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient

# ---------------- Azure helpers ----------------
def get_blob_client(container: str, blob: str):
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set.")
    svc = BlobServiceClient.from_connection_string(conn)
    return svc.get_blob_client(container=container, blob=blob)

def download_blob_bytes(container: str, blob: str) -> bytes:
    return get_blob_client(container, blob).download_blob().readall()

def upload_blob_text(container: str, blob: str, text: str):
    get_blob_client(container, blob).upload_blob(text.encode("utf-8"), overwrite=True)

def upload_blob_bytes(container: str, blob: str, data: bytes):
    get_blob_client(container, blob).upload_blob(data, overwrite=True)

# ---------------- Generic utils ----------------
def safe_read_csv(bytes_data: bytes, dtype=None):
    buf = io.BytesIO(bytes_data)
    try:
        return pd.read_csv(buf, dtype=dtype)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, sep=";", dtype=dtype)

def ensure_cols(df: pd.DataFrame, rename_map: dict):
    """Rename columns if close alternatives exist."""
    for target, alts in rename_map.items():
        if target in df.columns: 
            continue
        for a in alts:
            if a in df.columns:
                df.rename(columns={a: target}, inplace=True)
                break
    return df

def histplot(series: pd.Series, title: str, path_png: str, bins=50):
    plt.figure()
    series = series.dropna()
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    upload_blob_bytes(ARGS.container, f"{ARGS.out_prefix}{path_png}", buf.getvalue())

def barplot(index_vals, counts, title, path_png, rotate=45):
    plt.figure()
    positions = np.arange(len(index_vals))
    plt.bar(positions, counts)
    plt.xticks(positions, index_vals, rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel("count")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    upload_blob_bytes(ARGS.container, f"{ARGS.out_prefix}{path_png}", buf.getvalue())

# ---------------- Quantity parsing ----------------
SIZE_PATTERNS = [
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>kg|g|gr|gramos|l|lt|litro|ml|mL)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>oz|lb)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?x\s?(?P<pack>\d+)\b',
]

def extract_qty(text: str):
    if not isinstance(text, str):
        return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})
    t = text.replace(",", ".")
    for pat in SIZE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m and m.groupdict().get("qty") and m.groupdict().get("unit"):
            try:
                return pd.Series({"qty_value": float(m.group("qty")), "qty_unit": m.group("unit").lower()})
            except ValueError:
                pass
    return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})

def normalize_units(q, u):
    if pd.isna(q) or not isinstance(u, str): 
        return pd.NA, pd.NA
    u = u.lower()
    if u in ["g", "gr", "gramos"]: return round(q,3), "g"
    if u == "kg":                   return round(q*1000,3), "g"
    if u in ["ml"]:                 return round(q,3), "ml"
    if u in ["l","lt","litro"]:     return round(q*1000,3), "ml"
    if u == "oz":                   return round(q*28.3495,3), "g"
    if u == "lb":                   return round(q*453.592,3), "g"
    return pd.NA, pd.NA

# ---------------- Dictionary parsing ----------------
def find_values_list(entry: dict) -> list[dict]:
    for k in ["b","values","codes","items","options","map"]:
        if k in entry and isinstance(entry[k], list):
            return entry[k]
    for v in entry.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []

def pick_code(d: dict):
    for k in ["c","code","id","k","key"]:
        if k in d: return str(d[k])
    return None

def pick_value(d: dict):
    for k in ["v","value","label","name","text"]:
        if k in d: return str(d[k])
    return None

# ---------------- Core analyses ----------------
def analyze_ocr(df: pd.DataFrame):
    df = ensure_cols(df, {"product_id":["prod_id","id","productid"], "ocr_text":["text","ocr","ocrstring"]})
    assert "product_id" in df.columns, "OCR: product_id column missing"
    if "ocr_text" not in df.columns:
        df["ocr_text"] = ""

    # basic stats
    n_rows = len(df)
    n_prods = df["product_id"].nunique()
    dup_pids = df.duplicated("product_id").sum()
    empty_txt = df["ocr_text"].isna().sum() + (df["ocr_text"].astype(str).str.strip()=="").sum()
    df["text_len"] = df["ocr_text"].fillna("").astype(str).str.len()

    # multi OCR per product
    per_pid = df.groupby("product_id")["ocr_text"].nunique().reset_index(name="n_unique_texts")
    conflicts = per_pid[per_pid["n_unique_texts"]>1]

    # common duplicate texts
    top_texts = df["ocr_text"].value_counts().head(20).reset_index()
    top_texts.columns = ["ocr_text","count"]

    # qty parsing
    qty = df["ocr_text"].apply(extract_qty)
    df = pd.concat([df, qty], axis=1)
    std_vals = df.apply(lambda r: normalize_units(r["qty_value"], r["qty_unit"]), axis=1, result_type="expand")
    std_vals.columns = ["qty_std_value","qty_std_unit"]
    df = pd.concat([df, std_vals], axis=1)

    # plots
    histplot(df["text_len"], "OCR text length", "ocr_text_length.png")
    histplot(df["qty_std_value"], "Parsed quantity (std units)", "ocr_qty_std.png")

    # issues table
    issues = df[(df["ocr_text"].isna()) | (df["ocr_text"].astype(str).str.strip()=="") | (df["text_len"]<10)]
    return {
        "df": df,
        "summary": {
            "rows": n_rows, "unique_products": n_prods, "dup_product_ids": int(dup_pids),
            "empty_or_blank_texts": int(empty_txt),
            "pct_empty_texts": round(100*empty_txt/max(1,n_rows),2)
        },
        "conflicts": conflicts,
        "top_texts": top_texts,
        "issues": issues
    }

def explode_attributes(attr_str: str):
    if not isinstance(attr_str, str) or not attr_str.strip(): return []
    pairs=[]
    for token in re.split(r'[;,\|]+', attr_str):
        token=token.strip()
        if ":" in token:
            k,v=token.split(":",1)
            pairs.append((k.strip(), v.strip()))
    return pairs

def analyze_master(df: pd.DataFrame):
    df = ensure_cols(df, {
        "product_id":["prod_id","id","productid"],
        "description":["desc","product_desc","name"],
        "attributes_raw":["attributes","attr_map","attr_codes"]
    })
    assert "product_id" in df.columns, "Master: product_id column missing"
    if "attributes_raw" not in df.columns: df["attributes_raw"] = ""

    n_rows = len(df)
    n_prods = df["product_id"].nunique()
    dup_prod = df.duplicated("product_id").sum()

    # explode to long
    rows=[]
    invalid_tokens=[]
    for _, r in df.iterrows():
        pid = r["product_id"]
        raw = str(r.get("attributes_raw",""))
        if not raw.strip(): 
            continue
        for t in re.split(r'[;,\|]+', raw):
            t=t.strip()
            if not t: 
                continue
            if ":" not in t:
                invalid_tokens.append({"product_id":pid, "token":t})
                continue
            aid, val = t.split(":",1)
            rows.append({"product_id":pid, "attr_id":aid.strip(), "value_code":val.strip()})
    long = pd.DataFrame(rows)
    invalid = pd.DataFrame(invalid_tokens)

    # conflicts: same product & attr with multiple values
    if not long.empty:
        dup = long.groupby(["product_id","attr_id"])["value_code"].nunique().reset_index()
        conflicts = dup[dup["value_code"]>1]
        attr_density = long.groupby("product_id")["attr_id"].nunique().describe()
        top_attrs = long["attr_id"].value_counts().head(25).reset_index()
        top_attrs.columns=["attr_id","count"]
        barplot(top_attrs["attr_id"].tolist(), top_attrs["count"].tolist(), 
                "Top attributes frequency", "master_top_attrs.png", rotate=90)
    else:
        conflicts = pd.DataFrame(columns=["product_id","attr_id","value_code"])
        attr_density = pd.Series(dtype=float)
        top_attrs = pd.DataFrame(columns=["attr_id","count"])

    return {
        "df": df,
        "long": long,
        "invalid_tokens": invalid,
        "summary": {
            "rows": n_rows, "unique_products": n_prods, "dup_product_ids": int(dup_prod),
            "exploded_rows": int(len(long)), "products_with_attrs": int(long["product_id"].nunique() if not long.empty else 0)
        },
        "conflicts": conflicts,
        "attr_density": attr_density,
        "top_attrs": top_attrs
    }

def analyze_dictionary(J: dict):
    entries = J.get("dict", [])
    attr_name = {}
    value_map = {}
    dict_conflicts = []

    seen_attr = Counter()
    for e in entries:
        aid = e.get("id")
        if not aid: 
            continue
        seen_attr[aid]+=1
        aname = e.get("sl") or e.get("name") or e.get("label")
        if aname: attr_name[aid]=aname
        items = find_values_list(e)
        local = {}
        for it in items:
            code = pick_code(it)
            val  = pick_value(it)
            if code is None or val is None: 
                continue
            # conflicting duplicate inside same attribute
            if code in local and local[code] != val:
                dict_conflicts.append({"attr_id":aid, "value_code":code, "label_a":local[code], "label_b":val, "type":"intra-attr duplicate"})
            local[code]=val
            # global conflict across entries
            if (aid, str(code)) in value_map and value_map[(aid,str(code))]!=val:
                dict_conflicts.append({"attr_id":aid, "value_code":code, "label_a":value_map[(aid,str(code))], "label_b":val, "type":"cross-entry duplicate"})
            value_map[(aid, str(code))]=val

    dup_attr_defs = [ {"attr_id":k, "definitions":v} for k,v in seen_attr.items() if v>1 ]
    return {
        "attr_name": attr_name,
        "value_map": value_map,
        "conflicts": pd.DataFrame(dict_conflicts),
        "dup_attr_defs": pd.DataFrame(dup_attr_defs)
    }

def analyze_joins(ocr, master_long, value_map):
    # coverage between OCR and Master
    ocr_p = set(ocr["product_id"].astype(str))
    mst_p = set(master_long["product_id"].astype(str)) if not master_long.empty else set()
    only_ocr = len(ocr_p - mst_p)
    only_mst = len(mst_p - ocr_p)
    both = len(ocr_p & mst_p)

    join_cov = pd.DataFrame([{
        "products_in_ocr": len(ocr_p),
        "products_in_master": len(mst_p),
        "products_in_both": both,
        "products_only_ocr": only_ocr,
        "products_only_master": only_mst,
        "pct_ocr_in_master": round(100*both/max(1,len(ocr_p)),2),
        "pct_master_in_ocr": round(100*both/max(1,len(mst_p)),2),
    }])

    # mapping coverage
    if not master_long.empty:
        master_long = master_long.copy()
        master_long["has_label"] = master_long.apply(lambda r: (r["attr_id"], str(r["value_code"])) in value_map, axis=1)
        missing = master_long[~master_long["has_label"]]
        by_attr = (missing.groupby("attr_id").size().sort_values(ascending=False).head(30).reset_index(name="missing_pairs"))
        if not by_attr.empty:
            barplot(by_attr["attr_id"].tolist(), by_attr["missing_pairs"].tolist(), 
                    "Top attributes with missing dictionary pairs", "missing_pairs_by_attr.png", rotate=90)
    else:
        missing = pd.DataFrame(columns=master_long.columns.tolist() + ["has_label"]) if isinstance(master_long, pd.DataFrame) else pd.DataFrame()
        by_attr = pd.DataFrame(columns=["attr_id","missing_pairs"])

    return join_cov, missing, by_attr

# ---------------- Scorecard ----------------
def build_scorecard(ocr_summ, master_summ, join_cov, missing_df, dict_conflicts):
    # naive scores (0-100). Tune thresholds as needed.
    completeness = 100 - ocr_summ["pct_empty_texts"]
    uniqueness = 100 - (100 * master_summ["dup_product_ids"] / max(1, master_summ["rows"]))
    validity = 100 - (100 * len(missing_df) / max(1, master_summ["exploded_rows"]))
    consistency = 100 - (100 * len(dict_conflicts) / max(1, len(dict_conflicts)+master_summ["exploded_rows"]))

    scorecard = pd.DataFrame([
        {"dimension":"Completeness (OCR text present)", "score": round(completeness,2)},
        {"dimension":"Uniqueness (Master product_id)", "score": round(uniqueness,2)},
        {"dimension":"Validity (codes mapped to labels)", "score": round(validity,2)},
        {"dimension":"Consistency (dictionary conflicts low)", "score": round(consistency,2)},
        {"dimension":"Join coverage (% OCR in Master)", "score": float(join_cov["pct_ocr_in_master"].iloc[0])},
        {"dimension":"Join coverage (% Master in OCR)", "score": float(join_cov["pct_master_in_ocr"].iloc[0])},
    ])
    scorecard["grade"] = pd.cut(scorecard["score"],
                                bins=[-1,60,75,90,100],
                                labels=["Poor","Fair","Good","Excellent"])
    return scorecard

# ---------------- Main ----------------
def main(container, ocr_blob, master_blob, dict_blob, out_prefix):
    # read
    ocr = safe_read_csv(download_blob_bytes(container, ocr_blob), dtype={"product_id":str})
    mst = safe_read_csv(download_blob_bytes(container, master_blob), dtype={"product_id":str})
    J = json.loads(download_blob_bytes(container, dict_blob).decode("utf-8", errors="ignore"))

    # analyses
    ocr_res = analyze_ocr(ocr)
    mst_res = analyze_master(mst)
    dic_res = analyze_dictionary(J)
    join_cov, missing_pairs, missing_by_attr = analyze_joins(ocr_res["df"], mst_res["long"], dic_res["value_map"])

    # scorecard
    scorecard = build_scorecard(ocr_res["summary"], mst_res["summary"], join_cov, missing_pairs, dic_res["conflicts"])

    # uploads
    upload_blob_text(container, f"{out_prefix}summary_scorecard.csv", scorecard.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}ocr_issues.csv", ocr_res["issues"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}ocr_top_texts.csv", ocr_res["top_texts"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_attr_long.csv", mst_res["long"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_conflicts.csv", mst_res["conflicts"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_invalid_tokens.csv", mst_res["invalid_tokens"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}dictionary_conflicts.csv", dic_res["conflicts"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}dictionary_duplicate_attr_defs.csv", dic_res["dup_attr_defs"].to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}mapping_missing.csv", missing_pairs.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}join_coverage.csv", join_cov.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}missing_by_attr.csv", missing_by_attr.to_csv(index=False))

    # mini Markdown report
    md = f"""
# Data Quality & EDA Report

**Container**: `{container}`  
**Inputs**: `{ocr_blob}`, `{master_blob}`, `{dict_blob}`  
**Outputs prefix**: `{out_prefix}`

## Executive Scorecard
(see `summary_scorecard.csv`)

- OCR rows: **{ocr_res['summary']['rows']}**, unique products: **{ocr_res['summary']['unique_products']}**, empty OCR texts: **{ocr_res['summary']['empty_or_blank_texts']}** ({ocr_res['summary']['pct_empty_texts']}%)
- Master rows: **{mst_res['summary']['rows']}**, unique products: **{mst_res['summary']['unique_products']}**, duplicate product_ids: **{mst_res['summary']['dup_product_ids']}**
- Exploded attribute pairs: **{mst_res['summary']['exploded_rows']}**
- Products with attributes: **{mst_res['summary']['products_with_attrs']}**

## Key Charts
- `ocr_text_length.png` – OCR text length distribution  
- `ocr_qty_std.png` – Parsed quantity (standard units)  
- `master_top_attrs.png` – Top attributes by frequency  
- `missing_pairs_by_attr.png` – Attributes with most missing dictionary pairs

## High-Priority Issue Tables
- `ocr_issues.csv` – blank/short OCR texts  
- `master_conflicts.csv` – same product + attr with multiple values  
- `mapping_missing.csv` – master pairs without labels (dictionary coverage gaps)  
- `dictionary_conflicts.csv` – inconsistent code→label definitions

## Notes
- Keep both **raw codes** and **labels** for audit.  
- Address missing OCR and conflicting attributes first; then close dictionary gaps.  
"""
    upload_blob_text(container, f"{out_prefix}README.md", textwrap.dedent(md).strip())

    print("✅ Finished. Report written to container under:", out_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--ocr_blob", required=True)
    parser.add_argument("--master_blob", required=True)
    parser.add_argument("--dict_blob", required=True)
    parser.add_argument("--out_prefix", default="quality_report/")
    ARGS = parser.parse_args()
    main(ARGS.container, ARGS.ocr_blob, ARGS.master_blob, ARGS.dict_blob, ARGS.out_prefix)


























import os, io, re, json, argparse, base64, textwrap, datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
from jinja2 import Template

# ---------------- Azure helpers ----------------
def get_blob_client(container: str, blob: str):
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set.")
    svc = BlobServiceClient.from_connection_string(conn)
    return svc.get_blob_client(container=container, blob=blob)

def download_blob_bytes(container: str, blob: str) -> bytes:
    return get_blob_client(container, blob).download_blob().readall()

def upload_blob_text(container: str, blob: str, text: str):
    get_blob_client(container, blob).upload_blob(text.encode("utf-8"), overwrite=True)

def upload_blob_bytes(container: str, blob: str, data: bytes):
    get_blob_client(container, blob).upload_blob(data, overwrite=True)

# ---------------- CSV / utils ----------------
def safe_read_csv(bytes_data: bytes, dtype=None):
    buf = io.BytesIO(bytes_data)
    try:
        return pd.read_csv(buf, dtype=dtype)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, sep=";", dtype=dtype)

def ensure_cols(df: pd.DataFrame, rename_map: dict):
    for target, alts in rename_map.items():
        if target in df.columns:
            continue
        for a in alts:
            if a in df.columns:
                df.rename(columns={a: target}, inplace=True)
                break
    return df

def fig_to_base64():
    b = io.BytesIO()
    plt.savefig(b, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    return base64.b64encode(b.getvalue()).decode("utf-8")

def hist_b64(series: pd.Series, title: str, bins=50, xlabel=None, ylabel="count"):
    plt.figure()
    series = series.dropna()
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel or title)
    plt.ylabel(ylabel)
    return fig_to_base64()

def bar_b64(labels, values, title, rotate=45, ylabel="count"):
    plt.figure()
    pos = np.arange(len(labels))
    plt.bar(pos, values)
    plt.xticks(pos, labels, rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    return fig_to_base64()

# ---------------- Quantity parsing ----------------
SIZE_PATTERNS = [
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>kg|g|gr|gramos|l|lt|litro|ml|mL)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?(?P<unit>oz|lb)\b',
    r'(?P<qty>\d+[.,]?\d*)\s?x\s?(?P<pack>\d+)\b',
]
def extract_qty(text: str):
    if not isinstance(text, str):
        return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})
    t = text.replace(",", ".")
    for pat in SIZE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m and m.groupdict().get("qty") and m.groupdict().get("unit"):
            try:
                return pd.Series({"qty_value": float(m.group("qty")), "qty_unit": m.group("unit").lower()})
            except ValueError:
                pass
    return pd.Series({"qty_value": pd.NA, "qty_unit": pd.NA})

def normalize_units(q, u):
    if pd.isna(q) or not isinstance(u, str): 
        return pd.NA, pd.NA
    u = u.lower()
    if u in ["g", "gr", "gramos"]: return round(q,3), "g"
    if u == "kg":                   return round(q*1000,3), "g"
    if u in ["ml"]:                 return round(q,3), "ml"
    if u in ["l","lt","litro"]:     return round(q*1000,3), "ml"
    if u == "oz":                   return round(q*28.3495,3), "g"
    if u == "lb":                   return round(q*453.592,3), "g"
    return pd.NA, pd.NA

# ---------------- Dictionary parsing ----------------
def find_values_list(entry: dict) -> list[dict]:
    for k in ["b","values","codes","items","options","map"]:
        if k in entry and isinstance(entry[k], list):
            return entry[k]
    for v in entry.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []

def pick_code(d: dict):
    for k in ["c","code","id","k","key"]:
        if k in d: return str(d[k])
    return None

def pick_value(d: dict):
    for k in ["v","value","label","name","text"]:
        if k in d: return str(d[k])
    return None

# ---------------- Analyses ----------------
def analyze_ocr(df: pd.DataFrame):
    df = ensure_cols(df, {"product_id":["prod_id","id","productid"], "ocr_text":["text","ocr","ocrstring"]})
    assert "product_id" in df.columns, "OCR: product_id column missing"
    if "ocr_text" not in df.columns:
        df["ocr_text"] = ""

    n_rows = len(df)
    n_prods = df["product_id"].nunique()
    dup_pids = df.duplicated("product_id").sum()
    empty_txt = df["ocr_text"].isna().sum() + (df["ocr_text"].astype(str).str.strip()=="").sum()
    df["text_len"] = df["ocr_text"].fillna("").astype(str).str.len()

    per_pid = df.groupby("product_id")["ocr_text"].nunique().reset_index(name="n_unique_texts")
    conflicts = per_pid[per_pid["n_unique_texts"]>1]
    top_texts = df["ocr_text"].value_counts().head(20).reset_index()
    top_texts.columns = ["ocr_text","count"]

    qty = df["ocr_text"].apply(extract_qty)
    df = pd.concat([df, qty], axis=1)
    std_vals = df.apply(lambda r: normalize_units(r["qty_value"], r["qty_unit"]), axis=1, result_type="expand")
    std_vals.columns = ["qty_std_value","qty_std_unit"]
    df = pd.concat([df, std_vals], axis=1)

    charts = {
        "text_len": hist_b64(df["text_len"], "OCR text length"),
        "qty_std": hist_b64(df["qty_std_value"], "Parsed quantity (std units)")
    }

    issues = df[(df["ocr_text"].isna()) | (df["ocr_text"].astype(str).str.strip()=="") | (df["text_len"]<10)]

    summary = {
        "rows": n_rows, "unique_products": n_prods, "dup_product_ids": int(dup_pids),
        "empty_or_blank_texts": int(empty_txt),
        "pct_empty_texts": round(100*empty_txt/max(1,n_rows),2)
    }
    return df, conflicts, top_texts, issues, charts, summary

def explode_attributes(attr_str: str):
    if not isinstance(attr_str, str) or not attr_str.strip(): return []
    pairs=[]
    for token in re.split(r'[;,\|]+', attr_str):
        token=token.strip()
        if ":" in token:
            k,v=token.split(":",1)
            pairs.append((k.strip(), v.strip()))
    return pairs

def analyze_master(df: pd.DataFrame):
    df = ensure_cols(df, {
        "product_id":["prod_id","id","productid"],
        "description":["desc","product_desc","name"],
        "attributes_raw":["attributes","attr_map","attr_codes"]
    })
    assert "product_id" in df.columns, "Master: product_id column missing"
    if "attributes_raw" not in df.columns: df["attributes_raw"] = ""

    n_rows = len(df)
    n_prods = df["product_id"].nunique()
    dup_prod = df.duplicated("product_id").sum()

    rows=[]; invalid_tokens=[]
    for _, r in df.iterrows():
        pid = r["product_id"]
        raw = str(r.get("attributes_raw",""))
        if not raw.strip(): 
            continue
        for t in re.split(r'[;,\|]+', raw):
            t=t.strip()
            if not t: 
                continue
            if ":" not in t:
                invalid_tokens.append({"product_id":pid, "token":t})
                continue
            aid, val = t.split(":",1)
            rows.append({"product_id":pid, "attr_id":aid.strip(), "value_code":val.strip()})
    long = pd.DataFrame(rows)
    invalid = pd.DataFrame(invalid_tokens)

    if not long.empty:
        dup = long.groupby(["product_id","attr_id"])["value_code"].nunique().reset_index()
        conflicts = dup[dup["value_code"]>1]
        top_attrs = long["attr_id"].value_counts().head(25).reset_index(names=["attr_id","count"])
        chart_top_attrs = bar_b64(top_attrs["attr_id"].tolist(), top_attrs["count"].tolist(), "Top attributes frequency", rotate=90)
        attr_density_desc = long.groupby("product_id")["attr_id"].nunique().describe()
    else:
        conflicts = pd.DataFrame(columns=["product_id","attr_id","value_code"])
        chart_top_attrs = None
        attr_density_desc = pd.Series(dtype=float)

    summary = {
        "rows": n_rows, "unique_products": n_prods, "dup_product_ids": int(dup_prod),
        "exploded_rows": int(len(long)), "products_with_attrs": int(long["product_id"].nunique() if not long.empty else 0)
    }
    return df, long, invalid, conflicts, chart_top_attrs, attr_density_desc, summary

def analyze_dictionary(J: dict):
    entries = J.get("dict", [])
    attr_name = {}
    value_map = {}
    dict_conflicts = []
    seen_attr = Counter()

    for e in entries:
        aid = e.get("id")
        if not aid: 
            continue
        seen_attr[aid]+=1
        aname = e.get("sl") or e.get("name") or e.get("label")
        if aname: attr_name[aid]=aname
        items = find_values_list(e)
        local = {}
        for it in items:
            code = pick_code(it)
            val  = pick_value(it)
            if code is None or val is None: 
                continue
            if code in local and local[code] != val:
                dict_conflicts.append({"attr_id":aid, "value_code":code, "label_a":local[code], "label_b":val, "type":"intra-attr duplicate"})
            local[code]=val
            if (aid, str(code)) in value_map and value_map[(aid,str(code))]!=val:
                dict_conflicts.append({"attr_id":aid, "value_code":code, "label_a":value_map[(aid,str(code))], "label_b":val, "type":"cross-entry duplicate"})
            value_map[(aid, str(code))]=val

    dup_attr_defs = [ {"attr_id":k, "definitions":v} for k,v in seen_attr.items() if v>1 ]
    return attr_name, value_map, pd.DataFrame(dict_conflicts), pd.DataFrame(dup_attr_defs)

def analyze_joins(ocr_df, master_long, value_map):
    ocr_p = set(ocr_df["product_id"].astype(str))
    mst_p = set(master_long["product_id"].astype(str)) if not master_long.empty else set()
    only_ocr = len(ocr_p - mst_p); only_mst = len(mst_p - ocr_p); both = len(ocr_p & mst_p)

    join_cov = pd.DataFrame([{
        "products_in_ocr": len(ocr_p),
        "products_in_master": len(mst_p),
        "products_in_both": both,
        "products_only_ocr": only_ocr,
        "products_only_master": only_mst,
        "pct_ocr_in_master": round(100*both/max(1,len(ocr_p)),2),
        "pct_master_in_ocr": round(100*both/max(1,len(mst_p)),2),
    }])

    if not master_long.empty:
        master_long = master_long.copy()
        master_long["has_label"] = master_long.apply(lambda r: (r["attr_id"], str(r["value_code"])) in value_map, axis=1)
        missing = master_long[~master_long["has_label"]]
        by_attr = (missing.groupby("attr_id").size().sort_values(ascending=False).head(30).reset_index(name="missing_pairs"))
        chart_missing = None
        if not by_attr.empty:
            chart_missing = bar_b64(by_attr["attr_id"].tolist(), by_attr["missing_pairs"].tolist(),
                                    "Top attributes with missing dictionary pairs", rotate=90)
    else:
        missing = pd.DataFrame(columns=["product_id","attr_id","value_code","has_label"])
        by_attr = pd.DataFrame(columns=["attr_id","missing_pairs"])
        chart_missing = None

    return join_cov, missing, by_attr, chart_missing

def build_scorecard(ocr_summ, master_summ, join_cov, missing_df, dict_conflicts):
    completeness = 100 - ocr_summ["pct_empty_texts"]
    uniqueness = 100 - (100 * master_summ["dup_product_ids"] / max(1, master_summ["rows"]))
    validity = 100 - (100 * len(missing_df) / max(1, master_summ["exploded_rows"]))
    consistency = 100 - (100 * len(dict_conflicts) / max(1, len(dict_conflicts)+master_summ["exploded_rows"]))
    scorecard = pd.DataFrame([
        {"dimension":"Completeness (OCR text present)", "score": round(completeness,2)},
        {"dimension":"Uniqueness (Master product_id)", "score": round(uniqueness,2)},
        {"dimension":"Validity (codes→labels coverage)", "score": round(validity,2)},
        {"dimension":"Consistency (dictionary conflicts low)", "score": round(consistency,2)},
        {"dimension":"Join coverage (% OCR in Master)", "score": float(join_cov["pct_ocr_in_master"].iloc[0])},
        {"dimension":"Join coverage (% Master in OCR)", "score": float(join_cov["pct_master_in_ocr"].iloc[0])},
    ])
    scorecard["grade"] = pd.cut(scorecard["score"], bins=[-1,60,75,90,100], labels=["Poor","Fair","Good","Excellent"])
    return scorecard

# ---------------- HTML template ----------------
HTML_TMPL = Template("""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Data Quality & EDA Report</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:32px;line-height:1.4;color:#111}
 h1,h2{margin:0 0 12px}
 h3{margin:24px 0 8px}
 .kpi{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px;margin:12px 0 24px}
 .card{border:1px solid #eee;border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
 table{border-collapse:collapse;width:100%;margin:12px 0}
 th,td{border:1px solid #eaeaea;padding:6px 8px;text-align:left;font-size:13px}
 th{background:#fafafa}
 .muted{color:#666}
 img{max-width:100%}
 .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#f2f2f2;font-size:12px}
 .good{background:#e6f7ed} .warn{background:#fff7e6} .bad{background:#ffefef}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 footer{margin-top:32px;font-size:12px;color:#666}
</style>
</head>
<body>

<h1>Data Quality & EDA Report</h1>
<div class="muted">Generated: {{ now }} | Container: <b>{{ container }}</b> | Inputs: {{ ocr_blob }}, {{ master_blob }}, {{ dict_blob }}</div>

<h2>Executive Scorecard</h2>
<div class="kpi">
  {% for r in scorecard %}
  <div class="card">
    <div class="muted">{{ r.dimension }}</div>
    <div style="font-size:28px;font-weight:700">{{ "%.1f"|format(r.score) }}</div>
    <div class="pill {% if r.grade=='Excellent' %}good{% elif r.grade in ['Good','Fair'] %}warn{% else %}bad{% endif %}">{{ r.grade }}</div>
  </div>
  {% endfor %}
</div>

<h2>Source A: OCR</h2>
<div class="grid2">
  <div class="card">
    <h3>Text length</h3>
    <img src="data:image/png;base64,{{ ocr_charts.text_len }}">
  </div>
  <div class="card">
    <h3>Parsed quantity (standard units)</h3>
    <img src="data:image/png;base64,{{ ocr_charts.qty_std }}">
  </div>
</div>
<table>
<tr><th>Rows</th><th>Unique products</th><th>Duplicate product_ids</th><th>Empty/blank texts</th><th>% Empty</th></tr>
<tr>
<td>{{ ocr_summary.rows }}</td>
<td>{{ ocr_summary.unique_products }}</td>
<td>{{ ocr_summary.dup_product_ids }}</td>
<td>{{ ocr_summary.empty_or_blank_texts }}</td>
<td>{{ ocr_summary.pct_empty_texts }}%</td>
</tr>
</table>

<h3>Top repeated OCR texts (sanity check)</h3>
<table>
<tr><th>ocr_text</th><th>count</th></tr>
{% for _,r in ocr_top_texts.iterrows() %}
<tr><td class="muted">{{ r.ocr_text }}</td><td>{{ int(r['count']) }}</td></tr>
{% endfor %}
</table>

<h2>Source B: Masterfile</h2>
{% if master_chart_top_attrs %}
<div class="card">
  <h3>Most frequent attributes</h3>
  <img src="data:image/png;base64,{{ master_chart_top_attrs }}">
</div>
{% endif %}
<table>
<tr><th>Rows</th><th>Unique products</th><th>Duplicate product_ids</th><th>Exploded attribute pairs</th><th>Products with attributes</th></tr>
<tr>
<td>{{ master_summary.rows }}</td>
<td>{{ master_summary.unique_products }}</td>
<td>{{ master_summary.dup_product_ids }}</td>
<td>{{ master_summary.exploded_rows }}</td>
<td>{{ master_summary.products_with_attrs }}</td>
</tr>
</table>

<h3>Conflicts in masterfile</h3>
<div class="muted">Same product + attribute appearing with multiple values.</div>
<table>
<tr><th>product_id</th><th>attr_id</th><th>distinct_values</th></tr>
{% for _,r in master_conflicts.head(50).iterrows() %}
<tr><td>{{ r.product_id }}</td><td>{{ r.attr_id }}</td><td>{{ int(r.value_code) if r.value_code==r.value_code else r.value_code }}</td></tr>
{% endfor %}
</table>

<h2>Source C: Dictionary</h2>
<table>
<tr><th>Conflicts found</th><th>Duplicate attribute definitions</th></tr>
<tr><td>{{ len(dict_conflicts) }}</td><td>{{ len(dict_dup_defs) }}</td></tr>
</table>
{% if len(dict_conflicts)>0 %}
<h3>Example dictionary conflicts</h3>
<table>
<tr><th>attr_id</th><th>value_code</th><th>label_a</th><th>label_b</th><th>type</th></tr>
{% for _,r in dict_conflicts.head(50).iterrows() %}
<tr><td>{{ r.attr_id }}</td><td>{{ r.value_code }}</td><td>{{ r.label_a }}</td><td>{{ r.label_b }}</td><td>{{ r.type }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>Joins & Coverage</h2>
<table>
<tr><th>products_in_ocr</th><th>products_in_master</th><th>products_in_both</th><th>only_ocr</th><th>only_master</th><th>% ocr in master</th><th>% master in ocr</th></tr>
{% for _,r in join_cov.iterrows() %}
<tr>
<td>{{ int(r.products_in_ocr) }}</td>
<td>{{ int(r.products_in_master) }}</td>
<td>{{ int(r.products_in_both) }}</td>
<td>{{ int(r.products_only_ocr) }}</td>
<td>{{ int(r.products_only_master) }}</td>
<td>{{ r.pct_ocr_in_master }}%</td>
<td>{{ r.pct_master_in_ocr }}%</td>
</tr>
{% endfor %}
</table>

{% if missing_by_attr_chart %}
<div class="card">
  <h3>Attributes with most missing code→label mappings</h3>
  <img src="data:image/png;base64,{{ missing_by_attr_chart }}">
</div>
{% endif %}

<h3>Actions & Recommendations</h3>
<ul>
  <li><b>Close OCR gaps</b>: remove boilerplate duplicates and fix records with blank/short text (&lt;10 chars).</li>
  <li><b>Resolve master conflicts</b>: for each (product_id, attr_id) with multiple values, define a single source of truth.</li>
  <li><b>Patch dictionary coverage</b>: add missing (attr_id, value_code) pairs highlighted above.</li>
  <li><b>Standardize quantities</b>: rely on parsed <i>qty_std_value / qty_std_unit</i> (g/ml) for modeling.</li>
  <li><b>Data contracts</b>: enforce schema & valid ranges (e.g., attr IDs match ^A\\d{4}$, value codes numeric).</li>
</ul>

<footer>
Report generated automatically · {{ now }}
</footer>
</body>
</html>
""".strip())

# ---------------- Main ----------------
def main(container, ocr_blob, master_blob, dict_blob, out_prefix, report_name):
    # read
    ocr = safe_read_csv(download_blob_bytes(container, ocr_blob), dtype={"product_id":str})
    mst = safe_read_csv(download_blob_bytes(container, master_blob), dtype={"product_id":str})
    J = json.loads(download_blob_bytes(container, dict_blob).decode("utf-8", errors="ignore"))

    # analyses
    ocr_df, ocr_conflicts, ocr_top, ocr_issues, ocr_charts, ocr_sum = analyze_ocr(ocr)
    mst_df, mst_long, mst_invalid, mst_conflicts, master_chart_top_attrs, attr_density_desc, mst_sum = analyze_master(mst)
    attr_name, value_map, dict_conflicts, dict_dup_defs = analyze_dictionary(J)
    join_cov, missing_pairs, missing_by_attr, missing_by_attr_chart = analyze_joins(ocr_df, mst_long, value_map)

    # scorecard
    scorecard = build_scorecard(ocr_sum, mst_sum, join_cov, missing_pairs, dict_conflicts)

    # uploads of CSV artifacts
    upload_blob_text(container, f"{out_prefix}summary_scorecard.csv", scorecard.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}ocr_issues.csv", ocr_issues.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}ocr_top_texts.csv", ocr_top.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_attr_long.csv", mst_long.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_conflicts.csv", mst_conflicts.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}master_invalid_tokens.csv", mst_invalid.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}dictionary_conflicts.csv", dict_conflicts.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}dictionary_duplicate_attr_defs.csv", dict_dup_defs.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}mapping_missing.csv", missing_pairs.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}join_coverage.csv", join_cov.to_csv(index=False))
    upload_blob_text(container, f"{out_prefix}missing_by_attr.csv", missing_by_attr.to_csv(index=False))

    # render HTML
    html = HTML_TMPL.render(
        now=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        container=container, ocr_blob=ocr_blob, master_blob=master_blob, dict_blob=dict_blob,
        scorecard=scorecard.to_dict(orient="records"),
        ocr_charts=ocr_charts, ocr_summary=ocr_sum, ocr_top_texts=ocr_top,
        master_summary=mst_sum, master_conflicts=mst_conflicts,
        master_chart_top_attrs=master_chart_top_attrs,
        dict_conflicts=dict_conflicts, dict_dup_defs=dict_dup_defs,
        join_cov=join_cov, missing_by_attr_chart=missing_by_attr_chart
    )

    upload_blob_text(container, f"{out_prefix}{report_name}", html)
    print(f"✅ HTML report written to: {out_prefix}{report_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ocr_blob", required=True)
    ap.add_argument("--master_blob", required=True)
    ap.add_argument("--dict_blob", required=True)
    ap.add_argument("--out_prefix", default="quality_report/")
    ap.add_argument("--report_name", default="dq_report.html")
    args = ap.parse_args()
    main(args.container, args.ocr_blob, args.master_blob, args.dict_blob, args.out_prefix, args.report_name)

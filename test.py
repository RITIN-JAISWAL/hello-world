# -*- coding: utf-8 -*-
"""
FAST non-LLM pipeline (uses in-memory dataframes `Train` and `Test`)

Assumptions:
- `Train` and `Test` are pandas DataFrames already in memory.
- Train has columns: clean_description, Sector, Categoría, Marca, (optional) OCR_Size, OCR_Measure
- Test  has columns: ocr_text, (optional) Sector, Categoría, Marca, OCR_Size, OCR_Measure

What it does:
- Trains shared TF-IDF (char-grams) + LinearSVCs (global sector/category + sector-specific category)
- TEST inference from Test.ocr_text with len>20 filter
- TRAIN sanity-check inference from Train.clean_description with len>20 filter
- Handles binary LinearSVC decision_function with two-sided scores fix
- Extracts brand (dictionary + fuzzy) and quantity/unit (regex + priors)
"""

import os, re, numpy as np, pandas as pd
from typing import Any, Dict, List, Optional
from difflib import get_close_matches

# Set threads for Azure ML nodes (tweak to your vCPU count)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# Optional deps
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import csr_matrix

# ---------- Cleaning ----------
NOISE_PATTERNS = [
    r"exceso en [a-záéíóúüñ ]+",
    r"cont\.?\s*neto",
    r"lote[:\s]\S+",
    r"fecha\s*(de)?\s*(vencimiento|caducidad)[:\s]\S+",
    r"hecho en [a-záéíóúüñ ]+",
    r"importado por [a-záéíóúüñ ]+",
    r"\bwww\.\S+",
    r"\b@\S+",
    r"\b\d{1,2}\s*(oct|nov|dic|ene|feb|mar|abr|may|jun|jul|ago|sep)\b"
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.I)

def clean_ocr_for_inference(s: str) -> str:
    if not isinstance(s, str): return ""
    t = unidecode(s.lower())
    t = NOISE_RE.sub(" ", t)
    t = re.sub(r"[^a-z0-9%/.,\- ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def soft_clean(s: str) -> str:
    if not isinstance(s, str): return ""
    t = unidecode(s.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_combined_text(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str)
    a = raw.map(clean_ocr_for_inference)
    b = raw.map(soft_clean)
    return (a + " || " + b).str.strip()

# ---------- Size/Unit ----------
UNIT_MAP = {'gr':'g','g':'g','kg':'kg','ml':'ml','l':'L','lt':'L','l.':'L','un':'un','u':'un','oz':'oz','uds':'un'}
SIZE_RES = [re.compile(p, re.I) for p in [
    r'(?P<num>\d{1,4}(?:[.,]\d{1,3})?)\s*(?P<unit>kg|g|gr|ml|l|lt|l\.|oz|un|u|uds)\b',
    r'(?P<unit>kg|g|gr|ml|l|lt|l\.|oz|un|u|uds)\s*(?P<num>\d{1,4}(?:[.,]\d{1,3})?)\b',
    r'(?:x|\*)\s*(?P<num>\d{1,3})\s*(?P<unit>un|u|uds)\b',
    r'(?P<num>\d{1,3})\s*(?P<unit>un|u|uds)\b'
]]
KEY_NEAR_RE = re.compile(r'(cont|contenido|neto|peso|volumen)', re.I)

def extract_all_sizes(text: str):
    if not isinstance(text,str): return []
    cands = []
    for rx in SIZE_RES:
        for m in rx.finditer(text):
            num = m.groupdict().get('num')
            unit = m.group('unit').lower()
            if not num: continue
            try:
                val = float(num.replace(',', '.'))
            except: continue
            unit = UNIT_MAP.get(unit, unit)
            bias = 0.0
            if KEY_NEAR_RE.search(text[max(0,m.start()-30): m.end()+30]): bias += 1.0
            if unit in {'g','kg','ml','L','oz'}: bias += 0.5
            cands.append((val, unit, bias))
    return cands

def normalize_unit(u: Any) -> Any:
    if isinstance(u, str):
        return UNIT_MAP.get(u.strip().lower(), u.strip().lower())
    return u

# ---------- Brand tools ----------
def normalize_text(s: Any) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+"," ", s.strip().lower())

def strip_parentheses_brand(s: str) -> str:
    return re.sub(r'\s*\(.*?\)\s*', '', s or "", flags=re.S)

def build_brand_dictionary(train_df: pd.DataFrame):
    df = train_df.copy()
    df['brand_full'] = df['Marca'].astype(str)
    df['brand_main'] = df['Marca'].astype(str).apply(strip_parentheses_brand)
    pool = pd.concat([df['brand_full'], df['brand_main']]).dropna().unique().tolist()
    normalized = list({normalize_text(x) for x in pool if isinstance(x,str) and x.strip()})
    brand_sorted = sorted(normalized, key=len, reverse=True)
    canon_map = {}
    for b in brand_sorted:
        mask = df['Marca'].str.lower().str.contains(re.escape(b), regex=True, na=False)
        candidates = df.loc[mask, 'Marca']
        canon_map[b] = candidates.mode().iloc[0] if len(candidates) else b
    brand_main_list = sorted(df['brand_main'].dropna().unique().tolist(), key=len, reverse=True)
    return brand_sorted, canon_map, brand_main_list

CAPS_RE = re.compile(r'\b[A-Z][A-Z0-9ÁÉÍÓÚÜÑ\-]{2,}\b')

def find_brand_in_text_exact_or_caps(text: Any, brand_sorted, canon_map, brand_main_list) -> Optional[str]:
    if not isinstance(text, str) or not text.strip(): return None
    t_norm = normalize_text(text)
    for b in brand_sorted:
        if b and b in t_norm:
            return canon_map.get(b, b)
    caps = CAPS_RE.findall(text.upper())
    for token in caps:
        exact = [bm for bm in brand_main_list if bm.upper() == token]
        if exact: return exact[0]
    if caps:
        candidate = " ".join(caps)
        match_list = get_close_matches(candidate, [bm.upper() for bm in brand_main_list], n=1, cutoff=0.85)
        if match_list: return match_list[0].title()
    return None

def pick_brand_with_prior(text: str, brand_main_list, pred_cat, cat_brand_counts, threshold:int=82) -> Optional[str]:
    if not HAVE_RAPIDFUZZ or not isinstance(text,str) or not text.strip():
        return None
    pr = process.extract(text, brand_main_list, scorer=fuzz.partial_ratio, limit=10)
    ts = process.extract(text, brand_main_list, scorer=fuzz.token_set_ratio, limit=10)
    scores = {}
    for b, sc, _ in pr: scores[b] = max(scores.get(b,0), sc)
    for b, sc, _ in ts: scores[b] = max(scores.get(b,0), sc)
    cand = [(b, s) for b,s in scores.items() if s >= threshold]
    if not cand: return None
    pri = cat_brand_counts.get(pred_cat, {})
    BAD_TOKENS = (" s.a", "sa ", "industr", "compañ", "company", "fabric")
    def penalty(b): return -0.2 if any(t in b.lower() for t in BAD_TOKENS) else 0.0
    def score(b, s): return 0.6*(s/100.0) + 0.4*pri.get(b,0.0) + penalty(b)
    return max(cand, key=lambda x: score(x[0], x[1]))[0]

# ---------- Vectorizer (shared) ----------
def fit_char_vectorizer(X_text: List[str], min_df=3, max_features=250_000):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,6),
        min_df=min_df, max_features=max_features,
        dtype=np.float32
    )
    X = vec.fit_transform(X_text).astype(np.float32, copy=False)
    return vec, X

def transform_char_vectorizer(vec: TfidfVectorizer, texts: List[str]):
    X = vec.transform(texts).astype(np.float32, copy=False)
    return X

# ---------- Train global & sector models on SAME feature space ----------
def train_global_models(train_df: pd.DataFrame):
    tr_text = train_df["clean_description"].fillna("").astype(str)
    combined = build_combined_text(tr_text)

    vec, X = fit_char_vectorizer(combined.tolist(), min_df=3, max_features=250_000)

    y_cat = train_df["Categoría"].astype(str).tolist()
    y_sec = train_df["Sector"].astype(str).tolist()

    cat_clf = LinearSVC(C=5.0, class_weight="balanced", max_iter=6000, tol=2e-4)
    sec_clf = LinearSVC(C=3.0, class_weight="balanced", max_iter=5000, tol=2e-4)
    cat_clf.fit(X, y_cat)
    sec_clf.fit(X, y_sec)

    sec_cat_models = {}
    for sec, idxs in train_df.groupby("Sector").indices.items():
        sub = train_df.loc[idxs]
        if sub["Categoría"].nunique() < 2:
            continue
        y_sub = sub["Categoría"].astype(str).values
        X_sub = X[idxs]
        clf = LinearSVC(C=4.0, class_weight="balanced", max_iter=6000, tol=2e-4)
        clf.fit(X_sub, y_sub)
        sec_cat_models[sec] = clf

    cat2sec = train_df.groupby("Categoría")["Sector"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    sec2cats = (train_df.groupby("Sector")["Categoría"].apply(lambda s: sorted(s.unique())).to_dict())
    return vec, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats

# ---------- Category ensemble using shared X (with binary fix) ----------
CATEGORY_HINTS = {
    "Café": ["cafe","colcafe","instantaneo","molido"],
    "Chocolate Para Taza / Cocoa": ["chocolate","cocoa","mesa","taza"],
    "Jabón En Barra Para Ropa": ["jabon","jabón","ropa","barra","detergente"],
}

def _binary_two_sided_scores_batch(dec_values: np.ndarray) -> np.ndarray:
    d = np.ravel(dec_values).astype(float)
    return np.vstack([-d, d]).T  # (n_samples, 2)

def category_ensemble_predict_sharedX(
    X: csr_matrix,
    raw_soft_text: List[str],
    sec_pred: List[str],
    cat_clf: LinearSVC,
    sec_cat_models: Dict[str, LinearSVC],
    sec2cats: Dict[str, List[str]]
):
    gscores = cat_clf.decision_function(X)
    if gscores.ndim == 1: gscores = gscores.reshape(-1,1)
    gclasses = list(cat_clf.classes_)
    gindex = {c:i for i,c in enumerate(gclasses)}

    gmean = gscores.mean(axis=1, keepdims=True)
    gstd  = gscores.std(axis=1, keepdims=True) + 1e-6
    gstd_scores = (gscores - gmean)/gstd

    topk_idx = np.argsort(-gscores, axis=1)[:, :5]
    topk = [[gclasses[j] for j in row] for row in topk_idx]

    final = [None]*X.shape[0]
    sec_pred_arr = np.array(sec_pred)

    for sec, clf in sec_cat_models.items():
        rows = np.where(sec_pred_arr == sec)[0]
        if len(rows) == 0: continue
        Xs = X[rows]

        sdec = clf.decision_function(Xs)
        sclasses = list(clf.classes_)
        sindex = {c:j for j,c in enumerate(sclasses)}
        if np.ndim(sdec) == 1:
            sfull = _binary_two_sided_scores_batch(sdec)
        else:
            sfull = sdec

        smean = sfull.mean(axis=1, keepdims=True)
        sstd  = sfull.std(axis=1, keepdims=True) + 1e-6
        sstd_scores = (sfull - smean)/sstd

        allowed = set(sec2cats.get(sec, []))
        for k, i in enumerate(rows):
            best_c = None; best_val = -1e9
            for c in allowed:
                gv = gstd_scores[i, gindex[c]] if c in gindex else -1e9
                sv = sstd_scores[k, sindex[c]] if c in sindex else 0.0
                val = 0.6*gv + 0.4*sv
                if val > best_val:
                    best_val, best_c = val, c

            chosen = best_c if best_c is not None else gclasses[int(np.argmax(gscores[i]))]

            raw = raw_soft_text[i]
            hints_best, best_hits = chosen, -1
            for c in topk[i]:
                hits = sum(1 for h in CATEGORY_HINTS.get(c, []) if h in raw)
                if hits > best_hits:
                    hints_best, best_hits = c, hits

            final[i] = hints_best if hints_best in allowed else chosen

    for i in range(len(final)):
        if final[i] is not None:
            continue
        sec = sec_pred[i]
        allowed = sec2cats.get(sec, None)
        if not allowed:
            final[i] = gclasses[int(np.argmax(gscores[i]))]
        else:
            best_c = None; best_val = -1e9
            for c in allowed:
                gv = gstd_scores[i, gindex[c]] if c in gindex else -1e9
                if gv > best_val:
                    best_val, best_c = gv, c
            final[i] = best_c if best_c is not None else gclasses[int(np.argmax(gscores[i]))]
    return final

# ---------- Brand & size prediction ----------
def predict_brand_and_size(train_df: pd.DataFrame, df_to_predict: pd.DataFrame, pred_cat: List[str], text_col: str):
    brand_sorted, canon_map, brand_main_list = build_brand_dictionary(train_df)
    cat_brand_counts = (train_df.groupby(['Categoría','Marca']).size()
                        .groupby(level=0).apply(lambda s: (s/s.sum()).to_dict()).to_dict())
    if "OCR_Measure" in train_df.columns:
        unit_priors = (
            train_df[["Categoría", "OCR_Measure"]]
            .dropna()
            .assign(u=lambda d: d["OCR_Measure"].astype(str).str.strip().str.lower())
            .groupby("Categoría")["u"]
            .apply(lambda s: s.value_counts(normalize=True).to_dict())
            .to_dict()
        )
    else:
        unit_priors = {}

    raw_txt = df_to_predict[text_col].fillna("").astype(str).tolist()

    pred_brand = []
    for txt, cat_label in zip(raw_txt, pred_cat):
        b = None
        if HAVE_RAPIDFUZZ:
            b = pick_brand_with_prior(txt, brand_main_list, cat_label, cat_brand_counts, threshold=82)
        if not b:
            b = find_brand_in_text_exact_or_caps(txt, brand_sorted, canon_map, brand_main_list)
        pred_brand.append(b)

    def pick_size_for_category(text_raw: str, cat_lbl: str):
        cands = extract_all_sizes(text_raw)
        if not cands: return (np.nan, np.nan)
        pri = unit_priors.get(cat_lbl, {})
        def score(c):
            v,u,b = c
            return b + 0.6*pri.get(str(u).lower(), 0.0)
        best = max(cands, key=score)
        return (best[0], best[1])

    qty, unit = zip(*[pick_size_for_category(t, c) for t, c in zip(raw_txt, pred_cat)])
    return list(pred_brand), list(qty), list(unit)

# ---------- Valid masks ----------
def valid_mask_for_text(df: pd.DataFrame, text_col: str, min_len: int = 20) -> pd.Series:
    if text_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[text_col].notna() & (df[text_col].astype(str).str.strip().str.len() > min_len)

# ---------- Metrics ----------
def eval_cls(y_true: pd.Series, y_pred: pd.Series, name: str) -> Dict[str, Any]:
    mask = ~pd.isna(y_true)
    if mask.sum() == 0:
        return {"metric": name, "weighted_f1": np.nan, "accuracy": np.nan, "n": 0}
    y_t = y_true[mask].astype(str)
    y_p = y_pred[mask].astype(str)
    return {
        "metric": name,
        "weighted_f1": float(f1_score(y_t, y_p, average='weighted')),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "n": int(mask.sum())
    }

def evaluate_df(df_true: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    res = []
    res.append(eval_cls(df_true.get("Sector"),    pred_df["Pred_Sector"],    "Sector"))
    res.append(eval_cls(df_true.get("Categoría"), pred_df["Pred_Categoría"], "Categoría"))
    res.append(eval_cls(df_true.get("Marca"),     pred_df["Pred_Marca"].fillna("N/A"), "Marca"))
    size_true = pd.to_numeric(df_true.get("OCR_Size", pd.Series(index=df_true.index)), errors='coerce')
    unit_true = df_true.get("OCR_Measure", pd.Series(index=df_true.index)).apply(normalize_unit)
    size_pred = pd.to_numeric(pred_df["Pred_Quantity"], errors='coerce')
    unit_pred = pred_df["Pred_Unit"].apply(normalize_unit)
    size_mask = size_true.notna()
    unit_mask = unit_true.notna()
    size_acc = float(np.nanmean((size_true[size_mask] == size_pred[size_mask]).astype(float))) if size_mask.any() else np.nan
    unit_acc = float(np.nanmean((unit_true[unit_mask] == unit_pred[unit_mask]).astype(float))) if unit_mask.any() else np.nan
    res.append({"metric":"Quantity", "accuracy": size_acc, "n": int(size_mask.sum())})
    res.append({"metric":"Unit",     "accuracy": unit_acc,  "n": int(unit_mask.sum())})
    return pd.DataFrame(res)

# ===========================
# USE IN-MEMORY DFS: Train, Test
# ===========================
# 1) Train models on Train
vec, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats = train_global_models(Train)

# 2) TEST inference from Test.ocr_text (len > 20)
mask_test = valid_mask_for_text(Test, "ocr_text", 20)
Test_valid = Test.loc[mask_test].copy()

test_combined = build_combined_text(Test_valid["ocr_text"])
X_test = transform_char_vectorizer(vec, test_combined.tolist())

sec_pred = sec_clf.predict(X_test).tolist()
soft_txt = Test_valid["ocr_text"].fillna("").astype(str).map(soft_clean).tolist()
cat_pred = category_ensemble_predict_sharedX(
    X_test, soft_txt, sec_pred, cat_clf, sec_cat_models, sec2cats
)
sec_final = [cat2sec.get(c, s) for c, s in zip(cat_pred, sec_pred)]

brand_pred, qty_pred, unit_pred = predict_brand_and_size(Train, Test_valid, cat_pred, text_col="ocr_text")

# Full TEST output with NaNs for invalid rows
out_test = Test.copy()
for col in ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]:
    out_test[col] = np.nan
out_test.loc[mask_test, "Pred_Sector"]    = sec_final
out_test.loc[mask_test, "Pred_Categoría"] = cat_pred
out_test.loc[mask_test, "Pred_Marca"]     = brand_pred
out_test.loc[mask_test, "Pred_Quantity"]  = qty_pred
out_test.loc[mask_test, "Pred_Unit"]      = unit_pred

# Optional: evaluate on TEST (only if ground truth exists)
results_test = evaluate_df(
    Test.loc[mask_test],
    out_test.loc[mask_test, ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]]
)
print(f"\nValid TEST rows (ocr_text len>20): {mask_test.sum()} / {len(Test)}")
print("\n=== FAST Non-LLM Metrics on TEST (valid subset) ===")
with pd.option_context('display.max_colwidth', 200):
    print(results_test.to_string(index=False))

# 3) TRAIN sanity-check from Train.clean_description (len > 20)
mask_train = valid_mask_for_text(Train, "clean_description", 20)
Train_valid = Train.loc[mask_train].copy()

train_combined = build_combined_text(Train_valid["clean_description"])
X_train_eval = transform_char_vectorizer(vec, train_combined.tolist())

sec_pred_tr = sec_clf.predict(X_train_eval).tolist()
soft_txt_tr = Train_valid["clean_description"].fillna("").astype(str).map(soft_clean).tolist()
cat_pred_tr = category_ensemble_predict_sharedX(
    X_train_eval, soft_txt_tr, sec_pred_tr, cat_clf, sec_cat_models, sec2cats
)
sec_final_tr = [cat2sec.get(c, s) for c, s in zip(cat_pred_tr, sec_pred_tr)]
brand_pred_tr, qty_pred_tr, unit_pred_tr = predict_brand_and_size(Train, Train_valid, cat_pred_tr, "clean_description")

pred_train_df = pd.DataFrame({
    "Pred_Sector": sec_final_tr,
    "Pred_Categoría": cat_pred_tr,
    "Pred_Marca": brand_pred_tr,
    "Pred_Quantity": qty_pred_tr,
    "Pred_Unit": unit_pred_tr
}, index=Train_valid.index)

results_train = evaluate_df(Train_valid, pred_train_df)
print(f"\nValid TRAIN rows (clean_description len>20): {mask_train.sum()} / {len(Train)}")
print("\n=== FAST Non-LLM Metrics on TRAIN (valid subset) ===")
with pd.option_context('display.max_colwidth', 200):
    print(results_train.to_string(index=False))

# Optionally save:
# out_test.to_csv("/mnt/data/kantar_predictions_nonllm_ensemble.csv", index=False)

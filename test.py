# -*- coding: utf-8 -*-
"""
NON-LLM OCR attribute prediction — upgraded end-to-end (binary SVC fix).

- Trains Sector & Category from text-only (clean_description) -> avoids leakage
- Inference uses ONLY ocr_text
- Sector predicted directly from OCR (clean view)
- Category via GLOBAL model + SECTOR-SPECIFIC model ENSEMBLE (restricted to sector)
- Test-Time Augmentation (TTA): clean + soft OCR views blended
- Top-K category re-rank using tiny keyword hints
- Final Sector derived from final Category (hierarchy consistency)
- Brand via dictionary + RapidFuzz fuzzy + category prior + frequent-brand classifier fallback
- Quantity/Unit via regex++ with multi-candidate selection; safe if OCR_Measure missing
- Evaluation per target and CSV output

Inputs : /mnt/data/Kantar_train.csv, /mnt/data/Kantar_test.csv
Output : /mnt/data/kantar_predictions_nonllm_ensemble.csv

Requires:
    pip install pandas numpy scikit-learn Unidecode rapidfuzz
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from difflib import get_close_matches

# --- Optional deps with graceful fallback ---
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x  # no-op

try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------
# Paths
# ---------------------------
TRAIN_PATH = Path("/content/Kantar_train.csv")
TEST_PATH  = Path("/content/Kantar_test.csv")
OUT_PATH   = Path("/content/kantar_predictions_nonllm_ensemble.csv")

# ---------------------------
# OCR cleaning / normalization
# ---------------------------
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

# Robust normalizer: handles Series, list, ndarray, or scalar
def normalize_for_model(texts):
    def norm_one(v):
        if v is None:
            s = ""
        elif isinstance(v, float) and np.isnan(v):
            s = ""
        else:
            s = str(v)
        s = unidecode(s).lower().strip()
        return re.sub(r"\s+"," ", s)
    if isinstance(texts, pd.Series):
        return texts.fillna("").map(norm_one)
    elif isinstance(texts, (list, tuple, np.ndarray)):
        return np.array([norm_one(x) for x in texts], dtype=object)
    else:
        return np.array([norm_one(texts)], dtype=object)

# ---------------------------
# Size/Unit extraction (regex++) + multi-candidate selection
# ---------------------------
UNIT_MAP = {
    'gr':'g','g':'g','kg':'kg','ml':'ml','l':'L','lt':'L','l.':'L','un':'un','u':'un','oz':'oz','uds':'un'
}
SIZE_PATTERNS = [
    r'(?P<num>\d{1,4}(?:[.,]\d{1,3})?)\s*(?P<unit>kg|g|gr|ml|l|lt|l\.|oz|un|u|uds)\b',
    r'(?P<unit>kg|g|gr|ml|l|lt|l\.|oz|un|u|uds)\s*(?P<num>\d{1,4}(?:[.,]\d{1,3})?)\b',
    r'(?:x|\*)\s*(?P<num>\d{1,3})\s*(?P<unit>un|u|uds)\b',
    r'(?P<num>\d{1,3})\s*(?P<unit>un|u|uds)\b'
]
SIZE_RES = [re.compile(p, re.I) for p in SIZE_PATTERNS]
KEY_NEAR_RE = re.compile(r'(cont|contenido|neto|peso|volumen)', re.I)

def extract_all_sizes(text: str):
    if not isinstance(text,str): return []
    cands = []
    for rx in SIZE_RES:
        for m in rx.finditer(text):
            num = m.groupdict().get('num')
            unit = m.group('unit').lower()
            if not num:
                continue
            try:
                val = float(num.replace(',', '.'))
            except:
                continue
            unit = UNIT_MAP.get(unit, unit)
            bias = 0.0
            if KEY_NEAR_RE.search(text[max(0,m.start()-30): m.end()+30]): bias += 1.0
            if unit in {'g','kg','ml','L','oz'}: bias += 0.5
            cands.append((val, unit, bias))
    return cands

def normalize_unit(u: Any) -> Any:
    if isinstance(u, str):
        u2 = u.strip().lower()
        return UNIT_MAP.get(u2, u2)
    return u

# ---------------------------
# Brand dictionary + matching
# ---------------------------
def normalize_text(s: Any) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def strip_parentheses_brand(s: str) -> str:
    return re.sub(r'\s*\(.*?\)\s*', '', s or "", flags=re.S)

def build_brand_dictionary(train_df: pd.DataFrame) -> Tuple[List[str], Dict[str, str], List[str]]:
    df = train_df.copy()
    df['brand_full'] = df['Marca'].astype(str)
    df['brand_main'] = df['Marca'].astype(str).apply(strip_parentheses_brand)

    pool = pd.concat([df['brand_full'], df['brand_main']]).dropna().unique().tolist()
    normalized = [normalize_text(x) for x in pool if isinstance(x, str) and x.strip()]
    normalized = list(set(normalized))
    brand_sorted = sorted(normalized, key=len, reverse=True)

    canon_map = {}
    for b in brand_sorted:
        mask = df['Marca'].str.lower().str.contains(re.escape(b), regex=True, na=False)
        candidates = df.loc[mask, 'Marca']
        canon_map[b] = candidates.mode().iloc[0] if len(candidates) else b

    brand_main_list = sorted(df['brand_main'].dropna().unique().tolist(), key=len, reverse=True)
    return brand_sorted, canon_map, brand_main_list

CAPS_RE = re.compile(r'\b[A-Z][A-Z0-9ÁÉÍÓÚÜÑ\-]{2,}\b')

def find_brand_in_text_exact_or_caps(text: Any,
                                     brand_sorted: List[str],
                                     canon_map: Dict[str,str],
                                     brand_main_list: List[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip(): return None
    t_norm = normalize_text(text)
    for b in brand_sorted:
        if b and b in t_norm:
            return canon_map.get(b, b)
    caps = CAPS_RE.findall(text.upper())
    for token in caps:
        exact = [bm for bm in brand_main_list if bm.upper() == token]
        if exact:
            return exact[0]
    if caps:
        candidate = " ".join(caps)
        match_list = get_close_matches(candidate, [bm.upper() for bm in brand_main_list], n=1, cutoff=0.85)
        if match_list:
            return match_list[0].title()
    return None

def pick_brand_with_prior(text: str,
                          brand_main_list: List[str],
                          pred_cat: str,
                          cat_brand_counts: Dict[str, Dict[str,float]],
                          threshold:int=80) -> Optional[str]:
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

# ---------------------------
# Models & mappings
# ---------------------------
def make_text_clf(max_char_feats=400_000, max_word_feats=200_000, C=6.0, min_df=3):
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,7),
                               min_df=min_df, max_features=max_char_feats)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                               min_df=min_df, max_features=max_word_feats)
    feats = FeatureUnion([("char", char_vec), ("word", word_vec)])
    return Pipeline([
        ("norm", FunctionTransformer(normalize_for_model, validate=False)),
        ("feats", feats),
        ("clf", LinearSVC(C=C, class_weight="balanced", max_iter=10000, tol=1e-4))
    ])

def train_category_model(train_df: pd.DataFrame):
    if "clean_description" not in train_df.columns:
        raise ValueError("Training data must contain 'clean_description'.")
    X = train_df["clean_description"].fillna("")
    y = train_df["Categoría"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(sss.split(X, y))
    cat_clf = make_text_clf(C=6.0, min_df=3)
    cat_clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    cat2sec = train_df.groupby("Categoría")["Sector"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    return cat_clf, cat2sec

def train_sector_model(train_df: pd.DataFrame):
    X = train_df["clean_description"].fillna("")
    y = train_df["Sector"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(sss.split(X, y))
    sec_clf = make_text_clf(max_char_feats=300_000, max_word_feats=150_000, C=3.0, min_df=3)
    sec_clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return sec_clf

def build_sector_to_categories(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    return (train_df.groupby("Sector")["Categoría"]
            .apply(lambda s: sorted(s.unique()))
            .to_dict())

def train_sector_specific_category_models(train_df):
    models = {}
    for sec, sub in train_df.groupby("Sector"):
        if sub["Categoría"].nunique() < 2:
            continue
        Xs = sub["clean_description"].fillna("")
        ys = sub["Categoría"].astype(str)
        clf = make_text_clf(max_char_feats=250_000, max_word_feats=120_000, C=4.0, min_df=2)
        clf.fit(Xs, ys)
        models[sec] = clf
    return models

# ---------------------------
# Category prediction: ensemble (global + sector model) + TTA + Top-K rerank
# ---------------------------
CATEGORY_HINTS = {
    "Café": ["cafe","colcafe","instantaneo","molido"],
    "Chocolate Para Taza / Cocoa": ["chocolate","cocoa","mesa","taza"],
    "Jabón En Barra Para Ropa": ["jabon","jabón","ropa","barra","detergente"],
    # Add more high-support categories here if desired
}

def _binary_two_sided_scores(sdec, sclasses):
    """
    Ensure sector-model scores are per-class even in binary case.
    For LinearSVC binary, decision_function -> shape (1,), positive for sclasses[1].
    We map to [-d, +d] aligned with classes order.
    """
    d = float(np.ravel(sdec)[0])
    if len(sclasses) == 2:
        # convention: score for class 0 = -d, class 1 = +d
        return np.array([[-d, d]], dtype=float)
    else:
        # fall back (unlikely)
        return np.array([np.ravel(sdec)], dtype=float)

def sector_ensemble_category_predict(ocrs_clean, ocrs_soft,
                                     global_cat_clf, sec_pred,
                                     sec2cats, sec_cat_models):
    # Global features with TTA (sum of clean+soft)
    Xg1 = global_cat_clf[:-1].transform(ocrs_clean)
    Xg2 = global_cat_clf[:-1].transform(ocrs_soft)
    try:
        Xg = Xg1 + Xg2
    except:
        Xg = Xg1
    gclf = global_cat_clf.named_steps["clf"]
    gscores = gclf.decision_function(Xg)
    if gscores.ndim == 1: gscores = gscores.reshape(-1,1)
    gclasses = list(gclf.classes_)
    gindex = {c:i for i,c in enumerate(gclasses)}

    # Per-row z-score
    gmean = gscores.mean(axis=1, keepdims=True)
    gstd  = gscores.std(axis=1, keepdims=True) + 1e-6
    gstd_scores = (gscores - gmean)/gstd

    final = []
    # Keep Top-K for re-rank
    topk_idx = np.argsort(-gscores, axis=1)[:, :5]
    topk = [[gclasses[j] for j in row] for row in topk_idx]

    for i, sec in enumerate(sec_pred):
        allowed = sec2cats.get(sec, None)
        sstd_scores = None; sindex = {}

        if sec in sec_cat_models:
            sclf = sec_cat_models[sec]
            Xs1 = sclf[:-1].transform([ocrs_clean.iloc[i]])
            Xs2 = sclf[:-1].transform([ocrs_soft.iloc[i]])
            try:
                Xs = Xs1 + Xs2
            except:
                Xs = Xs1
            sdec = sclf.named_steps["clf"].decision_function(Xs)
            sclasses = list(sclf.named_steps["clf"].classes_)
            sindex = {c:j for j,c in enumerate(sclasses)}
            # --- FIX: make per-class scores in binary case ---
            if np.ndim(sdec) == 1:
                sfull = _binary_two_sided_scores(sdec, sclasses)
            else:
                sfull = sdec
            smean = sfull.mean(axis=1, keepdims=True)
            sstd  = sfull.std(axis=1, keepdims=True) + 1e-6
            sstd_scores = (sfull - smean)/sstd

        # Ensemble within allowed set
        if not allowed:
            chosen = gclasses[int(np.argmax(gscores[i]))]
        else:
            best_c = None; best_val = -1e9
            for c in allowed:
                gv = gstd_scores[i, gindex[c]] if c in gindex else -1e9
                sv = (sstd_scores[0, sindex[c]] if (sstd_scores is not None and c in sindex) else 0.0)
                val = 0.6*gv + 0.4*sv
                if val > best_val:
                    best_val, best_c = val, c
            chosen = best_c if best_c is not None else gclasses[int(np.argmax(gscores[i]))]

        # Tiny keyword re-rank among global Top-K
        raw = ocrs_soft.iloc[i]
        hints_best = chosen; best_hits = -1
        for c in topk[i]:
            hits = sum(1 for h in CATEGORY_HINTS.get(c, []) if h in raw)
            if hits > best_hits:
                hints_best, best_hits = c, hits

        if allowed and hints_best in allowed:
            final.append(hints_best)
        else:
            final.append(chosen)

    return final, gscores, gclasses

# ---------------------------
# Frequent-brand classifier (fallback, non-LLM)
# ---------------------------
def train_freq_brand_clf(train_df: pd.DataFrame):
    brand_counts = train_df['Marca'].value_counts()
    freq_brands = set(brand_counts[brand_counts>=30].index)
    mask = train_df['Marca'].isin(freq_brands)
    if mask.sum() == 0:
        return None, set()
    clf = make_text_clf(max_char_feats=300_000, max_word_feats=150_000, C=3.0, min_df=2)
    clf.fit(train_df.loc[mask,'clean_description'].fillna(''),
            train_df.loc[mask,'Marca'].astype(str))
    return clf, freq_brands

# ---------------------------
# Predict (non-LLM)
# ---------------------------
def predict_nonllm(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   cat_clf: Pipeline,
                   sec_clf: Pipeline,
                   cat2sec: Dict[str,str],
                   sec2cats: Dict[str, List[str]],
                   sec_cat_models: Dict[str, Pipeline],
                   brand_freq_clf=None,
                   freq_brand_set: Optional[set]=None) -> pd.DataFrame:

    brand_sorted, canon_map, brand_main_list = build_brand_dictionary(train_df)
    cat_brand_counts = (train_df.groupby(['Categoría','Marca']).size()
                        .groupby(level=0).apply(lambda s: (s/s.sum()).to_dict()).to_dict())

    # Unit priors per category (safe if OCR_Measure missing)
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

    ocrs_raw   = test_df["ocr_text"].fillna("")
    ocrs_clean = ocrs_raw.map(clean_ocr_for_inference)
    ocrs_soft  = ocrs_raw.map(soft_clean)

    pred_sector_direct = sec_clf.predict(ocrs_clean)

    pred_cat, gscores, gclasses = sector_ensemble_category_predict(
        ocrs_clean, ocrs_soft, cat_clf, pred_sector_direct, sec2cats, sec_cat_models
    )

    pred_sec = [cat2sec.get(c, None) for c in pred_cat]

    # Brand with layered strategy
    pred_brand = []
    for txt_raw, cat_label in zip(ocrs_raw, pred_cat):
        b = None
        if HAVE_RAPIDFUZZ:
            b = pick_brand_with_prior(txt_raw, brand_main_list, cat_label, cat_brand_counts, threshold=80)
        if not b:
            b = find_brand_in_text_exact_or_caps(txt_raw, brand_sorted, canon_map, brand_main_list)
        if not b and (brand_freq_clf is not None):
            try:
                b = brand_freq_clf.predict([clean_ocr_for_inference(txt_raw)])[0]
                if freq_brand_set and b not in freq_brand_set:
                    b = None
            except:
                b = None
        pred_brand.append(b)

    # Quantity/Unit with priors
    def pick_size_for_category(text_raw: str, pred_cat: str):
        cands = extract_all_sizes(text_raw)
        if not cands: return (np.nan, np.nan)
        pri = unit_priors.get(pred_cat, {})
        def score(c):
            v,u,b = c
            return b + 0.6*pri.get(str(u).lower(), 0.0)
        best = max(cands, key=score)
        return (best[0], best[1])

    pred_qty, pred_unit = zip(*[pick_size_for_category(t, c) for t,c in zip(ocrs_raw, pred_cat)])

    pred = pd.DataFrame({
        "Pred_Sector":    pred_sec,
        "Pred_Categoría": pred_cat,
        "Pred_Marca":     pred_brand,
        "Pred_Quantity":  list(pred_qty),
        "Pred_Unit":      list(pred_unit),
    }, index=test_df.index)

    # final pass consistency
    pred["Pred_Sector"] = pred["Pred_Categoría"].map(cat2sec).fillna(pred["Pred_Sector"])
    return pred

# ---------------------------
# Evaluation
# ---------------------------
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

def evaluate_on_test(test_df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    results = []
    results.append(eval_cls(test_df.get("Sector"),    pred["Pred_Sector"],    "Sector"))
    results.append(eval_cls(test_df.get("Categoría"), pred["Pred_Categoría"], "Categoría"))
    results.append(eval_cls(test_df.get("Marca"),     pred["Pred_Marca"].fillna("N/A"), "Marca"))

    size_true = pd.to_numeric(test_df.get("OCR_Size", pd.Series(index=test_df.index)), errors='coerce')
    unit_true = test_df.get("OCR_Measure", pd.Series(index=test_df.index)).apply(normalize_unit)

    size_pred = pd.to_numeric(pred["Pred_Quantity"], errors='coerce')
    unit_pred = pred["Pred_Unit"].apply(normalize_unit)

    size_mask = size_true.notna()
    unit_mask = unit_true.notna()

    size_acc = float(np.nanmean((size_true[size_mask] == size_pred[size_mask]).astype(float))) if size_mask.any() else np.nan
    unit_acc = float(np.nanmean((unit_true[unit_mask] == unit_pred[unit_mask]).astype(float))) if unit_mask.any() else np.nan

    results.append({"metric":"Quantity", "accuracy": size_acc, "n": int(size_mask.sum())})
    results.append({"metric":"Unit",     "accuracy": unit_acc,  "n": int(unit_mask.sum())})
    return pd.DataFrame(results)

# ---------------------------
# Main
# ---------------------------
def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Please place Kantar_train.csv and Kantar_test.csv in /mnt/data")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    cat_clf, cat2sec = train_category_model(train_df)
    sec_clf          = train_sector_model(train_df)
    sec2cats         = build_sector_to_categories(train_df)
    sec_cat_models   = train_sector_specific_category_models(train_df)
    brand_freq_clf, freq_brand_set = train_freq_brand_clf(train_df)

    pred = predict_nonllm(train_df, test_df,
                          cat_clf, sec_clf, cat2sec, sec2cats, sec_cat_models,
                          brand_freq_clf=brand_freq_clf, freq_brand_set=freq_brand_set)

    results_df = evaluate_on_test(test_df, pred)

    out = test_df.copy()
    out["Pred_Sector"]    = pred["Pred_Sector"]
    out["Pred_Categoría"] = pred["Pred_Categoría"]
    out["Pred_Marca"]     = pred["Pred_Marca"]
    out["Pred_Quantity"]  = pred["Pred_Quantity"]
    out["Pred_Unit"]      = pred["Pred_Unit"]
    out.to_csv(OUT_PATH, index=False)

    print("\n=== Non-LLM Metrics (using only ocr_text at inference) ===")
    with pd.option_context('display.max_colwidth', 200):
        print(results_df.to_string(index=False))
    print(f"\nPredictions saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()

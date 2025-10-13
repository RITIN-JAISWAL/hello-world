# -*- coding: utf-8 -*-
"""
FAST non-LLM pipeline with OCR-like synthetic training text
- Use in-memory Train, Test DataFrames (no CSV reads here)
- Train text = synthetic OCR-ized version of clean_description (no label leakage)
- Vectorizer vocabulary = Train_synth ∪ sample(Train.ocr_text) ∪ Test.ocr_text
- Rest: shared char TF-IDF, LinearSVC (global + sector-specific), binary fix, ensemble,
        brand & size extraction, metrics, only Test.ocr_text len>20.

This cell **replaces** train_global_models() to consume OCR-like training text.
"""

import os, re, random
import numpy as np, pandas as pd
from typing import Any, Dict, List, Optional
from difflib import get_close_matches
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

# Thread caps for Azure ML CPU nodes (tweak to vCPU count)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

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

# ---------- OCR-like synthesizer (NO label strings inserted) ----------
_unit_variants = {
    "l": ["l","lt","l.","litro","litros"],
    "ml": ["ml","m.l","mililitros"],
    "g": ["g","gr","gramo","gramos"],
    "kg": ["kg","kilo","kilogramos"],
    "un": ["un","u","uds","unidad","unid"]
}
def _random_unit(u: str) -> str:
    u = (u or "").strip().lower()
    if u in _unit_variants: return random.choice(_unit_variants[u])
    return u

def _jitter_text(t: str) -> str:
    # light OCR-ish jitter: drop accents (already), collapse/expand spaces, strip punctuation,
    # swap similar glyphs occasionally (o/0, l/1), random hyphens/dots
    t = re.sub(r"[,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # small prob glyph noise
    t = re.sub(r"0", "o", t) if random.random()<0.1 else t
    t = re.sub(r"o", "0", t) if random.random()<0.05 else t
    t = re.sub(r"l", "1", t) if random.random()<0.05 else t
    # spacing noise
    if random.random()<0.2:
        t = t.replace(" ", "  ")
    # random dash/dot injections between tokens
    if random.random()<0.15:
        t = re.sub(r"\b(\w{3,})\b", r"\1-", t, count=1)
    return t

def synth_ocr_like_from_clean_row(row: pd.Series) -> str:
    """
    Build a synthetic OCR-like string from clean_description + **non-label** fields.
    We never include explicit 'Sector' or 'Categoría' text tokens here.
    """
    desc = clean_ocr_for_inference(str(row.get("clean_description", "")))
    brand = str(row.get("Marca","")).strip()
    size  = row.get("OCR_Size", "")
    unit  = row.get("OCR_Measure", "")
    # safe tokens to include: brand, numbers/units, generic cues
    segs = [desc]
    if brand:
        segs.append(brand)
    # include size/unit as they appear in real OCR
    if pd.notna(size) or pd.notna(unit):
        u = _random_unit(str(unit))
        try:
            if pd.notna(size) and str(size).strip():
                # 20% chance to use comma decimal
                s = str(size)
                if random.random()<0.2 and "." in s:
                    s = s.replace(".", ",")
                segs.append(f"{s} {u}".strip())
            else:
                segs.append(u)
        except Exception:
            pass
    # generic OCR markers
    if random.random()<0.15:
        segs.append("contenido neto")
    if random.random()<0.1:
        segs.append("lote 1234")

    txt = " ".join([x for x in segs if isinstance(x,str) and x.strip()])
    txt = _jitter_text(txt)
    return txt

def make_train_synth_text(train_df: pd.DataFrame) -> pd.Series:
    return train_df.apply(synth_ocr_like_from_clean_row, axis=1)

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
            num = m.groupdict().get('num'); unit = m.group('unit').lower()
            if not num: continue
            try: val = float(num.replace(',', '.'))
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

# ---------- Vectorizer ----------
def fit_char_vectorizer(X_text: List[str], min_df=3, max_features=300_000):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,7),
        min_df=min_df, max_features=max_features,
        dtype=np.float32, sublinear_tf=True, norm="l2"
    )
    X = vec.fit_transform(X_text).astype(np.float32, copy=False)
    return vec, X

def transform_char_vectorizer(vec: TfidfVectorizer, texts: List[str]):
    return vec.transform(texts).astype(np.float32, copy=False)

# ---------- Train global & sector models on OCR-like text ----------
def train_global_models(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        sample_frac_train_ocr: float = 0.25,
                        max_vocab_features: int = 350_000):
    df = train_df.reset_index(drop=True).copy()

    # 1) Primary training text (one per row): synthetic OCR-like from clean_description (NO label words)
    train_synth = make_train_synth_text(df)

    # 2) Unlabeled pool for vocabulary only: Test.ocr_text + sampled Train.ocr_text (if present)
    unlabeled = []
    if "ocr_text" in df.columns:
        # sample some true OCR to mix into vec.fit
        tr_ocr = df["ocr_text"].dropna().astype(str)
        if len(tr_ocr) > 0 and sample_frac_train_ocr>0:
            unlabeled.append(tr_ocr.sample(frac=min(1.0, sample_frac_train_ocr), random_state=42))
    if "ocr_text" in test_df.columns:
        unlabeled.append(test_df["ocr_text"].dropna().astype(str))
    unlabeled_corpus = pd.concat([train_synth] + unlabeled, axis=0).map(clean_ocr_for_inference)

    # 3) Fit vectorizer on the union (unlabeled), transform train_synth for supervised training
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,7),
        min_df=3, max_features=max_vocab_features,
        dtype=np.float32, sublinear_tf=True, norm="l2"
    )
    vec.fit(unlabeled_corpus.tolist())

    X = vec.transform(train_synth.tolist()).astype(np.float32, copy=False)

    y_cat = df["Categoría"].astype(str).tolist()
    y_sec = df["Sector"].astype(str).tolist()

    cat_clf = LinearSVC(C=5.0, class_weight="balanced", max_iter=8000, tol=2e-4)
    sec_clf = LinearSVC(C=3.5, class_weight="balanced", max_iter=7000, tol=2e-4)
    cat_clf.fit(X, y_cat)
    sec_clf.fit(X, y_sec)

    # sector-specific models on the same X
    sec_cat_models: Dict[str, LinearSVC] = {}
    for sec, idxs in df.groupby("Sector").indices.items():
        if df.iloc[idxs]["Categoría"].nunique() < 2:
            continue
        y_sub = df.iloc[idxs]["Categoría"].astype(str).values
        X_sub = X[idxs]
        clf = LinearSVC(C=4.0, class_weight="balanced", max_iter=8000, tol=2e-4)
        clf.fit(X_sub, y_sub)
        sec_cat_models[sec] = clf

    cat2sec = df.groupby("Categoría")["Sector"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    sec2cats = df.groupby("Sector")["Categoría"].apply(lambda s: sorted(s.unique())).to_dict()
    return vec, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats

# ---------- Category ensemble (binary fix) ----------
CATEGORY_HINTS = {
    "Café": ["cafe","colcafe","instantaneo","molido"],
    "Chocolate Para Taza / Cocoa": ["chocolate","cocoa","mesa","taza"],
    "Jabón En Barra Para Ropa": ["jabon","jabón","ropa","barra","detergente"],
}
def _binary_two_sided_scores_batch(dec_values: np.ndarray) -> np.ndarray:
    d = np.ravel(dec_values).astype(float)
    return np.vstack([-d, d]).T

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
        sfull = _binary_two_sided_scores_batch(sdec) if np.ndim(sdec) == 1 else sdec

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
        if final[i] is not None: continue
        sec = sec_pred[i]
        allowed = sec2cats.get(sec, None)
        if not allowed:
            final[i] = gclasses[int(np.argmax(gscores[i]))]
        else:
            best_c = None; best_val = -1e9
            for c in allowed:
                gv = gstd_scores[i, gindex[c]] if c in gindex else -1e9
                if gv > best_val: best_val, best_c = gv, c
            final[i] = best_c if best_c is not None else gclasses[int(np.argmax(gscores[i]))]
    return final

# ---------- Brand & size ----------
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
        def score(c): v,u,b = c; return b + 0.6*pri.get(str(u).lower(), 0.0)
        best = max(cands, key=score)
        return (best[0], best[1])

    qty, unit = zip(*[pick_size_for_category(t, c) for t, c in zip(raw_txt, pred_cat)])
    return list(pred_brand), list(qty), list(unit)

# ---------- Valid mask ----------
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
assert 'Train' in globals() and 'Test' in globals(), "Please define Train and Test DataFrames first."

# Train models on OCR-like synthetic text, vectorizer fitted on union corpus
vec, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats = train_global_models(Train, Test,
                                                                               sample_frac_train_ocr=0.25,
                                                                               max_vocab_features=350_000)

# -------- TEST inference (ocr_text len>20) --------
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

out_test = Test.copy()
for col in ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]:
    out_test[col] = np.nan
out_test.loc[mask_test, "Pred_Sector"]    = sec_final
out_test.loc[mask_test, "Pred_Categoría"] = cat_pred
out_test.loc[mask_test, "Pred_Marca"]     = brand_pred
out_test.loc[mask_test, "Pred_Quantity"]  = qty_pred
out_test.loc[mask_test, "Pred_Unit"]      = unit_pred

# Optional metrics on TEST if labels exist
results_test = evaluate_df(
    Test.loc[mask_test],
    out_test.loc[mask_test, ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]]
)
print(f"\nValid TEST rows (ocr_text len>20): {mask_test.sum()} / {len(Test)}")
print("\n=== FAST Non-LLM Metrics on TEST (valid subset) — with OCR-like training text ===")
with pd.option_context('display.max_colwidth', 200):
    print(results_test.to_string(index=False))

# -------- TRAIN sanity-check (clean_description len>20) --------
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












# -*- coding: utf-8 -*-
"""
Non-LLM OCR pipeline — hardened against pandas truthiness + float casting

• Sector + Categoría on shared TF-IDF (char 3–7 + word 1–2)
• Per-sector specialists; mask to sector’s category set
• Lexicon nudges + brand→category priors
• Robust conversions: never rely on pandas truthiness; cast scalars to float
• Test on Test.ocr_text (len>20); sanity on Train.clean_description (len>20)
"""

import os, re, numpy as np, pandas as pd
from typing import Any, Dict, List, Optional
from difflib import get_close_matches
from scipy.sparse import csr_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

# Threads
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

# ---------------- Cleaning ----------------
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

# ---------------- Size/Unit ----------------
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
            num = m.groupdict().get('num'); unit = m.group('unit').lower()
            if not num: continue
            try: val = float(num.replace(',', '.'))
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

# ---------------- Brand tools ----------------
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

# ---------------- Brand→Categoría priors ----------------
def build_brand_cat_priors(train_df: pd.DataFrame, min_count=5, smoothing=0.5) -> pd.DataFrame:
    pairs = train_df.groupby(["Marca","Categoría"]).size().unstack(fill_value=0)
    brand_counts = pairs.sum(axis=1)
    keep = brand_counts[brand_counts >= min_count].index
    pairs = pairs.loc[keep]
    priors = (pairs + smoothing)
    priors = priors.div(priors.sum(axis=1), axis=0)  # row-normalize
    priors.index = priors.index.map(lambda x: normalize_text(strip_parentheses_brand(str(x))))
    priors.columns = priors.columns.astype(str)
    priors = priors.astype(float)   # << ensure numeric float
    return priors

# ---------------- Lexicon nudges ----------------
LEX = {
    # Alimentos
    "Snacks": ["papas","chips","snack","yuca frita","nachos","platanitos","onduladas","sabor a","barbacoa","limon","picante"],
    "Galletas": ["galleta","cookies","cracker","wafer","saltin","ducales","oreo","club social"],
    "Pan Industrializado": ["pan tajado","pan molde","pan blanco","pan integral","pan hamburguesa","pan perro"],
    "Granos Y Legumbres": ["frijol","lenteja","garbanzo","caraota","arveja","legumbre","grano seco"],
    "Condimentos / Adobos Y Sazonadores": ["adobo","sazonador","caldo","consome","comino","pimienta","aliño","ajo en polvo","oregano"],
    # Belleza
    "Acondicionador / Tratamientos / Cremas": ["acondicionador","tratamiento capilar","crema para peinar","keratina","baba de caracol","reparacion"],
    "Shampoo": ["shampoo","anticaspa","higiene capilar","control caida","fortalecedor"],
    "Tintes Para Cabello": ["tinte","coloracion","permanente","rubio","castaño","negro natural"],
    "Crema Corporal": ["hidratacion corporal","locion corporal","piel seca","cuidado corporal","body lotion","humectante corporal"],
    "Crema Facial": ["crema facial","hidratante facial","antiarrugas","acido hialuronico","retinol","piel mixta"],
    "Fragancias / Perfumes / Agua De Colonia": ["eau de parfum","eau de toilette","colonia","fragancia","body spray"],
}

def lexicon_logboost(text: str, classes: List[str], gindex: Dict[str,int], log_boost=0.35):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(len(classes), dtype=np.float64)
    low = text.lower()
    bonus = np.zeros(len(classes), dtype=np.float64)
    for cat, terms in LEX.items():
        j = gindex.get(cat, None)
        if j is None:
            continue
        for t in terms:
            if t in low:
                bonus[j] += float(log_boost)   # ensure scalar float
                break
    return bonus

# ---------------- Vectorizers ----------------
def fit_char_word_vectorizers(texts: List[str],
                              min_df_char=3, min_df_word=3,
                              max_char=300_000, max_word=120_000):
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,7),
                               min_df=min_df_char, max_features=max_char,
                               dtype=np.float32, sublinear_tf=True, norm="l2")
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                               min_df=min_df_word, max_features=max_word,
                               dtype=np.float32, sublinear_tf=True, norm="l2")
    feats = FeatureUnion([("char", char_vec), ("word", word_vec)])
    X = feats.fit_transform(texts).astype(np.float32, copy=False)
    return feats, X

def transform_char_word(feats: FeatureUnion, texts: List[str]):
    return feats.transform(texts).astype(np.float32, copy=False)

# ---------------- Train models ----------------
def train_global_models(train_df: pd.DataFrame):
    df = train_df.reset_index(drop=True).copy()
    tr_text = df["clean_description"].fillna("").astype(str)
    combined = build_combined_text(tr_text)

    feats, X = fit_char_word_vectorizers(combined.tolist(),
                                         min_df_char=3, min_df_word=3,
                                         max_char=300_000, max_word=120_000)

    y_cat = df["Categoría"].astype(str).tolist()
    y_sec = df["Sector"].astype(str).tolist()

    cat_clf = LinearSVC(C=5.0, class_weight="balanced", max_iter=8000, tol=2e-4)
    sec_clf = LinearSVC(C=3.0, class_weight="balanced", max_iter=7000, tol=2e-4)
    cat_clf.fit(X, y_cat)
    sec_clf.fit(X, y_sec)

    sec_cat_models: Dict[str, LinearSVC] = {}
    for sec, idxs in df.groupby("Sector").indices.items():
        if df.iloc[idxs]["Categoría"].nunique() < 2:
            continue
        y_sub = df.iloc[idxs]["Categoría"].astype(str).values
        X_sub = X[idxs]
        clf = LinearSVC(C=4.0, class_weight="balanced", max_iter=8000, tol=2e-4)
        clf.fit(X_sub, y_sub)
        sec_cat_models[sec] = clf

    cat2sec = df.groupby("Categoría")["Sector"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    sec2cats_list = df.groupby("Sector")["Categoría"].apply(lambda s: sorted(list(pd.Series(s.unique()).astype(str)))).to_dict()

    return feats, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats_list

# ---------------- Helpers ----------------
def to_float_scalar(x) -> float:
    """Convert any numpy/pandas scalar to Python float robustly."""
    if isinstance(x, (pd.Series, pd.Index)):
        x = np.asarray(x).ravel()
        x = x[0] if x.size else 0.0
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else 0.0
    return float(x)

# ---------------- Category ensemble (robust) ----------------
CATEGORY_HINTS = {
    "Café": ["cafe","colcafe","instantaneo","molido"],
    "Chocolate Para Taza / Cocoa": ["chocolate","cocoa","mesa","taza"],
    "Jabón En Barra Para Ropa": ["jabon","jabón","ropa","barra","detergente"],
}

def category_ensemble_predict_sharedX(
    X: csr_matrix,
    raw_soft_text,            # list-like
    raw_text_for_brand,       # list-like
    sec_pred,                 # list-like
    cat_clf: LinearSVC,
    sec_cat_models: Dict[str, LinearSVC],
    sec2cats,                 # dict-like
    brand_sorted: List[str],
    canon_map: Dict[str,str],
    brand_main_list: List[str],
    brand_cat_priors: Optional[pd.DataFrame] = None,
    lexicon_log_boost: float = 0.35
):
    # ---- Defensive coercions on inputs ----
    if isinstance(raw_soft_text, pd.Series):        raw_soft_text = raw_soft_text.astype(str).tolist()
    else:                                           raw_soft_text = list(map(str, raw_soft_text))
    if isinstance(raw_text_for_brand, pd.Series):   raw_text_for_brand = raw_text_for_brand.astype(str).tolist()
    else:                                           raw_text_for_brand = list(map(str, raw_text_for_brand))
    if isinstance(sec_pred, (pd.Series, pd.Index)): sec_pred = sec_pred.astype(str).tolist()
    else:                                           sec_pred = list(map(str, sec_pred))

    # Coerce sec2cats -> dict[str, set[str]]
    sec2cats_sets: Dict[str, set] = {}
    if isinstance(sec2cats, pd.Series): sec2cats = sec2cats.to_dict()
    if isinstance(sec2cats, dict):
        for k, v in sec2cats.items():
            if isinstance(v, (pd.Series, pd.Index)): vals = set(map(str, v.tolist()))
            elif isinstance(v, (list, tuple, set)): vals = set(map(str, v))
            else: vals = set()
            sec2cats_sets[str(k)] = vals

    # Global scores
    gscores = cat_clf.decision_function(X)
    if gscores.ndim == 1: gscores = gscores.reshape(-1, 1)
    gclasses = list(map(str, cat_clf.classes_))
    gindex = {c: i for i, c in enumerate(gclasses)}

    gmean = gscores.mean(axis=1, keepdims=True)
    gstd  = gscores.std(axis=1, keepdims=True) + 1e-6
    gstd_scores = (gscores - gmean) / gstd

    topk_idx = np.argsort(-gscores, axis=1)[:, :5]
    topk = [[gclasses[j] for j in row] for row in topk_idx]

    # Detected brands (normalized)
    detected_brands = []
    for txt in raw_text_for_brand:
        b = find_brand_in_text_exact_or_caps(txt, brand_sorted, canon_map, brand_main_list)
        detected_brands.append(normalize_text(strip_parentheses_brand(b)) if b else None)

    final = [None] * X.shape[0]
    sec_pred_arr = np.array(sec_pred, dtype=object)

    def _binary_two_sided_scores_batch(dec_values: np.ndarray) -> np.ndarray:
        d = np.ravel(dec_values).astype(float)
        return np.vstack([-d, d]).T

    def _softmax_from_scores(vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float64)
        m = np.max(v)
        v = v - m
        e = np.exp(v)
        s = e.sum()
        return e / s if s > 0 else np.full_like(e, 1.0 / len(e))

    # ----- with specialists -----
    for sec, clf in sec_cat_models.items():
        rows = np.where(sec_pred_arr == sec)[0]
        if rows.size == 0: continue

        Xs = X[rows]
        sdec = clf.decision_function(Xs)
        sclasses = list(map(str, clf.classes_))
        sindex = {c: j for j, c in enumerate(sclasses)}
        sfull = _binary_two_sided_scores_batch(sdec) if np.ndim(sdec) == 1 else sdec

        smean = sfull.mean(axis=1, keepdims=True)
        sstd  = sfull.std(axis=1, keepdims=True) + 1e-6
        sstd_scores = (sfull - smean) / sstd

        allowed = sec2cats_sets.get(str(sec), set())
        has_allowed = (len(allowed) > 0)

        for k, i in enumerate(rows):
            raw_scores = np.full(len(gclasses), -1e9, dtype=np.float64)
            iter_classes = allowed if has_allowed else gclasses
            for c in iter_classes:
                if c in gindex:
                    gv = gstd_scores[i, gindex[c]]
                    sv = sstd_scores[k, sindex[c]] if c in sindex else 0.0
                    raw_scores[gindex[c]] = 0.6 * to_float_scalar(gv) + 0.4 * to_float_scalar(sv)

            raw_scores += lexicon_logboost(raw_soft_text[i], gclasses, gindex, log_boost=lexicon_log_boost)

            probs = _softmax_from_scores(raw_scores)

            br = detected_brands[i]
            if (br is not None) and (brand_cat_priors is not None) and (br in brand_cat_priors.index):
                p_prior = np.ones_like(probs)
                prior_row = brand_cat_priors.loc[br].astype(float)
                for c, p in prior_row.items():
                    j = gindex.get(str(c), None)
                    if j is not None:
                        p_prior[j] = max(to_float_scalar(p), 1e-6)
                probs *= p_prior
                s = probs.sum()
                if s > 0: probs /= s

            if has_allowed:
                idx = [gindex[c] for c in allowed if c in gindex]
                if len(idx) > 0:
                    maskv = np.zeros_like(probs); maskv[idx] = probs[idx]
                    s = to_float_scalar(maskv.sum())
                    probs = (maskv / s) if s > 0 else maskv

            chosen = gclasses[int(np.argmax(probs))]

            raw = raw_soft_text[i]
            hints_best, best_hits = chosen, -1
            for c in topk[i]:
                hits = sum(1 for h in CATEGORY_HINTS.get(c, []) if h in raw)
                if hits > best_hits:
                    hints_best, best_hits = c, hits

            final[i] = hints_best if (not has_allowed or hints_best in allowed) else chosen

    # ----- fallback -----
    for i in range(len(final)):
        if final[i] is not None: continue
        sec = sec_pred[i]
        allowed = sec2cats_sets.get(str(sec), set())
        has_allowed = (len(allowed) > 0)

        raw_scores = np.full(len(gclasses), -1e9, dtype=np.float64)
        iter_classes = allowed if has_allowed else gclasses
        for c in iter_classes:
            if c in gindex:
                raw_scores[gindex[c]] = to_float_scalar(gstd_scores[i, gindex[c]])

        raw_scores += lexicon_logboost(raw_soft_text[i], gclasses, gindex, log_boost=lexicon_log_boost)
        probs = _softmax_from_scores(raw_scores)

        br = detected_brands[i]
        if (br is not None) and (brand_cat_priors is not None) and (br in brand_cat_priors.index):
            p_prior = np.ones_like(probs)
            prior_row = brand_cat_priors.loc[br].astype(float)
            for c, p in prior_row.items():
                j = gindex.get(str(c), None)
                if j is not None:
                    p_prior[j] = max(to_float_scalar(p), 1e-6)
            probs *= p_prior
            s = to_float_scalar(probs.sum())
            if s > 0: probs /= s

        if has_allowed:
            idx = [gindex[c] for c in allowed if c in gindex]
            if len(idx) > 0:
                maskv = np.zeros_like(probs); maskv[idx] = probs[idx]
                s = to_float_scalar(maskv.sum())
                probs = (maskv / s) if s > 0 else maskv

        chosen = gclasses[int(np.argmax(probs))]
        raw = raw_soft_text[i]
        hints_best, best_hits = chosen, -1
        for c in topk[i]:
            hits = sum(1 for h in CATEGORY_HINTS.get(c, []) if h in raw)
            if hits > best_hits:
                hints_best, best_hits = c, hits

        final[i] = hints_best if (not has_allowed or hints_best in allowed) else chosen

    return final

# ---------------- Metrics ----------------
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
    unit_true = df_true.get("OCR_Measure", pd.Series(index=df_true.index)).apply(lambda x: x if pd.isna(x) else str(x).strip().lower())
    size_pred = pd.to_numeric(pred_df["Pred_Quantity"], errors='coerce')
    unit_pred = pred_df["Pred_Unit"].apply(lambda x: x if pd.isna(x) else str(x).strip().lower())
    size_mask = size_true.notna()
    unit_mask = unit_true.notna()
    size_acc = float(np.nanmean((size_true[size_mask] == size_pred[size_mask]).astype(float))) if size_mask.any() else np.nan
    unit_acc = float(np.nanmean((unit_true[unit_mask] == unit_pred[unit_mask]).astype(float))) if unit_mask.any() else np.nan
    res.append({"metric":"Quantity", "accuracy": size_acc, "n": int(size_mask.sum())})
    res.append({"metric":"Unit",     "accuracy": unit_acc,  "n": int(unit_mask.sum())})
    return pd.DataFrame(res)

# ===========================
# USE IN-MEMORY DFS: Train, Test (or load)
# ===========================
try:
    Train
    Test
except NameError:
    Train = pd.read_csv("/mnt/data/Kantar_train.csv")
    Test  = pd.read_csv("/mnt/data/Kantar_test.csv")

# Train models
feats, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats = train_global_models(Train)

# Brand resources once
brand_sorted_REF, canon_map_REF, brand_main_list_REF = build_brand_dictionary(Train)
brand_cat_priors = build_brand_cat_priors(Train, min_count=5, smoothing=0.5)  # cast to float inside

# TEST inference (ocr_text len>20)
def valid_mask_for_text(df: pd.DataFrame, text_col: str, min_len: int = 20) -> pd.Series:
    if text_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[text_col].notna() & (df[text_col].astype(str).str.strip().str.len() > min_len)

mask_test = valid_mask_for_text(Test, "ocr_text", 20)
Test_valid = Test.loc[mask_test].copy()

test_combined = build_combined_text(Test_valid["ocr_text"])
X_test = transform_char_word(feats, test_combined.tolist())

sec_pred = sec_clf.predict(X_test).tolist()

# Defensive coercions
soft_txt       = Test_valid["ocr_text"].fillna("").astype(str).map(soft_clean).tolist()
raw_brand_txt  = Test_valid["ocr_text"].fillna("").astype(str).tolist()
sec_pred_list  = list(map(str, sec_pred))
sec2cats_safe  = {str(k): (list(v) if not isinstance(v, (pd.Series, pd.Index)) else v.astype(str).tolist())
                  for k, v in sec2cats.items()}

# Categoría predictions
cat_pred = category_ensemble_predict_sharedX(
    X_test,
    soft_txt,
    raw_brand_txt,
    sec_pred_list,
    cat_clf,
    sec_cat_models,
    sec2cats_safe,
    brand_sorted_REF,
    canon_map_REF,
    brand_main_list_REF,
    brand_cat_priors=brand_cat_priors,
    lexicon_log_boost=0.35
)
sec_final = [cat2sec.get(c, s) for c, s in zip(cat_pred, sec_pred_list)]

# Brand & size (optional)
def predict_brand_and_size(train_df: pd.DataFrame, df_to_predict: pd.DataFrame, pred_cat: List[str], text_col: str):
    brand_sorted, canon_map, brand_main_list = build_brand_dictionary(train_df)
    cat_brand_counts = (train_df.groupby(['Categoría','Marca']).size()
                        .groupby(level=0).apply(lambda s: (s/s.sum()).to_dict()).to_dict())
    unit_priors = {}
    if "OCR_Measure" in train_df.columns:
        unit_priors = (
            train_df[["Categoría", "OCR_Measure"]]
            .dropna()
            .assign(u=lambda d: d["OCR_Measure"].astype(str).str.strip().str.lower())
            .groupby("Categoría")["u"]
            .apply(lambda s: s.value_counts(normalize=True).to_dict())
            .to_dict()
        )
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
        def score(c): v,u,b = c; return b + 0.6*pri.get(str(u).lower(), 0.0)
        best = max(cands, key=score); return (best[0], best[1])
    qty, unit = zip(*[pick_size_for_category(t, c) for t, c in zip(raw_txt, pred_cat)])
    return list(pred_brand), list(qty), list(unit)

brand_pred, qty_pred, unit_pred = predict_brand_and_size(Train, Test_valid, cat_pred, text_col="ocr_text")

# Assemble TEST output
out_test = Test.copy()
for col in ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]:
    out_test[col] = np.nan
out_test.loc[mask_test, "Pred_Sector"]    = np.array(sec_final, dtype=object)
out_test.loc[mask_test, "Pred_Categoría"] = np.array(cat_pred, dtype=object)
out_test.loc[mask_test, "Pred_Marca"]     = np.array(brand_pred, dtype=object)
out_test.loc[mask_test, "Pred_Quantity"]  = np.array(qty_pred, dtype=object)
out_test.loc[mask_test, "Pred_Unit"]      = np.array(unit_pred, dtype=object)

# Metrics on TEST (if labels exist)
results_test = evaluate_df(
    Test.loc[mask_test],
    out_test.loc[mask_test, ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]]
)
print(f"\nValid TEST rows (ocr_text len>20): {mask_test.sum()} / {len(Test)}")
print("\n=== Non-LLM Metrics on TEST (hardened) ===")
with pd.option_context('display.max_colwidth', 200):
    print(results_test.to_string(index=False))

# TRAIN sanity (optional)
if "clean_description" in Train.columns:
    mask_train = Train["clean_description"].notna() & (Train["clean_description"].astype(str).str.strip().str.len() > 20)
    if mask_train.any():
        Train_valid = Train.loc[mask_train].copy()
        train_combined = build_combined_text(Train_valid["clean_description"])
        X_train_eval = transform_char_word(feats, train_combined.tolist())

        sec_pred_tr = sec_clf.predict(X_train_eval).tolist()
        soft_txt_tr = Train_valid["clean_description"].fillna("").astype(str).map(soft_clean).tolist()
        raw_brand_txt_tr = Train_valid["clean_description"].fillna("").astype(str).tolist()

        sec_pred_tr_list = list(map(str, sec_pred_tr))
        sec2cats_tr_safe = {str(k): (list(v) if not isinstance(v, (pd.Series, pd.Index)) else v.astype(str).tolist())
                            for k, v in sec2cats.items()}

        cat_pred_tr = category_ensemble_predict_sharedX(
            X_train_eval,
            soft_txt_tr,
            raw_brand_txt_tr,
            sec_pred_tr_list,
            cat_clf,
            sec_cat_models,
            sec2cats_tr_safe,
            brand_sorted_REF,
            canon_map_REF,
            brand_main_list_REF,
            brand_cat_priors=brand_cat_priors,
            lexicon_log_boost=0.35
        )
        sec_final_tr = [cat2sec.get(c, s) for c, s in zip(cat_pred_tr, sec_pred_tr_list)]
        pred_train_df = pd.DataFrame({
            "Pred_Sector": sec_final_tr,
            "Pred_Categoría": cat_pred_tr,
            "Pred_Marca": [None]*len(sec_final_tr),
            "Pred_Quantity": [np.nan]*len(sec_final_tr),
            "Pred_Unit": [np.nan]*len(sec_final_tr)
        }, index=Train_valid.index)

        results_train = evaluate_df(Train_valid, pred_train_df)
        print(f"\nValid TRAIN rows (clean_description len>20): {mask_train.sum()} / {len(Train)}")
        print("\n=== Non-LLM Metrics on TRAIN (sanity) ===")
        with pd.option_context('display.max_colwidth', 200):
            print(results_train.to_string(index=False))














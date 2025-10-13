# -*- coding: utf-8 -*-
"""
FULL PIPELINE with FEW-SHOT BANK (codes, sector & confusion pairs)
Memory ➜ Baseline ➜ LLM Router (token-min) + Few-shot JSON builder/loader

Assumptions (already in memory):
    Train: DataFrame with columns:
        clean_description (str), Sector (str), Categoría (str), Marca (str)
        optional: OCR_Size, OCR_Measure
    Test:  DataFrame with columns:
        ocr_text (str)
        optional labels: Sector, Categoría, Marca, OCR_Size, OCR_Measure

Env (only required if you want LLM fallback enabled):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT         (e.g., gpt-4o-mini)
    AZURE_OPENAI_DEPLOYMENT_BIG     (optional; e.g., gpt-4o)
    AZURE_OPENAI_EMBEDDING          (optional; e.g., text-embedding-3-small)

pip install:
    pandas numpy scikit-learn scipy unidecode rapidfuzz openai joblib
"""

# ===========================
# Imports & setup
# ===========================
import os, re, math, json, hashlib, random, pathlib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from difflib import get_close_matches

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize as sknorm

# Thread caps (tune to node vCPU)
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

# ===========================
# Cleaning & tokenization
# ===========================
NOISE_PATTERNS = [
    r"exceso en [a-záéíóúüñ ]+",
    r"cont\.?\s*neto",
    r"lote[:\s]\S+",
    r"fecha\s*(de)?\s*(vencimiento|caducidad)[:\s]\S+",
    r"hecho en [a-záéíóúüñ ]+",
    r"importado por [a-záéíóúüñ ]+",
    r"\bwww\.\S+", r"\b@\S+",
    r"\b\d{1,2}\s*(oct|nov|dic|ene|feb|mar|abr|may|jun|jul|ago|sep)\b"
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.I)
UNIT_RE = re.compile(r'\b(?:kg|g|gr|ml|l|lt|oz|un|u|uds)\b', re.I)
NUM_RE  = re.compile(r'\b\d{1,4}(?:[.,]\d{1,3})?\b')
CAPS_RE = re.compile(r'\b[A-ZÁÉÍÓÚÜÑ0-9][A-ZÁÉÍÓÚÜÑ0-9\-]{2,}\b')
SPAN_KEEPERS = [re.compile(r'\b(cafe|chocolate|cocoa|jabon|detergente|shampoo|pasta|galleta|yogurt|leche|aceite|arroz|azucar)\b', re.I)]

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

def valid_mask_for_text(df: pd.DataFrame, text_col: str, min_len: int = 20) -> pd.Series:
    return df[text_col].notna() & (df[text_col].astype(str).str.strip().str.len() > min_len)

def compress_ocr_text(s: str, max_chars: int = 380) -> str:
    if not isinstance(s, str): return ""
    s = unidecode(s)
    s = NOISE_RE.sub(" ", s)
    s = re.sub(r'\s+', ' ', s.strip())
    out = []
    for t in s.split(' '):
        if len(t) > 30: continue
        if CAPS_RE.search(t) or UNIT_RE.search(t) or NUM_RE.search(t):
            out.append(t); continue
        if any(rx.search(t) for rx in SPAN_KEEPERS):
            out.append(t); continue
        if len(out) < 80 and t.isalpha() and t.lower() not in {'de','la','el','y','con','para','en','del','por','a','the','and','of'} and len(t)>3:
            out.append(t)
    comp = ' '.join(out)
    if not comp: comp = s[:max_chars]
    return comp[:max_chars]

# ===========================
# Size / Unit extraction
# ===========================
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

# ===========================
# Brand helpers
# ===========================
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

CAPS_BRAND_RE = re.compile(r'\b[A-Z][A-Z0-9ÁÉÍÓÚÜÑ\-]{2,}\b')

def find_brand_in_text_exact_or_caps(text: Any, brand_sorted, canon_map, brand_main_list) -> Optional[str]:
    if not isinstance(text, str) or not text.strip(): return None
    t_norm = normalize_text(text)
    for b in brand_sorted:
        if b and b in t_norm:
            return canon_map.get(b, b)
    caps = CAPS_BRAND_RE.findall(text.upper())
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

# ===========================
# Vectorizer + Models (Baseline)
# ===========================
def fit_char_vectorizer(X_text: List[str], min_df=3, max_features=250_000):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,6),
        min_df=min_df, max_features=max_features, dtype=np.float32
    )
    X = vec.fit_transform(X_text).astype(np.float32, copy=False)
    return vec, X

def transform_char_vectorizer(vec: TfidfVectorizer, texts: List[str]):
    return vec.transform(texts).astype(np.float32, copy=False)

def train_global_models(train_df: pd.DataFrame):
    df = train_df.reset_index(drop=True).copy()
    tr_text = df["clean_description"].fillna("").astype(str)
    combined = build_combined_text(tr_text)
    vec, X = fit_char_vectorizer(combined.tolist(), min_df=3, max_features=250_000)

    y_cat = df["Categoría"].astype(str).tolist()
    y_sec = df["Sector"].astype(str).tolist()

    cat_clf = LinearSVC(C=5.0, class_weight="balanced", max_iter=6000, tol=2e-4)
    sec_clf = LinearSVC(C=3.0, class_weight="balanced", max_iter=5000, tol=2e-4)
    cat_clf.fit(X, y_cat); sec_clf.fit(X, y_sec)

    sec_cat_models: Dict[str, LinearSVC] = {}
    for sec, idxs in df.groupby("Sector").indices.items():
        if df.iloc[idxs]["Categoría"].nunique() < 2:
            continue
        y_sub = df.iloc[idxs]["Categoría"].astype(str).values
        X_sub = X[idxs]
        clf = LinearSVC(C=4.0, class_weight="balanced", max_iter=6000, tol=2e-4)
        clf.fit(X_sub, y_sub)
        sec_cat_models[sec] = clf

    cat2sec = df.groupby("Categoría")["Sector"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    sec2cats = df.groupby("Sector")["Categoría"].apply(lambda s: sorted(s.unique())).to_dict()
    return vec, X, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats, y_cat

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

    final = [None]*X.shape[0]
    sec_pred_arr = np.array(sec_pred)

    for sec, clf in sec_cat_models.items():
        rows = np.where(sec_pred_arr == sec)[0]
        if len(rows) == 0: continue
        Xs = X[rows]
        sdec = clf.decision_function(Xs)
        sclasses = list(clf.classes_); sindex = {c:j for j,c in enumerate(sclasses)}
        sfull = _binary_two_sided_scores_batch(sdec) if np.ndim(sdec) == 1 else sdec
        smean = sfull.mean(axis=1, keepdims=True); sstd  = sfull.std(axis=1, keepdims=True) + 1e-6
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
            final[i] = chosen

    # fallback for sectors without a per-sector model
    for i in range(len(final)):
        if final[i] is not None: continue
        sec = sec_pred[i]; allowed = sec2cats.get(sec, None)
        if not allowed:
            final[i] = gclasses[int(np.argmax(gscores[i]))]
        else:
            best_c = None; best_val = -1e9
            for c in allowed:
                gv = gstd_scores[i, gindex[c]] if c in gindex else -1e9
                if gv > best_val: best_val, best_c = gv, c
            final[i] = best_c if best_c is not None else gclasses[int(np.argmax(gscores[i]))]
    return final, gscores, gclasses

# ===========================
# Product Memory (for repeats)
# ===========================
def tokenize_for_memory(txt: str) -> List[str]:
    t = soft_clean(txt)
    t = re.sub(r"[^a-z0-9áéíóúüñ ]+", " ", t)
    toks = [w for w in t.split() if 3 <= len(w) <= 20]
    return toks

def build_memory_from_train(Train: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    memory: Dict[str, Dict[str, Any]] = {}
    for _, r in Train.iterrows():
        sec = str(r["Sector"]); cat = str(r["Categoría"]); br = strip_parentheses_brand(r.get("Marca","")).lower()
        key = hashlib.sha1(f"{sec}|{cat}|{br}".encode("utf-8")).hexdigest()
        if key not in memory:
            memory[key] = {"sector":sec, "category":cat, "brand":br,
                           "units":Counter(), "tokens":Counter(),
                           "caps":set(), "numbers":set()}
        toks = tokenize_for_memory(str(r.get("clean_description","")))
        memory[key]["tokens"].update(set(toks))
        caps = CAPS_RE.findall(str(r.get("clean_description","")).upper())
        for c in caps: memory[key]["caps"].add(c)
        if "OCR_Measure" in r:
            u = normalize_unit(r.get("OCR_Measure"))
            if pd.notna(u) and str(u).strip():
                memory[key]["units"].update([str(u)])
        for m in NUM_RE.findall(str(r.get("clean_description",""))):
            memory[key]["numbers"].add(m)
    return memory

def memory_similarity(ocr_text: str, memory_entry: Dict[str,Any]) -> float:
    toks = set(tokenize_for_memory(ocr_text))
    mem_toks = set(memory_entry["tokens"].keys())
    j = len(toks & mem_toks) / (len(toks | mem_toks) + 1e-9)
    caps = set(CAPS_RE.findall(ocr_text.upper()))
    cap_bonus = 0.05 * len(caps & memory_entry["caps"])
    nums = set(NUM_RE.findall(ocr_text))
    num_bonus = 0.02 * len(nums & memory_entry["numbers"])
    return float(j + cap_bonus + num_bonus)

def memory_lookup(ocr_text: str, memory: Dict[str, Dict[str,Any]], theta: float = 0.62) -> Optional[Dict[str,Any]]:
    best_key, best_sim = None, 0.0
    for k, v in memory.items():
        s = memory_similarity(ocr_text, v)
        if s > best_sim:
            best_key, best_sim = k, s
    if best_key and best_sim >= theta:
        return memory[best_key]
    return None

# ===========================
# Baseline prediction (non-LLM)
# ===========================
def nonllm_predict(
    Train: pd.DataFrame, TestLike: pd.DataFrame, text_col: str,
    vec: TfidfVectorizer, cat_clf: LinearSVC, sec_clf: LinearSVC,
    sec_cat_models: Dict[str,LinearSVC], sec2cats: Dict[str,List[str]], cat2sec: Dict[str,str]
) -> pd.DataFrame:
    out = pd.DataFrame(index=TestLike.index)
    comb = build_combined_text(TestLike[text_col])
    X = transform_char_vectorizer(vec, comb.tolist())
    sec_pred = sec_clf.predict(X).tolist()
    soft_txt = TestLike[text_col].fillna("").astype(str).map(soft_clean).tolist()
    cat_pred, gscores, gclasses = category_ensemble_predict_sharedX(X, soft_txt, sec_pred, cat_clf, sec_cat_models, sec2cats)
    sec_final = [cat2sec.get(c, s) for c, s in zip(cat_pred, sec_pred)]

    # brands
    brand_sorted, canon_map, brand_main_list = build_brand_dictionary(Train)
    cat_brand_counts = (Train.groupby(['Categoría','Marca']).size()
                        .groupby(level=0).apply(lambda s: (s/s.sum()).to_dict()).to_dict())
    raw_txt = TestLike[text_col].fillna("").astype(str).tolist()
    pred_brand = []
    for t, c in zip(raw_txt, cat_pred):
        b = None
        if HAVE_RAPIDFUZZ:
            b = pick_brand_with_prior(t, brand_main_list, c, cat_brand_counts, threshold=82)
        if not b:
            b = find_brand_in_text_exact_or_caps(t, brand_sorted, canon_map, brand_main_list)
        pred_brand.append(b)

    # quantity/unit using priors
    if "OCR_Measure" in Train.columns:
        unit_priors = (Train[["Categoría","OCR_Measure"]].dropna()
            .assign(u=lambda d: d["OCR_Measure"].astype(str).str.strip().str.lower())
            .groupby("Categoría")["u"].apply(lambda s: s.value_counts(normalize=True).to_dict()).to_dict())
    else:
        unit_priors = {}

    def pick_size_for_category(text_raw: str, cat_lbl: str):
        cands = extract_all_sizes(text_raw)
        if not cands: return (np.nan, np.nan)
        pri = unit_priors.get(cat_lbl, {})
        def score(c): v,u,b = c; return b + 0.6*pri.get(str(u).lower(), 0.0)
        best = max(cands, key=score); return (best[0], best[1])

    qty, unit = zip(*[pick_size_for_category(t, c) for t,c in zip(raw_txt, cat_pred)])

    out["Pred_Sector"]    = sec_final
    out["Pred_Categoría"] = cat_pred
    out["Pred_Marca"]     = pred_brand
    out["Pred_Quantity"]  = qty
    out["Pred_Unit"]      = unit
    return out

# ===========================
# Confidence/OOD for routing
# ===========================
def _softmax_row(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 1:
        p1 = 1.0/(1.0+math.exp(-float(v[0])))
        return np.array([1.0-p1, p1])
    v = v - v.max(); e = np.exp(v)
    return e/(e.sum()+1e-12)

def batch_confidence(clf, X) -> Tuple[np.ndarray, np.ndarray]:
    dec = clf.decision_function(X)
    if dec.ndim == 1:
        probs = np.vstack([_softmax_row([d]) for d in dec])
    else:
        probs = np.vstack([_softmax_row(row) for row in dec])
    top = probs.max(axis=1)
    if dec.ndim == 1:
        margins = np.abs(dec)
    else:
        srt = np.sort(dec, axis=1)[:, ::-1]
        margins = (srt[:,0]-srt[:,1])
    ent = -(probs*np.log(probs+1e-12)).sum(axis=1)
    return top, np.vstack([margins, ent]).T

def build_category_centroids(X_train, y_cat: List[str]) -> Dict[str, np.ndarray]:
    centroids = {}
    y_arr = np.array(y_cat)
    for c in np.unique(y_arr):
        idx = np.where(y_arr == c)[0]
        if len(idx)==0: continue
        centroid = X_train[idx].mean(axis=0)
        centroids[c] = sknorm(centroid)
    return centroids

def ood_distance(vec_row, centroid_vec) -> float:
    if centroid_vec is None: return 0.0
    a = sknorm(vec_row)
    cos = float(a.multiply(centroid_vec).sum())
    return 1.0 - cos

# ===========================
# Label codecs (short codes)
# ===========================
def make_label_codec(values: List[str], prefix: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    uniq = sorted(set(map(str, values)))
    code2label = {f"{prefix}{i:03d}": lab for i, lab in enumerate(uniq, start=1)}
    label2code = {lab: code for code, lab in code2label.items()}
    return code2label, label2code

# ===========================
# LLM client + caching
# ===========================
try:
    from openai import AzureOpenAI
    AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT","")
    AOAI_KEY      = os.getenv("AZURE_OPENAI_API_KEY","")
    AOAI_SMALL    = os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o-mini")
    AOAI_BIG      = os.getenv("AZURE_OPENAI_DEPLOYMENT_BIG","")
    AOAI_EMB      = os.getenv("AZURE_OPENAI_EMBEDDING","")
    aoai = AzureOpenAI(api_key=AOAI_KEY, api_version="2024-06-01", azure_endpoint=AOAI_ENDPOINT) if (AOAI_ENDPOINT and AOAI_KEY) else None
except Exception:
    aoai = None
    AOAI_SMALL, AOAI_BIG, AOAI_EMB = "", "", ""

LLM_CACHE: Dict[str, Dict[str,Any]] = {}
EMB_CACHE: Dict[str, Tuple[np.ndarray, Dict[str,Any]]] = {}

def cache_key(ocr_comp: str, sec_codes: List[str], cat_codes: List[str]) -> str:
    return hashlib.sha1(f"{ocr_comp[:280]}|S:{','.join(sec_codes)}|C:{','.join(cat_codes[:10])}".encode("utf-8")).hexdigest()

def get_embedding(text: str) -> Optional[np.ndarray]:
    if not (aoai and AOAI_EMB): return None
    try:
        r = aoai.embeddings.create(model=AOAI_EMB, input=[text])
        v = np.array(r.data[0].embedding, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v
    except Exception:
        return None

def semantic_lookup(ocr_comp: str, thresh=0.93) -> Optional[Dict[str,Any]]:
    v = get_embedding(ocr_comp)
    if v is None or not EMB_CACHE: return None
    best = None; best_sim = 0.0
    for _, (e, js) in EMB_CACHE.items():
        sim = float(np.dot(v, e))
        if sim > best_sim: best_sim, best = sim, js
    return best if best_sim >= thresh else None

# ===========================
# FEW-SHOT BANK (build, save, load, retrieve)
# ===========================
FEWSHOTS_JSON = "/mnt/data/fewshot_bank.json"

def fit_vec_for_confusions(texts: List[str], min_df=3, max_features=200_000):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=min_df, max_features=max_features, dtype=np.float32)
    X = vec.fit_transform(texts).astype(np.float32, copy=False)
    return vec, X

def build_fewshot_bank(Train: pd.DataFrame) -> Dict[str, Any]:
    SEC_code2label, SEC_label2code = make_label_codec(Train["Sector"].astype(str).tolist(), "S")
    CAT_code2label, CAT_label2code = make_label_codec(Train["Categoría"].astype(str).tolist(), "C")

    # For confusion mining
    vec_conf, X_conf = fit_vec_for_confusions(Train["clean_description"].fillna("").astype(str).tolist(),
                                              min_df=3, max_features=200_000)
    from sklearn.preprocessing import normalize as sknorm
    y_cat = Train["Categoría"].astype(str).values
    cats = sorted(pd.unique(y_cat))
    cat_to_idx = {c: np.where(y_cat==c)[0] for c in cats}
    centroids = {}
    for c, idx in cat_to_idx.items():
        if len(idx)==0: continue
        centroids[c] = sknorm(X_conf[idx].mean(axis=0))

    def nearest_categories(c: str, topk: int = 2) -> List[str]:
        if c not in centroids: return []
        base = centroids[c]
        sims = []
        for d, den in centroids.items():
            if d == c: continue
            sims.append((d, float(base.multiply(den).sum())))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [d for d,_ in sims[:topk]]

    UNIT_RE_INLINE = re.compile(r'\b(kg|g|gr|ml|l|lt|oz|un|uds|u)\b', re.I)
    def select_examples(sub_df: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
        if sub_df.empty: return []
        def score_row(r):
            txt = str(r.get("clean_description",""))
            brand = str(r.get("Marca",""))
            sc = 0
            if len(brand.strip())>1: sc += 2
            if re.search(r'\d', txt) and UNIT_RE_INLINE.search(txt): sc += 2
            if len(txt) > 45: sc += 1
            return sc
        tmp = sub_df.copy()
        tmp["__score__"] = tmp.apply(score_row, axis=1)
        tmp = tmp.sort_values(["__score__"], ascending=False)
        seen_brands, picks = set(), []
        for _, r in tmp.iterrows():
            b = str(r.get("Marca","")).strip().lower()
            if b in seen_brands and len(picks) < n-1:
                continue
            seen_brands.add(b)
            picks.append({
                "ocr": compress_ocr_text(str(r.get("clean_description","")), 320),
                "sector_code": SEC_label2code.get(str(r.get("Sector","")), "S000"),
                "cat_code": CAT_label2code.get(str(r.get("Categoría","")), "C000"),
                "brand": str(r.get("Marca","")),
                "qty": str(r.get("OCR_Size","")) if "OCR_Size" in r else "",
                "unit": str(r.get("OCR_Measure","")) if "OCR_Measure" in r else "",
            })
            if len(picks) >= n: break
        if len(picks) < n:
            for _, r in tmp.iloc[len(picks):len(picks)+(n-len(picks))].iterrows():
                picks.append({
                    "ocr": compress_ocr_text(str(r.get("clean_description","")), 320),
                    "sector_code": SEC_label2code.get(str(r.get("Sector","")), "S000"),
                    "cat_code": CAT_label2code.get(str(r.get("Categoría","")), "C000"),
                    "brand": str(r.get("Marca","")),
                    "qty": str(r.get("OCR_Size","")) if "OCR_Size" in r else "",
                    "unit": str(r.get("OCR_Measure","")) if "OCR_Measure" in r else "",
                })
                if len(picks) >= n: break
        return picks[:n]

    fewshot_bank: Dict[str, Any] = {"by_sector": {}, "by_confusion": {}}

    # By Sector
    for sec, grp in Train.groupby("Sector"):
        fewshot_bank["by_sector"][SEC_label2code.get(str(sec), "S000")] = select_examples(grp, n=3)

    # By Confusion pairs
    seen_pairs = set()
    for c in cats:
        neighs = nearest_categories(c, topk=2)
        for ncat in neighs:
            a = CAT_label2code.get(c, "C000")
            b = CAT_label2code.get(ncat, "C000")
            if a == "C000" or b == "C000": continue
            key = f"{a}|{b}" if a < b else f"{b}|{a}"
            if key in seen_pairs: continue
            sub = Train[Train["Categoría"].astype(str).isin([c, ncat])]
            fewshot_bank["by_confusion"][key] = select_examples(sub, n=3)
            seen_pairs.add(key)

    fewshot_bank["_meta"] = {
        "sector_labels": SEC_code2label,
        "category_labels": CAT_code2label,
        "built_from_rows": int(len(Train)),
        "note": "Examples use codes to minimize tokens. OCR is compressed; quantities/units may be empty."
    }

    # save JSON
    path = pathlib.Path(FEWSHOTS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fewshot_bank, f, ensure_ascii=False, indent=2)
    return fewshot_bank

def load_fewshot_bank() -> Dict[str, Any]:
    p = pathlib.Path(FEWSHOTS_JSON)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def fewshots_for_request(sector_code: str, candidate_cat_codes: List[str], bank: Dict[str, Any], limit: int = 3) -> List[Dict[str,str]]:
    shots: List[Dict[str,str]] = []
    if bank.get("by_sector") and sector_code in bank["by_sector"]:
        shots.extend(bank["by_sector"][sector_code][:limit])
    if len(shots) < limit and "by_confusion" in bank:
        cands = sorted(set(candidate_cat_codes))
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                a, b = cands[i], cands[j]
                key = f"{a}|{b}" if a < b else f"{b}|{a}"
                if key in bank["by_confusion"]:
                    for ex in bank["by_confusion"][key]:
                        shots.append(ex)
                        if len(shots) >= limit: break
            if len(shots) >= limit: break
    if len(shots) < limit and bank.get("by_sector"):
        for lst in bank["by_sector"].values():
            for ex in lst:
                shots.append(ex)
                if len(shots) >= limit: break
            if len(shots) >= limit: break
    return shots[:limit]

# ===========================
# LLM prompt/call (codes only)
# ===========================
SYS = ("Eres un clasificador de productos de retail. "
       "Devuelve SOLO un JSON con claves: sector_code, cat_code, marca, quantity, unit. "
       "Usa UNICAMENTE el OCR. Si no aparece explícito, usa null.")

def build_prompt_codes(ocr_text_comp: str, sector_codes: List[str], category_codes: List[str], fewshots: List[Dict[str,str]]) -> List[Dict[str,str]]:
    s = []
    for ex in fewshots:
        s.append(
            f"\n### ejemplo\n"
            f"OCR: {ex['ocr']}\n"
            f"JSON: {{\"sector_code\":\"{ex['sector_code']}\",\"cat_code\":\"{ex['cat_code']}\","
            f"\"marca\":\"{ex['brand']}\",\"quantity\":{json.dumps(ex['qty'])},\"unit\":{json.dumps(ex['unit'])}}}"
        )
    user = (f"Sectores candidatos (codigos): {','.join(sector_codes[:8])}\n"
            f"Categorias candidatas (codigos): {','.join(category_codes[:12])}\n"
            f"{''.join(s)}\n"
            f"### consulta\n"
            f"OCR: {ocr_text_comp}\n"
            f"Responde SOLO con JSON.")
    return [{"role":"system","content":SYS},{"role":"user","content":user}]

def call_aoai_json(messages, deployment: str, max_tokens=110, temperature=0.0) -> Dict[str,Any]:
    if aoai is None:
        return {}
    resp = aoai.chat.completions.create(
        model=deployment,
        messages=messages,
        response_format={"type":"json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {}

# ===========================
# Routing thresholds & policy
# ===========================
def routing_decisions(
    TestLike: pd.DataFrame, text_col: str,
    vec, cat_clf, sec_clf, y_cat_labels: List[str],
    base_pred_df: pd.DataFrame,
    cat_conf_thr=0.60, sec_conf_thr=0.72, margin_thr=0.35, entropy_thr=0.65, ood_thr=0.20,
    max_fraction=0.25, X_override: csr_matrix = None
) -> Tuple[pd.Index, csr_matrix, Dict[str,np.ndarray]]:
    comb = build_combined_text(TestLike[text_col])
    X = transform_char_vectorizer(vec, comb.tolist()) if X_override is None else X_override
    CAT_CENTROIDS = build_category_centroids(X_train_full, y_cat_labels)  # use train centroids

    cat_top, aux = batch_confidence(cat_clf, X); margins, entropy = aux[:,0], aux[:,1]
    sec_top, _ = batch_confidence(sec_clf, X)

    idx = TestLike.index
    pred_cat = base_pred_df.loc[idx, "Pred_Categoría"].astype(str).values
    ood = np.array([ood_distance(X[i], CAT_CENTROIDS.get(pred_cat[i])) for i in range(X.shape[0])])

    low_conf = (cat_top < cat_conf_thr) | (sec_top < sec_conf_thr)
    weak_margin = (margins < margin_thr)
    diffuse = (entropy > entropy_thr)
    brand_missing = base_pred_df.loc[idx, "Pred_Marca"].isna().values
    need = low_conf | weak_margin | diffuse | brand_missing | (ood > ood_thr)

    cap = int(max(1, math.floor(len(idx) * max_fraction)))
    score = (0.5*(margin_thr - margins)) + (0.3*(entropy - entropy_thr)) + (0.2*(ood - ood_thr))
    order = np.argsort(-score)  # hardest first
    chosen_ranked = idx[order][:cap]
    chosen = idx[np.where(need)[0]].intersection(chosen_ranked)
    return chosen, X, CAT_CENTROIDS

# ===========================
# Metrics
# ===========================
def eval_cls(y_true: pd.Series, y_pred: pd.Series, name: str) -> Dict[str, Any]:
    mask = ~pd.isna(y_true)
    if mask.sum() == 0:
        return {"metric": name, "weighted_f1": np.nan, "accuracy": np.nan, "n": 0}
    y_t = y_true[mask].astype(str); y_p = y_pred[mask].astype(str)
    return {"metric": name,
            "weighted_f1": float(f1_score(y_t, y_p, average='weighted')),
            "accuracy": float(accuracy_score(y_t, y_p)),
            "n": int(mask.sum())}

def evaluate_df(df_true: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    res = []
    res.append(eval_cls(df_true.get("Sector"),    pred_df["Pred_Sector"],    "Sector"))
    res.append(eval_cls(df_true.get("Categoría"), pred_df["Pred_Categoría"], "Categoría"))
    res.append(eval_cls(df_true.get("Marca"),     pred_df["Pred_Marca"].fillna("N/A"), "Marca"))
    size_true = pd.to_numeric(df_true.get("OCR_Size", pd.Series(index=df_true.index)), errors='coerce')
    unit_true = df_true.get("OCR_Measure", pd.Series(index=df_true.index)).apply(normalize_unit)
    size_pred = pd.to_numeric(pred_df["Pred_Quantity"], errors='coerce')
    unit_pred = pred_df["Pred_Unit"].apply(normalize_unit)
    size_mask = size_true.notna(); unit_mask = unit_true.notna()
    size_acc = float(np.nanmean((size_true[size_mask] == size_pred[size_mask]).astype(float))) if size_mask.any() else np.nan
    unit_acc = float(np.nanmean((unit_true[unit_mask] == unit_pred[unit_mask]).astype(float))) if unit_mask.any() else np.nan
    res.append({"metric":"Quantity", "accuracy": size_acc, "n": int(size_mask.sum())})
    res.append({"metric":"Unit",     "accuracy": unit_acc,  "n": int(unit_mask.sum())})
    return pd.DataFrame(res)

# ===========================
# RUN: Fit artifacts, build few-shot bank, validate on Train, infer on Test
# ===========================
assert 'Train' in globals() and 'Test' in globals(), "Please define Train and Test DataFrames in memory."

# 1) FEW-SHOT BANK (build & save JSON)
FEWSHOTS = build_fewshot_bank(Train)
print(f"Few-shot bank saved to: {FEWSHOTS_JSON}")

# 2) MEMORY from Train
MEMORY = build_memory_from_train(Train)

# 3) Split Train for validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(sss.split(Train["clean_description"].fillna(""), Train["Categoría"].astype(str)))
Train_tr = Train.iloc[tr_idx].reset_index(drop=True)
Train_va = Train.iloc[va_idx].reset_index(drop=True)

# 4) Train baseline models on Train_tr
vec, X_train_full, cat_clf, sec_clf, sec_cat_models, cat2sec, sec2cats, y_cat_labels = train_global_models(Train_tr)

# 5) Label codecs for LLM codes
SEC_c2l, SEC_l2c = make_label_codec(Train_tr["Sector"].astype(str).tolist(), "S")
CAT_c2l, CAT_l2c = make_label_codec(Train_tr["Categoría"].astype(str).tolist(), "C")
CAT_CENTROIDS_GLOBAL = build_category_centroids(X_train_full, y_cat_labels)

# 6) VALIDATION on Train_va (memory ➜ baseline ➜ LLM router)
mask_va = valid_mask_for_text(Train_va, "clean_description", 20)
va_df = Train_va.loc[mask_va].copy()

# Memory pass
mem_hit = [memory_lookup(t, MEMORY, theta=0.62) for t in va_df["clean_description"].astype(str)]
mem_sec = [m["sector"] if m else None for m in mem_hit]
mem_cat = [m["category"] if m else None for m in mem_hit]
mem_brand = [m["brand"] if m else None for m in mem_hit]

mask_no_mem = pd.Series([m is None for m in mem_hit], index=va_df.index)
base_pred_va = nonllm_predict(Train_tr, va_df.loc[mask_no_mem], "clean_description",
                              vec, cat_clf, sec_clf, sec_cat_models, sec2cats, cat2sec)

# Stitch predictions
pred_val = pd.DataFrame(index=va_df.index)
pred_val["Pred_Sector"]    = [mem_sec[i] if m else None for i,(m) in enumerate(mem_hit)]
pred_val["Pred_Categoría"] = [mem_cat[i] if m else None for i,(m) in enumerate(mem_hit)]
pred_val["Pred_Marca"]     = [mem_brand[i] if m else None for i,(m) in enumerate(mem_hit)]
pred_val["Pred_Quantity"]  = np.nan
pred_val["Pred_Unit"]      = np.nan
pred_val.loc[mask_no_mem, ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]] = base_pred_va

# LLM Router on hard validation rows (if AOAI configured)
try:
    from openai import AzureOpenAI  # ensure import ok
    if aoai is not None:
        base_df_for_route = pred_val.copy()
        chosen, X_va, _ = routing_decisions(
            va_df, "clean_description", vec, cat_clf, sec_clf, y_cat_labels, base_df_for_route,
            cat_conf_thr=0.60, sec_conf_thr=0.72, margin_thr=0.35, entropy_thr=0.65, ood_thr=0.20, max_fraction=0.25
        )
        sec_codes_all = list(SEC_l2c.values())[:8]
        for idx in chosen:
            raw = str(va_df.at[idx, "clean_description"])
            ocr_comp = compress_ocr_text(raw, 380)
            # candidates from model top-k (cheap & accurate)
            Xi = X_va[va_df.index.get_loc(idx)]
            dec = cat_clf.decision_function(Xi)
            classes = list(cat_clf.classes_)
            if np.ndim(dec) == 1:
                top_codes = [CAT_l2c[c] for c in classes if c in CAT_l2c]
            else:
                ids = np.argsort(-dec[0])[:12]
                labs = [classes[i] for i in ids]
                top_codes = [CAT_l2c[c] for c in labs if c in CAT_l2c]

            # few-shots from saved bank
            # guess sector by majority sector among these candidate labels
            cand_labels = [CAT_c2l.get(c) for c in top_codes if c in CAT_c2l]
            sec_votes = Counter(Train_tr[Train_tr["Categoría"].astype(str).isin(cand_labels)]["Sector"].astype(str))
            sec_guess = sec_votes.most_common(1)[0][0] if sec_votes else None
            sec_guess_code = SEC_l2c.get(sec_guess, "S000") if sec_guess else None

            shots = fewshots_for_request(sec_guess_code, top_codes, FEWSHOTS, limit=3)
            key = cache_key(ocr_comp, sec_codes_all, top_codes)
            js = LLM_CACHE.get(key) or semantic_lookup(ocr_comp)
            if js is None:
                msgs = build_prompt_codes(ocr_comp, sec_codes_all, top_codes, shots)
                js = call_aoai_json(msgs, AOAI_SMALL, max_tokens=110, temperature=0.0)
                LLM_CACHE[key] = js
                if AOAI_EMB:
                    v = get_embedding(ocr_comp)
                    if v is not None: EMB_CACHE[key] = (v, js)
            sec_code, cat_code = js.get("sector_code"), js.get("cat_code")
            sec_label = SEC_c2l.get(sec_code) if sec_code else None
            cat_label = CAT_c2l.get(cat_code) if cat_code else None
            if sec_label: pred_val.at[idx, "Pred_Sector"] = sec_label
            if cat_label: pred_val.at[idx, "Pred_Categoría"] = cat_label
            if js.get("marca") not in (None,"","null"): pred_val.at[idx, "Pred_Marca"] = js["marca"]
            if js.get("quantity") not in (None,"","null"): pred_val.at[idx, "Pred_Quantity"] = js["quantity"]
            if js.get("unit") not in (None,"","null"): pred_val.at[idx, "Pred_Unit"] = normalize_unit(js["unit"])
except Exception:
    pass  # AOAI not configured; skip LLM layer during validation

# VALIDATION metrics
metrics_val = evaluate_df(va_df, pred_val)
print("\n=== Validation metrics on Train holdout (clean_description len>20) ===")
print(metrics_val.to_string(index=False))

# 7) TEST inference (ocr_text len>20) — Memory ➜ Baseline ➜ LLM Router
mask_test = valid_mask_for_text(Test, "ocr_text", 20)
Test_valid = Test.loc[mask_test].copy()

# Memory pass
mem_hit_t = [memory_lookup(t, MEMORY, theta=0.62) for t in Test_valid["ocr_text"].astype(str)]
mem_sec_t = [m["sector"] if m else None for m in mem_hit_t]
mem_cat_t = [m["category"] if m else None for m in mem_hit_t]
mem_brand_t = [m["brand"] if m else None for m in mem_hit_t]
mask_no_mem_t = pd.Series([m is None for m in mem_hit_t], index=Test_valid.index)

# Baseline on non-memory
base_pred_test = nonllm_predict(Train_tr, Test_valid.loc[mask_no_mem_t], "ocr_text",
                                vec, cat_clf, sec_clf, sec_cat_models, sec2cats, cat2sec)

# Stitch predictions
pred_test = pd.DataFrame(index=Test.index, columns=["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"])
pred_test.loc[mask_test, "Pred_Sector"]    = mem_sec_t
pred_test.loc[mask_test, "Pred_Categoría"] = mem_cat_t
pred_test.loc[mask_test, "Pred_Marca"]     = mem_brand_t
pred_test.loc[mask_no_mem_t.index[mask_no_mem_t], ["Pred_Sector","Pred_Categoría","Pred_Marca","Pred_Quantity","Pred_Unit"]] = base_pred_test.values

# LLM router on hard Test rows (if AOAI configured)
try:
    from openai import AzureOpenAI
    if aoai is not None:
        base_df_for_route_t = pred_test.loc[mask_test].copy()
        chosen_t, X_test_valid, _ = routing_decisions(
            Test_valid, "ocr_text", vec, cat_clf, sec_clf, y_cat_labels,
            base_df_for_route_t,
            cat_conf_thr=0.60, sec_conf_thr=0.72, margin_thr=0.35, entropy_thr=0.65, ood_thr=0.20, max_fraction=0.25
        )
        sec_codes_all = list(SEC_l2c.values())[:8]
        for idx in chosen_t:
            raw = str(Test.at[idx, "ocr_text"]); ocr_comp = compress_ocr_text(raw, 380)
            # candidate cat codes from model top-k
            Xi = X_test_valid[Test_valid.index.get_loc(idx)]
            dec = cat_clf.decision_function(Xi)
            classes = list(cat_clf.classes_)
            if np.ndim(dec) == 1:
                top_codes = [CAT_l2c[c] for c in classes if c in CAT_l2c]
            else:
                ids = np.argsort(-dec[0])[:12]
                labs = [classes[i] for i in ids]
                top_codes = [CAT_l2c[c] for c in labs if c in CAT_l2c]
            # sector guess from candidates
            cand_labels = [CAT_c2l.get(c) for c in top_codes if c in CAT_c2l]
            sec_votes = Counter(Train_tr[Train_tr["Categoría"].astype(str).isin(cand_labels)]["Sector"].astype(str))
            sec_guess = sec_votes.most_common(1)[0][0] if sec_votes else None
            sec_guess_code = SEC_l2c.get(sec_guess, "S000") if sec_guess else None

            shots = fewshots_for_request(sec_guess_code, top_codes, FEWSHOTS, limit=3)
            key = cache_key(ocr_comp, sec_codes_all, top_codes)
            js = LLM_CACHE.get(key) or semantic_lookup(ocr_comp)
            if js is None:
                msgs = build_prompt_codes(ocr_comp, sec_codes_all, top_codes, shots)
                js = call_aoai_json(msgs, AOAI_SMALL, max_tokens=110, temperature=0.0)
                LLM_CACHE[key] = js
                if AOAI_EMB:
                    v = get_embedding(ocr_comp)
                    if v is not None: EMB_CACHE[key] = (v, js)
            sec_code, cat_code = js.get("sector_code"), js.get("cat_code")
            sec_label = SEC_c2l.get(sec_code) if sec_code else None
            cat_label = CAT_c2l.get(cat_code) if cat_code else None
            if sec_label: pred_test.at[idx, "Pred_Sector"] = sec_label
            if cat_label: pred_test.at[idx, "Pred_Categoría"] = cat_label
            if js.get("marca") not in (None,"","null"): pred_test.at[idx, "Pred_Marca"] = js["marca"]
            if js.get("quantity") not in (None,"","null"): pred_test.at[idx, "Pred_Quantity"] = js["quantity"]
            if js.get("unit") not in (None,"","null"): pred_test.at[idx, "Pred_Unit"] = normalize_unit(js["unit"])
except Exception:
    pass  # AOAI not configured

# TEST metrics (if labels exist)
if {"Sector","Categoría","Marca"}.issubset(set(Test.columns)):
    metrics_test = evaluate_df(Test.loc[mask_test], pred_test.loc[mask_test])
    print("\n=== Test metrics (ocr_text len>20; where labels exist) ===")
    print(metrics_test.to_string(index=False))
else:
    metrics_test = pd.DataFrame([]); print("\n(Test labels not present for full evaluation.)")

# ===========================
# (Optional) Save artifacts for future runs
# ===========================
try:
    import joblib
    ARTIFACT_DIR = pathlib.Path("/mnt/artifacts")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, ARTIFACT_DIR/"vectorizer.joblib")
    joblib.dump(sec_clf, ARTIFACT_DIR/"sec_clf.joblib")
    joblib.dump(cat_clf, ARTIFACT_DIR/"cat_clf.joblib")
    joblib.dump(sec_cat_models, ARTIFACT_DIR/"sec_cat_models.joblib")
    joblib.dump(CAT_CENTROIDS_GLOBAL, ARTIFACT_DIR/"cat_centroids.joblib")
    # brand assets & unit priors
    brand_sorted, canon_map, brand_main_list = build_brand_dictionary(Train_tr)
    cat_brand_counts = (Train_tr.groupby(['Categoría','Marca']).size()
                        .groupby(level=0).apply(lambda s: (s/s.sum()).to_dict()).to_dict())
    brand_assets = {"brand_sorted": brand_sorted, "canon_map": canon_map,
                    "brand_main_list": brand_main_list, "cat_brand_counts": cat_brand_counts}
    joblib.dump(brand_assets, ARTIFACT_DIR/"brand_assets.joblib")
    if "OCR_Measure" in Train_tr.columns:
        unit_priors = (Train_tr[["Categoría","OCR_Measure"]].dropna()
            .assign(u=lambda d: d["OCR_Measure"].astype(str).str.strip().str.lower())
            .groupby("Categoría")["u"].apply(lambda s: s.value_counts(normalize=True).to_dict()).to_dict())
        joblib.dump(unit_priors, ARTIFACT_DIR/"unit_priors.joblib")
    joblib.dump(MEMORY, ARTIFACT_DIR/"product_memory.joblib")
    # save codecs & router cfg
    label_codecs = {"SEC_code2label": SEC_c2l, "SEC_label2code": SEC_l2c,
                    "CAT_code2label": CAT_c2l, "CAT_label2code": CAT_l2c}
    with open(ARTIFACT_DIR/"label_codecs.json","w") as f:
        json.dump(label_codecs, f)
    router_cfg = {"cat_conf_thr":0.60,"sec_conf_thr":0.72,"margin_thr":0.35,"entropy_thr":0.65,"ood_thr":0.20,"llm_cap":0.25}
    with open(ARTIFACT_DIR/"router_cfg.json","w") as f:
        json.dump(router_cfg, f)
    # (optional) cache persistence
    joblib.dump(LLM_CACHE, ARTIFACT_DIR/"llm_cache.joblib")
    print(f"\nArtifacts saved under {ARTIFACT_DIR}")
except Exception as e:
    print(f"\nArtifact saving skipped: {e}")

# Objects left in memory:
# FEWSHOTS (JSON-loaded dict), MEMORY, vec, cat_clf, sec_clf, sec_cat_models,
# cat2sec, sec2cats, pred_val, metrics_val, pred_test, metrics_test

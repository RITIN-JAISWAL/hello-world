# ===========================================
# End-to-End: Train -> Save Artifacts -> Load -> External Test
# Targets: Sector, Categoría, Marca (grouped), Quantity (bucketed), Unit
# Models: LinearSVC in sklearn Pipeline (TF-IDF + OneHotEncoder)
# Metrics: Accuracy, F1 macro, F1 weighted; (optional) Size MAE/R² via Quantity proxy
# ===========================================

import os
import re
import pickle
import warnings
from typing import Any, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, r2_score
)
from joblib import dump, load

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
TRAIN_PATH = "/content/Kantar_train.csv"
TEST_PATH  = "/content/Kantar_test.csv"
ARTIFACT_DIR = "/content/artifacts"

BRAND_MIN_COUNT = 60          # keep brands with >= this many rows; else -> 'Marca_Other'
QTY_COVERAGE    = 0.90        # top quantities covering ~90% kept as classes; rest -> 'Q_Other'
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM        = (1, 2)
LINEARSVC_C        = 2.0
LINEARSVC_MAX_ITER = 5000

os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_FILES = {
    'Sector':    f"{ARTIFACT_DIR}/model_Sector.joblib",
    'Categoria': f"{ARTIFACT_DIR}/model_Categoria.joblib",
    'Marca':     f"{ARTIFACT_DIR}/model_Marca.joblib",
    'Unit':      f"{ARTIFACT_DIR}/model_Unit.joblib",
    'Quantity':  f"{ARTIFACT_DIR}/model_Quantity.joblib",
}
META_FILES = {
    'feature_cols': f"{ARTIFACT_DIR}/FEATURE_COLS.pkl",
    'keep_brands':  f"{ARTIFACT_DIR}/keep_brands.pkl",
    'top_qty':      f"{ARTIFACT_DIR}/top_qty.pkl",
}

# Minimal inline Spanish stopwords (no downloads)
SPANISH_STOP = [
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo','como',
    'más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre','también','me',
    'hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra','otros','ese','eso',
    'ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra','él','tanto','esa','estos',
    'mucho','quienes','nada','muchos','cual','poco','ella','estar','estas','algunas','algo','nosotros','mi','mis','tú',
    'te','ti','tu','tus','ellas','nosotras','vosotros','vosotras','os','mío','mía','míos','mías','tuyo','tuya','tuyos',
    'tuyas','suyo','suya','suyos','suyas','nuestro','nuestra','nuestros','nuestras','vuestro','vuestra','vuestros',
    'vuestras','esos','esas','estoy','estás','está','estamos','estáis','están','esté','estés','estemos','estéis','estén',
    'estaré','estarás','estará','estaremos','estaréis','estarán','estaría','estarías','estaríamos','estaríais','estarían',
    'estaba','estabas','estábamos','estabais','estaban','estuve','estuviste','estuvo','estuvimos','estuvisteis','estuvieron',
    'estuviera','estuvieras','estuviéramos','estuvierais','estuvieran','estuviese','estuvieses','estuviésemos','estuvieseis',
    'estuviesen','estando','estado','estada','estados','estadas','estad'
]

# ---------------------------
# HELPERS (cleaning & extraction)
# ---------------------------
def clean_product_meta(s: Any) -> str:
    if pd.isna(s):
        return 'N/A'
    return str(s).split(':')[-1].strip()

def clean_marca(s: Any) -> str:
    v = clean_product_meta(s)
    v = re.sub(r'\s*\(.*\)', '', v).strip()
    return v.split('-')[0].strip()

def extract_size_measure_from_contenido(content: Any) -> Tuple[float, str]:
    if pd.isna(content):
        return (np.nan, 'Unknown')
    text = str(content).split(':')[-1].strip()
    m = re.match(r'(\d+(?:[.,]\d+)?)\s*([A-Za-z]+)', text)
    if m:
        val = float(m.group(1).replace(',', '.'))
        unit = m.group(2).lower()
        unit_map = {'g':'gr','grs':'gr','gramos':'gr','mls':'ml','cc':'ml','uds':'un','u':'un','uni':'un'}
        unit = unit_map.get(unit, unit)
        return (val, unit)
    return (np.nan, 'Unknown')

def extract_size_unit_from_text(desc: Any) -> Tuple[Any, Any]:
    if pd.isna(desc):
        return (None, None)
    s = str(desc).strip()
    m = re.search(r'(\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?)\s*([A-Za-z]+)\b', s)
    if m:
        size_str = m.group(1).replace(',', '.')
        unit = m.group(2).lower()
        unit_map = {'g':'gr','grs':'gr','gramos':'gr','mls':'ml','cc':'ml','uds':'un','u':'un','uni':'un'}
        unit = unit_map.get(unit, unit)
        if '-' in size_str:
            low, high = map(float, size_str.split('-'))
            size_val = (low + high) / 2.0
        else:
            size_val = float(size_str)
        return (size_val, unit)
    return (None, None)

def strip_size_tokens(desc: Any) -> str:
    if pd.isna(desc):
        return ''
    s = str(desc)
    # remove tokens like "500 g", "1.5L", "12un", "6-8 un"
    s = re.sub(r'\b\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?\s*[A-Za-z]+\b', ' ', s)
    s = s.replace('N/A', ' ').replace('No Informado', ' ')
    return re.sub(r'\s+', ' ', s).strip()

def strict_attr_clean(x: Any) -> str:
    TARGET_KEYWORDS = [
        'sector','categoría','categoria','marca','contenido','unidad','measure','tamaño','tamano',
        'gramo','gramos','gr','mililitro','mililitros','ml','unidades internas','cantidad'
    ]
    if pd.isna(x):
        return '0'
    xs = str(x).lower()
    if any(k in xs for k in TARGET_KEYWORDS):
        return '0'
    return str(x).split(':')[-1].strip()

def choose_text_col(df: pd.DataFrame) -> str:
    for cand in ['clean_description','description','ocr_text','texto','Desc']:
        if cand in df.columns:
            return cand
    df['safe_text'] = ''
    return 'safe_text'

def load_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        for sep in [';', ',', '\t', '|']:
            try:
                return pd.read_csv(path, sep=sep)
            except Exception:
                continue
    return pd.read_csv(path)

# ---------------------------
# VERSION-SAFE ENCODER
# ---------------------------
def make_ohe():
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        # sklearn <= 1.1
        return OneHotEncoder(handle_unknown='ignore', sparse=True)

def make_vectorizer():
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        stop_words=list(SPANISH_STOP)  # must be list/'english'/None
    )
def quantize_quantity_labels(raw_series: pd.Series, top_qty: set | None, coverage: float = 0.90):
    """
    Convert raw numeric quantities to string labels:
    - Coerce to numeric, drop non-finite, round to nearest int
    - Keep as strings, never cast to integer dtype (avoids pandas Int64 casting issues)
    - If top_qty is None (train-time): pick the smallest set of most frequent bins
      covering `coverage` fraction (min 10 bins). Return labels and the learned top_qty.
    """
    v = pd.to_numeric(raw_series, errors='coerce')           # float with NaNs
    v = v.where(np.isfinite(v), np.nan)                      # remove inf/-inf
    q = np.rint(v)                                           # nearest integer
    # keep as strings; NaNs become 'nan' sentinel
    q_str = pd.Series(q, index=raw_series.index).apply(
        lambda x: str(int(x)) if pd.notna(x) else 'nan'
    )

    if top_qty is None:
        counts = q_str[q_str != 'nan'].value_counts()
        if counts.empty:
            learned_top = set()
        else:
            cum = counts.cumsum() / counts.sum()
            K = max(int((cum <= coverage).sum()), 10)
            learned_top = set(counts.index[:K].tolist())
        top = learned_top
    else:
        top = set(top_qty)

    labels = np.where(q_str.isin(top), q_str, 'Q_Other')
    return pd.Series(labels, index=raw_series.index), top

# ---------------------------
# FEATURE ENGINEERING (shared for train/test)
# ---------------------------
def prepare_features(df: pd.DataFrame,
                     keep_brands: set = None,
                     top_qty: set = None,
                     feature_cols_existing: List[str] = None) -> (pd.DataFrame, List[str], set, set):
    """
    Returns: (prepared_df, FEATURE_COLS, keep_brands, top_qty)
    If keep_brands/top_qty/feature_cols_existing are provided, they are used (test-time).
    Otherwise they are inferred (train-time).
    """
    d = df.copy()

    # meta cleaning
    for col in ['Sector','Categoría','Compañía','Variedad De Marca','Valor Agregado','Venta Suelta (S/N)','Marca Propia (S/N)']:
        if col in d.columns:
            d[col] = d[col].apply(clean_product_meta)
    if 'Marca' in d.columns:
        d['Marca'] = d['Marca'].apply(clean_marca)

    txt_col = choose_text_col(d)

    # contenido -> size/unit
    if 'Contenido' in d.columns:
        size_meas = d['Contenido'].apply(extract_size_measure_from_contenido)
        d['Size_from_contenido'] = [x[0] for x in size_meas]
        d['Unit_from_contenido'] = [x[1] for x in size_meas]
    else:
        d['Size_from_contenido'] = np.nan
        d['Unit_from_contenido'] = 'Unknown'

    # text-derived (fallback only)
    desc_pair = d[txt_col].apply(extract_size_unit_from_text)
    d['desc_size'] = [x[0] for x in desc_pair]
    d['desc_unit'] = [x[1] for x in desc_pair]
    d['text_clean_noleak'] = d[txt_col].apply(strip_size_tokens)

    # attributes cleaning and package
    for i in range(10, 30):
        col = f'attribute_value {i}'
        if col in d.columns:
            d[f'attr_clean_{i}'] = d[col].apply(strict_attr_clean)

    def extract_package(row: pd.Series) -> str:
        for i in range(10, 30):
            col = f'attribute_value {i}'
            if col in row.index and pd.notna(row[col]) and 'Envase:Product Packaging:' in str(row[col]):
                return clean_product_meta(row[col])
        return 'Unknown'

    if 'Package' not in d.columns:
        d['Package'] = d.apply(extract_package, axis=1)

    # keep_brands (train-time inference if None)
    if keep_brands is None:
        if 'Marca' in d.columns:
            brand_counts = d['Marca'].value_counts()
            keep_brands = set(brand_counts[brand_counts >= BRAND_MIN_COUNT].index)
        else:
            keep_brands = set()
    if 'Marca' in d.columns:
        d['Marca_Grouped'] = np.where(d['Marca'].isin(keep_brands), d['Marca'], 'Marca_Other')
    else:
        d['Marca_Grouped'] = 'Marca_Other'

    # Unit normalization
    d['Unit'] = d['Unit_from_contenido']
    mu = (d['Unit'] == 'Unknown') & d['desc_unit'].notna()
    d.loc[mu, 'Unit'] = d.loc[mu, 'desc_unit'].fillna('Unknown')
    d['Unit'] = d['Unit'].replace({'g':'gr','grs':'gr','gramos':'gr','mls':'ml','cc':'ml','uds':'un','u':'un','uni':'un'})

    # Quantity bucketing (train-time inference if None)
    d['Quantity_raw'] = d['Size_from_contenido']
    d['Quantity_raw'] = d['Size_from_contenido']
    mq = d['Quantity_raw'].isna() & d['desc_size'].notna()
    d.loc[mq, 'Quantity_raw'] = d.loc[mq, 'desc_size']
    d['Quantity_Label'], learned_top = quantize_quantity_labels(
        d['Quantity_raw'], top_qty, coverage=QTY_COVERAGE
    )
    # if we're in train-time (top_qty is None), capture what we learned
    if top_qty is None:
        top_qty = learned_top


    # Feature columns (train-time inference if None)
    if feature_cols_existing is None:
        safe_cats = [c for c in ['Marca Propia (S/N)','Venta Suelta (S/N)','Variedad De Marca','Valor Agregado','Package'] if c in d.columns]
        safe_attrs = [c for c in d.columns if c.startswith('attr_clean_')]
        FEATURE_COLS = safe_cats + safe_attrs + ['text_clean_noleak']
    else:
        FEATURE_COLS = feature_cols_existing
        # ensure all expected features exist
        for c in FEATURE_COLS:
            if c not in d.columns:
                d[c] = '' if c == 'text_clean_noleak' else '0'

    return d, FEATURE_COLS, keep_brands, top_qty

# ---------------------------
# BUILD PIPELINES
# ---------------------------
def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if c != 'text_clean_noleak']
    return ColumnTransformer(
        transformers=[
            ('txt', make_vectorizer(), 'text_clean_noleak'),
            ('cat', make_ohe(), cat_cols)
        ]
    )

def make_clf_pipeline(feature_cols: List[str]) -> Pipeline:
    return Pipeline(steps=[
        ('prep', make_preprocessor(feature_cols)),
        ('clf', LinearSVC(C=LINEARSVC_C, class_weight='balanced', max_iter=LINEARSVC_MAX_ITER))
    ])

# ---------------------------
# 1) TRAIN ON TRAINING DATA
# ---------------------------
print(">> Loading training data...")
df_train = load_csv_safely(TRAIN_PATH)

print(">> Preparing training features/targets...")
df_train_prep, FEATURE_COLS, keep_brands, top_qty = prepare_features(df_train)

# Targets to train (classification)
TARGETS = [
    ('Sector',          'Sector'),
    ('Categoría',       'Categoria'),
    ('Marca_Grouped',   'Marca'),
    ('Unit',            'Unit'),
    ('Quantity_Label',  'Quantity'),
]

# Train pipelines on FULL training set (to maximize generalization for external test)
models: Dict[str, Pipeline] = {}
for tgt_col, key in TARGETS:
    mask = ~df_train_prep[tgt_col].isna()
    X = df_train_prep.loc[mask, FEATURE_COLS]
    y = df_train_prep.loc[mask, tgt_col].astype(str).values
    pipe = make_clf_pipeline(FEATURE_COLS)
    pipe.fit(X, y)
    models[key] = pipe
    print(f"Trained model for {key} on {mask.sum()} rows.")

# ---------------------------
# SAVE ARTIFACTS
# ---------------------------
print(">> Saving artifacts...")
for key, model in models.items():
    dump(model, MODEL_FILES[key])

with open(META_FILES['feature_cols'], 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
with open(META_FILES['keep_brands'], 'wb') as f:
    pickle.dump(keep_brands, f)
with open(META_FILES['top_qty'], 'wb') as f:
    pickle.dump(top_qty, f)

print(f"Artifacts saved to: {ARTIFACT_DIR}")

# ---------------------------
# 2) LOAD ARTIFACTS & EVALUATE ON EXTERNAL TEST
# ---------------------------
print(">> Loading artifacts for external evaluation...")
loaded_models = {k: load(v) for k, v in MODEL_FILES.items()}

with open(META_FILES['feature_cols'], 'rb') as f:
    FEATURE_COLS_LOADED = pickle.load(f)
with open(META_FILES['keep_brands'], 'rb') as f:
    KEEP_BRANDS_LOADED = pickle.load(f)
with open(META_FILES['top_qty'], 'rb') as f:
    TOP_QTY_LOADED = pickle.load(f)

print(">> Loading external test data...")
df_test = load_csv_safely(TEST_PATH)

print(">> Preparing external test features (identical pipeline, no refit)...")
df_test_prep, _, _, _ = prepare_features(
    df_test,
    keep_brands=KEEP_BRANDS_LOADED,
    top_qty=TOP_QTY_LOADED,
    feature_cols_existing=FEATURE_COLS_LOADED
)

# Evaluate
def evaluate_target(models: Dict[str, Pipeline],
                    key: str,
                    df_prep: pd.DataFrame,
                    target_col_in_df: str):
    m = models[key]
    if target_col_in_df not in df_prep.columns:
        print(f"[WARN] {target_col_in_df} not found in test; skipping {key}.")
        return None
    y_true = df_prep[target_col_in_df].astype(str).values
    X_te = df_prep[FEATURE_COLS_LOADED]
    y_pred = m.predict(X_te)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    f1w = f1_score(y_true, y_pred, average='weighted')

    print(f"\n--- External Evaluation: {key} ---")
    print("Accuracy:", f"{acc:.4f}")
    print("F1 (macro):", f"{f1m:.4f}")
    print("F1 (weighted):", f"{f1w:.4f}")
    # Uncomment for detailed diagnostics (may be large for Marca):
    # print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    # print("Classification Report:\n", classification_report(y_true, y_pred))

    return pd.DataFrame({'true': y_true, 'pred': y_pred})

pred_frames: List[pd.DataFrame] = []
pred_frames.append(evaluate_target(loaded_models, 'Sector',    df_test_prep, 'Sector'))
pred_frames.append(evaluate_target(loaded_models, 'Categoria', df_test_prep, 'Categoría'))
pred_frames.append(evaluate_target(loaded_models, 'Marca',     df_test_prep, 'Marca_Grouped'))
pred_frames.append(evaluate_target(loaded_models, 'Unit',      df_test_prep, 'Unit'))
pred_frames.append(evaluate_target(loaded_models, 'Quantity',  df_test_prep, 'Quantity_Label'))

# OPTIONAL: numeric Size metrics via Quantity label proxy (if numeric available)
def to_float_or_nan(v):
    try: return float(v)
    except Exception: return np.nan

if pred_frames[-1] is not None:
    true_size = df_test_prep['Quantity_raw'].astype(float).values
    pred_qty  = pred_frames[-1]['pred'].values
    pred_size = np.array([to_float_or_nan(v) for v in pred_qty])
    mask = (~np.isnan(true_size)) & (~np.isnan(pred_size))
    if mask.sum() >= 10:
        print("\n--- External Size metrics (proxy via Quantity_Label) ---")
        print("MAE:", f"{mean_absolute_error(true_size[mask], pred_size[mask]):.2f}")
        print("R² :", f"{r2_score(true_size[mask], pred_size[mask]):.3f}")
    else:
        print("\n[INFO] Not enough numeric overlap to compute Size MAE/R² on external test.")

# Save sample predictions for quick review
samples_out = []
names = ['Sector','Categoria','Marca','Unit','Quantity']
for name, frame in zip(names, pred_frames):
    if frame is not None:
        sm = frame.head(25).copy()
        sm.columns = [f'{name}_true', f'{name}_pred']
        samples_out.append(sm)

if samples_out:
    pred_samples = pd.concat(samples_out, axis=1)
    out_path = f"{ARTIFACT_DIR}/external_test_predictions_samples.csv"
    pred_samples.to_csv(out_path, index=False)
    print(f"\nSaved external prediction samples -> {out_path}")

print("\nAll done: trained, saved artifacts, and evaluated on external test.")

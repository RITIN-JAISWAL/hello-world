# =========================================================
# FULL Marca set (no filtering) + user-facing TEST view
# =========================================================
import re, numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

DF = merged_df_cleaned.copy()

# ---------- column detection ----------
def find_col(cands, cols):
    m = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in m: return m[c.lower()]
    return None

COL_DESC     = find_col(["clean_description","description_clean","desc_clean","description"], DF.columns)
COL_SECTOR   = find_col(["Sector","sector"], DF.columns)
COL_CATEGORY = find_col(["Categoría","Categoria","category"], DF.columns)
COL_BRAND    = find_col(["Marca","brand"], DF.columns)
COL_PACKAGE  = find_col(["Venta Suelta (S/N)","Venta Suelta","package","packaging"], DF.columns)
COL_QTY_VAL  = find_col(["qty_value","quantity_value","cantidad_valor","cantidad_value"], DF.columns)
COL_QTY_UNIT = find_col(["qty_unit","quantity_unit","unidad","unit"], DF.columns)
COL_MARCA_PROPIA = find_col(["Marca Propia (S/N)","Marca Propia","private_label"], DF.columns)

assert COL_DESC is not None and COL_BRAND is not None, "Need description and Marca columns."

# ---------- snapshot for user-facing join ----------
user_cols = [c for c in ["product_id", COL_SECTOR, COL_CATEGORY, "Compañía", "Compania",
                         COL_BRAND, COL_PACKAGE, "Contenido", COL_QTY_VAL, COL_QTY_UNIT, COL_DESC]
            if c in DF.columns]
orig_snapshot = DF[user_cols].copy()
orig_snapshot["__row_id__"] = DF.index

# ---------- minimal prep (no user-visible changes) ----------
DF = DF[DF[COL_DESC].notna() & (DF[COL_DESC].str.strip() != "")]
if COL_QTY_VAL and COL_QTY_VAL in DF.columns:
    DF[COL_QTY_VAL] = pd.to_numeric(DF[COL_QTY_VAL], errors="coerce")
DF["__row_id__"] = DF.index

# ---------- features ----------
class PresenceStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        s = X[COL_DESC].fillna("").astype(str).values
        out = []
        for t in s:
            has_num = int(bool(re.search(r"\d", t)))
            digits = len(re.findall(r"\d", t))
            tokens = len(re.findall(r"\w+", t))
            units_flag = int(bool(re.search(r"(?i)\b(ml|gr|g|kg|l|lt|un|unidad|pack|paquete|botella|lata|paños?)\b", t)))
            out.append([has_num, digits, tokens, units_flag])
        return np.asarray(out)

def make_text_union(max_word_feats=300, max_char_feats=300):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3,
                           max_features=max_word_feats, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3,
                           max_features=max_char_feats)
    return FeatureUnion([("word", word), ("char", char)])

text_fx = make_text_union(300, 300)
X_text = text_fx.fit_transform(DF[COL_DESC].fillna(""))
X_num  = PresenceStats().fit_transform(DF)

cat_cols, ohe = [], None
if COL_MARCA_PROPIA: cat_cols.append(COL_MARCA_PROPIA)
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_cat = ohe.fit_transform(DF[cat_cols])
    X = sparse.hstack([X_text, X_num, X_cat], format="csr")
else:
    X = sparse.hstack([X_text, X_num], format="csr")

# L2-normalize rows so cosine ~ dot
X = normalize(X, norm="l2", copy=False)

# ---------- targets ----------
targets_cls = [c for c in [COL_SECTOR, COL_CATEGORY, COL_PACKAGE, COL_QTY_UNIT, COL_BRAND] if c]
targets_reg = [c for c in [COL_QTY_VAL] if c]

Y = DF[targets_cls + targets_reg].copy()

# ---------- split (stratify on brand if possible) ----------
strat = DF[COL_BRAND] if COL_BRAND in DF.columns else None
train_idx, test_idx = train_test_split(
    DF.index, test_size=0.2, random_state=42,
    stratify=strat if strat is not None else None
)
tr_pos = DF.index.get_indexer(train_idx); te_pos = DF.index.get_indexer(test_idx)
X_tr, X_te = X[tr_pos], X[te_pos]
Y_tr, Y_te = Y.loc[train_idx], Y.loc[test_idx]
rowid_tr = DF.loc[train_idx, "__row_id__"].values
rowid_te = DF.loc[test_idx, "__row_id__"].values

# ---------- encoders for classification heads ----------
encoders, Ytr_enc, Yte_enc = {}, {}, {}
def fit_enc(name, s):
    le = LabelEncoder(); encoders[name] = le
    return le.fit_transform(s.astype(str))

for c in [t for t in targets_cls if t != COL_BRAND]:
    Ytr_enc[c] = fit_enc(c, Y_tr[c])
    Yte_enc[c] = encoders[c].transform(Y_te[c].astype(str))

# BRAND encoder (full set)
le_brand = LabelEncoder()
y_brand_tr = le_brand.fit_transform(Y_tr[COL_BRAND].astype(str))
y_brand_te = le_brand.transform(Y_te[COL_BRAND].astype(str))

# ---------- LightGBM for non-brand heads ----------
def lgbm_cls_params(num_class=None, is_binary=False):
    if is_binary:
        return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                    n_estimators=1200, subsample=0.9, colsample_bytree=0.9,
                    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)
    return dict(objective="multiclass", num_class=num_class, learning_rate=0.07,
                num_leaves=127, n_estimators=1800, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)

def lgbm_reg_params():
    return dict(objective="regression", learning_rate=0.05, num_leaves=63,
                n_estimators=1800, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)

models = {}

for c in [t for t in targets_cls if t != COL_BRAND]:
    if len(encoders[c].classes_) <= 2:
        clf = lgb.LGBMClassifier(**lgbm_cls_params(is_binary=True))
    else:
        clf = lgb.LGBMClassifier(**lgbm_cls_params(num_class=len(encoders[c].classes_)))
    clf.fit(X_tr, Ytr_enc[c])
    models[c] = clf

if targets_reg:
    reg = lgb.LGBMRegressor(**lgbm_reg_params())
    reg.fit(X_tr, np.log1p(Y_tr[COL_QTY_VAL].astype(float).fillna(0.0)))
    models[COL_QTY_VAL] = reg

# ---------- BRAND: cosine nearest-centroid over ALL classes ----------
# Compute per-class centroids in normalized space
n_classes = len(le_brand.classes_)
# build a list of centroids as CSR rows
centroid_rows = []
for cls_id in range(n_classes):
    cls_idx = np.where(y_brand_tr == cls_id)[0]
    # mean of rows; keep sparse mean
    if cls_idx.size == 1:
        mu = X_tr[cls_idx[0]]
    else:
        mu = X_tr[cls_idx].mean(axis=0)  # returns 1 x n_features np.matrix
    mu = sparse.csr_matrix(mu)
    # re-normalize centroid to unit norm (important for cosine)
    mu = normalize(mu, norm="l2", copy=False)
    centroid_rows.append(mu)

C = sparse.vstack(centroid_rows, format="csr")  # (n_classes, n_features)

def predict_brand_centroid(X_query, centroids, batch=500):
    """Cosine ~ dot in normalized space; chunked to control memory."""
    yhat = np.empty(X_query.shape[0], dtype=np.int32)
    CT = centroids.T.tocsc()  # speed up multiplication
    n = X_query.shape[0]
    for start in range(0, n, batch):
        end = min(start + batch, n)
        # dense scores (end-start) x n_classes; keep batch small to cap RAM
        scores = X_query[start:end].dot(CT).toarray()
        yhat[start:end] = scores.argmax(axis=1).astype(np.int32)
    return yhat

yhat_brand_te = predict_brand_centroid(X_te, C, batch=400)

# ---------- predictions for non-brand heads ----------
pred_cols = {}
for c in [t for t in targets_cls if t != COL_BRAND]:
    y_enc = models[c].predict(X_te)
    pred_cols[f"pred_{c}"] = encoders[c].inverse_transform(y_enc)

# brand decode (full set)
pred_cols[f"pred_{COL_BRAND}"] = le_brand.inverse_transform(yhat_brand_te)

# regression head decode
if targets_reg:
    pred_cols[f"pred_{COL_QTY_VAL}"] = np.expm1(models[COL_QTY_VAL].predict(X_te))

pred_df = pd.DataFrame(pred_cols, index=test_idx)
pred_df["__row_id__"] = rowid_te

# ---------- user-facing TEST view (original values + predictions) ----------
test_user_view = orig_snapshot.merge(pred_df, on="__row_id__", how="inner")

# ---------- quick metrics ----------
def quick_cls_report(true_s, pred_s, name):
    acc = accuracy_score(true_s, pred_s)
    f1m = f1_score(true_s, pred_s, average="macro")
    print(f"{name}: acc={acc:.4f}  f1_macro={f1m:.4f}")

print("\n=== Quick Test Metrics (full Marca set) ===")
for c in targets_cls:
    if c in test_user_view.columns and ("pred_"+c) in test_user_view.columns:
        true_vals = merged_df_cleaned.loc[test_user_view["__row_id__"], c].astype(str).values
        pred_vals = test_user_view["pred_"+c].astype(str).values
        quick_cls_report(true_vals, pred_vals, c)

if targets_reg and ("pred_"+COL_QTY_VAL) in test_user_view.columns and COL_QTY_VAL in test_user_view.columns:
    y_true = pd.to_numeric(merged_df_cleaned.loc[test_user_view["__row_id__"], COL_QTY_VAL], errors="coerce")
    y_pred = test_user_view["pred_"+COL_QTY_VAL]
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{COL_QTY_VAL}: MAE={mae:.3f}")

# ---------- save user-facing CSV ----------
out_path = "test_predictions_user_view_full_marca.csv"
test_user_view.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved: {out_path}  (original data + pred_* columns)")



















# =========================================================
# Filter by Marca frequency (>30), train, and show TEST VIEW
# (predictions joined back to original, untransformed columns)
# =========================================================
import re
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import lightgbm as lgb

# --------- Robust column detection ---------
def find_col(cands, cols):
    m = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in m:
            return m[c.lower()]
    return None

COL_DESC     = find_col(["clean_description","description_clean","desc_clean","description"], merged_df_cleaned.columns)
COL_SECTOR   = find_col(["Sector","sector"], merged_df_cleaned.columns)
COL_CATEGORY = find_col(["Categoría","Categoria","category"], merged_df_cleaned.columns)
COL_BRAND    = find_col(["Marca","brand"], merged_df_cleaned.columns)
COL_PACKAGE  = find_col(["Venta Suelta (S/N)","Venta Suelta","package","packaging"], merged_df_cleaned.columns)
COL_QTY_VAL  = find_col(["qty_value","quantity_value","cantidad_valor","cantidad_value"], merged_df_cleaned.columns)
COL_QTY_UNIT = find_col(["qty_unit","quantity_unit","unidad","unit"], merged_df_cleaned.columns)
COL_MARCA_PROPIA = find_col(["Marca Propia (S/N)","Marca Propia","private_label"], merged_df_cleaned.columns)

assert COL_DESC is not None, "Need a description column (e.g., clean_description)."
assert COL_BRAND is not None, "Need the brand column (e.g., Marca)."

# --------- Snapshot the original rows for a user-facing join later ---------
# Keep only columns that actually exist, and also keep the original row index
user_view_cols = [c for c in [
    "product_id", COL_SECTOR, COL_CATEGORY, "Compañía", "Compania",
    COL_BRAND, COL_PACKAGE, "Contenido", COL_QTY_VAL, COL_QTY_UNIT, COL_DESC
] if c in merged_df_cleaned.columns]
original_snapshot = merged_df_cleaned[user_view_cols].copy()
original_snapshot["__row_id__"] = merged_df_cleaned.index

# --------- Filter dataset to only brands with >30 occurrences ---------
brand_counts = merged_df_cleaned[COL_BRAND].value_counts(dropna=False)
keep_brands = set(brand_counts[brand_counts > 30].index)

mask_keep = merged_df_cleaned[COL_BRAND].isin(keep_brands)
df = merged_df_cleaned.loc[mask_keep].copy()

# Keep the original index (for joining predictions back later)
df["__row_id__"] = df.index

# --------- Minimal cleaning for features (no user-visible changes) ---------
df = df[df[COL_DESC].notna() & (df[COL_DESC].str.strip() != "")]
if COL_QTY_VAL and COL_QTY_VAL in df.columns:
    df[COL_QTY_VAL] = pd.to_numeric(df[COL_QTY_VAL], errors="coerce")

# --------- Features: TF-IDF + tiny numeric + (optional) OHE for Marca Propia ---------
class PresenceStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        s = X[COL_DESC].fillna("").astype(str).values
        out = []
        for t in s:
            has_num = int(bool(re.search(r"\d", t)))
            digits = len(re.findall(r"\d", t))
            tokens = len(re.findall(r"\w+", t))
            units_flag = int(bool(re.search(r"(?i)\b(ml|gr|g|kg|l|lt|un|unidad|pack|paquete|botella|lata|paños?)\b", t)))
            out.append([has_num, digits, tokens, units_flag])
        return np.asarray(out)

def make_text_union(max_word_feats=300, max_char_feats=300):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3,
                           max_features=max_word_feats, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3,
                           max_features=max_char_feats)
    return FeatureUnion([("word", word), ("char", char)])

text_fx = make_text_union(300, 300)
X_text = text_fx.fit_transform(df[COL_DESC].fillna(""))
X_num  = PresenceStats().fit_transform(df)

cat_cols = []
if COL_MARCA_PROPIA: cat_cols.append(COL_MARCA_PROPIA)
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_cat = ohe.fit_transform(df[cat_cols])
    X = sparse.hstack([X_text, X_num, X_cat], format="csr")
else:
    ohe = None
    X = sparse.hstack([X_text, X_num], format="csr")

# --------- Build targets (only those present) ---------
targets_cls = []
if COL_SECTOR:   targets_cls.append(COL_SECTOR)
if COL_CATEGORY: targets_cls.append(COL_CATEGORY)
if COL_PACKAGE:  targets_cls.append(COL_PACKAGE)
if COL_QTY_UNIT: targets_cls.append(COL_QTY_UNIT)
# Brand is definitely included (after filtering to >30)
targets_cls.append(COL_BRAND)

targets_reg = []
if COL_QTY_VAL: targets_reg.append(COL_QTY_VAL)

Y = df[targets_cls + targets_reg].copy()

# --------- Train/Test split (keep row ids for join) ---------
stratify_col = df[COL_BRAND] if COL_BRAND in df.columns else None
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42,
    stratify=stratify_col if stratify_col is not None else None
)
X_train, X_test = X[df.index.get_indexer(train_idx)], X[df.index.get_indexer(test_idx)]
Y_train, Y_test = Y.loc[train_idx], Y.loc[test_idx]
row_id_train = df.loc[train_idx, "__row_id__"].values
row_id_test  = df.loc[test_idx, "__row_id__"].values

# --------- Label encoders per classification head ---------
encoders = {}
def fit_encode(colname, y_series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    encoders[colname] = le
    return y_enc

def decode(colname, y_enc):
    return encoders[colname].inverse_transform(y_enc)

Y_train_enc = {}
Y_test_enc  = {}
for c in targets_cls:
    Y_train_enc[c] = fit_encode(c, Y_train[c])
    # test is encoded with the same encoder; all labels exist because we stratified inside kept brands
    Y_test_enc[c]  = encoders[c].transform(Y_test[c].astype(str))

# --------- Models ---------
def lgbm_cls_params(num_class=None, is_binary=False):
    if is_binary:
        return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                    n_estimators=1500, subsample=0.9, colsample_bytree=0.9,
                    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)
    return dict(objective="multiclass", num_class=num_class, learning_rate=0.07,
                num_leaves=127, n_estimators=2000, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)

def lgbm_reg_params():
    return dict(objective="regression", learning_rate=0.05, num_leaves=63,
                n_estimators=2000, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)

models = {}

# Train non-brand heads with LightGBM
for c in targets_cls:
    if c == COL_BRAND: 
        continue
    y_tr = Y_train_enc[c]
    if len(encoders[c].classes_) <= 2:
        clf = lgb.LGBMClassifier(**lgbm_cls_params(is_binary=True))
    else:
        clf = lgb.LGBMClassifier(**lgbm_cls_params(num_class=len(encoders[c].classes_)))
    clf.fit(X_train, y_tr)
    models[c] = clf

# Brand head (fast & safe on sparse text): LinearSVC OvR
# (We already filtered to brands with >30 occurrences, so this is accurate and fast.)
brand_clf = OneVsRestClassifier(LinearSVC())
brand_clf.fit(X_train, Y_train_enc[COL_BRAND])
models[COL_BRAND] = brand_clf

# Regression head (qty value) if present
if targets_reg:
    reg = lgb.LGBMRegressor(**lgbm_reg_params())
    y_tr_reg = np.log1p(Y_train[COL_QTY_VAL].astype(float).fillna(0.0))
    reg.fit(X_train, y_tr_reg)
    models[COL_QTY_VAL] = reg

# --------- Predict on TEST and decode back to original labels ---------
pred_cols = {}
for c in targets_cls:
    y_hat_enc = models[c].predict(X_test)
    pred_cols[f"pred_{c}"] = decode(c, y_hat_enc)

if targets_reg:
    y_hat_reg = np.expm1(models[COL_QTY_VAL].predict(X_test))
    pred_cols["pred_" + COL_QTY_VAL] = y_hat_reg

pred_df = pd.DataFrame(pred_cols, index=test_idx)
pred_df["__row_id__"] = row_id_test

# --------- Build the USER VIEW (join back to original, untransformed data) ---------
test_user_view = original_snapshot.merge(pred_df, on="__row_id__", how="inner")

# Optional: quick metrics printout
def quick_report(true_s, pred_s, name):
    acc = accuracy_score(true_s, pred_s)
    f1m = f1_score(true_s, pred_s, average="macro")
    print(f"{name}: acc={acc:.4f}  f1_macro={f1m:.4f}")

print("\n=== Quick Test Metrics (only rows in test_user_view) ===")
for c in targets_cls:
    if c in test_user_view.columns and ("pred_"+c) in test_user_view.columns:
        true_vals = merged_df_cleaned.loc[test_user_view["__row_id__"], c].astype(str).values
        pred_vals = test_user_view["pred_"+c].astype(str).values
        quick_report(true_vals, pred_vals, c)

if targets_reg and ("pred_"+COL_QTY_VAL) in test_user_view.columns and COL_QTY_VAL in test_user_view.columns:
    y_true = pd.to_numeric(merged_df_cleaned.loc[test_user_view["__row_id__"], COL_QTY_VAL], errors="coerce")
    y_pred = test_user_view["pred_"+COL_QTY_VAL]
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{COL_QTY_VAL}: MAE={mae:.3f}")

# Save to CSV for the business user (exact original values + predictions side-by-side)
out_path = "test_predictions_user_view.csv"
test_user_view.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nUser-facing test predictions saved to: {out_path}")
print("Columns shown are from the ORIGINAL data plus pred_* columns.")

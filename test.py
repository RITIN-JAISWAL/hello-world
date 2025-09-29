# ============================================
# Multi-task POC for product classification
# Requires: pandas as pd, numpy as np, sklearn, scipy
# Input: merged_df_cleaned (pandas.DataFrame)
# ============================================

import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

df = merged_df_cleaned.copy()

# -----------------------------
# 1) Helpers: cleaning & pickers
# -----------------------------
def clean_label(x: str):
    if pd.isna(x):
        return np.nan
    s = str(x)
    # strip common prefixes/noise you showed in screenshots
    s = re.sub(r'(?i)\bproduct\s*branding:\s*', '', s)
    s = re.sub(r'(?i)\bproduct\s*type:\s*', '', s)
    s = re.sub(r'(?i)\bproduct\s*benefits:\s*', '', s)
    s = re.sub(r'(?i)\bbrand(?:ing)?-?name(?:\s*product)?\b', '', s)
    s = re.sub(r'\([^)]*\)', '', s)  # drop parentheticals
    s = s.replace('N/A', '').replace('No Informado', '')
    s = re.sub(r'\s+', ' ', s).strip(' -_/').strip()
    return s if s else np.nan

def find_col(candidates):
    """Finds the first existing column among candidate names (case-insensitive)."""
    lowmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    return None

# -----------------------------
# 2) Identify columns (robust)
# -----------------------------
COL_DESC = find_col(["clean_description", "description_clean", "desc_clean", "description"])
COL_SECTOR = find_col(["Sector", "sector"])
COL_CATEGORY = find_col(["Categoría", "Categoria", "category"])
COL_BRAND = find_col(["Marca", "brand"])
COL_PACKAGE = find_col(["Venta Suelta (S/N)", "Venta Suelta", "package", "packaging"])
COL_QTY_VAL = find_col(["qty_value", "quantity_value", "cantidad_valor"])
COL_QTY_UNIT = find_col(["qty_unit", "quantity_unit", "unidad"])

assert COL_DESC is not None, "Couldn't find description column (e.g., 'clean_description')."

# Optional auxiliary safe categorical features (non-leaking)
COL_MARCA_PROPIA = find_col(["Marca Propia (S/N)", "Marca Propia", "private_label"])
COL_COMPANIA = find_col(["Compañía", "Compania", "company"])  # NOTE: we will EXCLUDE it for brand head to avoid leakage

# -----------------------------
# 3) Clean targets (hierarchical)
# -----------------------------
for col in [COL_SECTOR, COL_CATEGORY, COL_BRAND, COL_PACKAGE, COL_QTY_UNIT]:
    if col is not None and col in df.columns:
        df[col] = df[col].map(clean_label)

# qty_value numeric if present
if COL_QTY_VAL is not None and COL_QTY_VAL in df.columns:
    df[COL_QTY_VAL] = pd.to_numeric(df[COL_QTY_VAL], errors="coerce")

# -----------------------------
# 4) Build features
# -----------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.col].fillna("")

class PresenceStats(BaseEstimator, TransformerMixin):
    """Simple engineered features from description: has_number, count_digits, count_tokens."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        desc = X[COL_DESC].fillna("").astype(str).values
        feats = []
        for s in desc:
            has_num = int(bool(re.search(r'\d', s)))
            digits = len(re.findall(r'\d', s))
            tokens = len(re.findall(r'\w+', s))
            units_flag = int(bool(re.search(r'(?i)\b(ml|gr|g|kg|l|lt|un|unidad|pack|paquete|botella|lata|paños?)\b', s)))
            feats.append([has_num, digits, tokens, units_flag])
        return np.asarray(feats)

def make_text_union(max_word_feats=300, max_char_feats=300):
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2),
        min_df=3, max_features=max_word_feats, strip_accents="unicode"
    )
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(3,5),
        min_df=3, max_features=max_char_feats
    )
    return FeatureUnion([
        ("word", word),
        ("char", char),
    ])

# Core text vectorizer
text_featurizer = make_text_union(max_word_feats=300, max_char_feats=300)

# Optional categorical (very light)
cat_cols = []
if COL_MARCA_PROPIA: cat_cols.append(COL_MARCA_PROPIA)
# we intentionally DO NOT add brand/company here to avoid leakage across tasks
if COL_PACKAGE and COL_PACKAGE not in [COL_SECTOR, COL_CATEGORY, COL_BRAND]:
    # Using package as a feature is risky for category/sector; we skip to stay clean.
    pass

ohe = None
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

# -----------------------------
# 5) Prepare feature matrix
# -----------------------------
# Split rows with valid description
df = df[df[COL_DESC].notna() & (df[COL_DESC].str.strip() != "")]
df = df.reset_index(drop=True)

# Fit text features
X_text = text_featurizer.fit_transform(df[COL_DESC].fillna(""))

# Engineered small numeric features
X_num = PresenceStats().fit_transform(df)

# Optional categorical features
X_cat = None
if cat_cols:
    X_cat = ohe.fit_transform(df[cat_cols])
    X = sparse.hstack([X_text, X_num, X_cat], format="csr")
else:
    X = sparse.hstack([X_text, X_num], format="csr")

print(f"Feature matrix: {X.shape}  (text {X_text.shape}, numeric {X_num.shape}" + (f", cats {X_cat.shape}" if X_cat is not None else "") + ")")

# -----------------------------
# 6) Build targets
# -----------------------------
def prep_labels(colname):
    if colname is None or colname not in df.columns: 
        return None, None, None
    y = df[colname].copy()
    mask = y.notna()
    y = y[mask]
    return y.index.values, y.values, mask

idx_sector, y_sector, m_sector = prep_labels(COL_SECTOR)
idx_category, y_category, m_category = prep_labels(COL_CATEGORY)
idx_brand, y_brand, m_brand = prep_labels(COL_BRAND)
idx_package, y_package, m_package = prep_labels(COL_PACKAGE)
idx_unit, y_unit, m_unit = prep_labels(COL_QTY_UNIT)
idx_qty, y_qty, m_qty = prep_labels(COL_QTY_VAL)  # regression

# -----------------------------
# 7) Models (fast, sparse-friendly)
# swap to LightGBM later if desired
# -----------------------------
# For multi-class with many labels, LinearSVC often performs great on TF-IDF.
def svc():
    return OneVsRestClassifier(LinearSVC())

def logreg(max_iter=4000, C=4.0):
    return LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, solver="saga", multi_class="multinomial")

MODELS = {
    "sector": svc(),      # or logreg()
    "category": svc(),    # many classes
    "brand": svc(),       # very high cardinality
    "package": logreg(),  # binary/multi
    "unit": svc(),        # units are handful of classes
    "qty_reg": Ridge(alpha=1.0)  # regression on log-quantity below
}

# -----------------------------
# 8) Cross-validation & training
# -----------------------------
def cv_classification(X, y, idx, n_splits=5, model=None, label=""):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    y_all_pred = pd.Series(index=idx, dtype=object)
    for fold, (tr, va) in enumerate(skf.split(idx, y)):
        tr_idx = idx[tr]; va_idx = idx[va]
        m = model
        m.fit(X[tr_idx], y[tr])
        yhat = m.predict(X[va_idx])
        acc = accuracy_score(y[va], yhat)
        f1m = f1_score(y[va], yhat, average="macro")
        accs.append(acc); f1s.append(f1m)
        y_all_pred.loc[va_idx] = yhat
        print(f"[{label}] Fold {fold+1}: acc={acc:.4f}  f1_macro={f1m:.4f}")
    print(f"[{label}] CV mean: acc={np.mean(accs):.4f} ±{np.std(accs):.4f} | f1={np.mean(f1s):.4f}")
    return y_all_pred, np.mean(accs), np.mean(f1s)

def cv_regression(X, y, idx, n_splits=5, model=None, label=""):
    # Use log1p transform to stabilize
    y_log = np.log1p(y.astype(float))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, mape_like = [], []
    y_all_pred = pd.Series(index=idx, dtype=float)
    for fold, (tr, va) in enumerate(kf.split(idx)):
        tr_idx = idx[tr]; va_idx = idx[va]
        m = model
        m.fit(X[tr_idx], y_log[tr])
        yhat = np.expm1(m.predict(X[va_idx]))
        mae = mean_absolute_error(y[va], yhat)
        # % within 10% tolerance
        eps = 1e-9
        pct_err = np.abs(y[va] - yhat) / (np.abs(y[va]) + eps)
        within10 = (pct_err <= 0.10).mean()
        maes.append(mae); mape_like.append(within10)
        y_all_pred.loc[va_idx] = yhat
        print(f"[{label}] Fold {fold+1}: MAE={mae:.3f}, within10%={within10:.3f}")
    print(f"[{label}] CV mean: MAE={np.mean(maes):.3f} ±{np.std(maes):.3f} | within10%={np.mean(mape_like):.3f}")
    return y_all_pred, np.mean(maes), np.mean(mape_like)

results = {}

# Sector
if idx_sector is not None:
    y_pred_sector, acc, f1m = cv_classification(X, y_sector, idx_sector, model=MODELS["sector"], label="SECTOR")
    results["sector_cv_acc"] = acc

# Category
if idx_category is not None:
    y_pred_category, acc, f1m = cv_classification(X, y_category, idx_category, model=MODELS["category"], label="CATEGORY")
    results["category_cv_acc"] = acc

# Brand
if idx_brand is not None:
    # (Optionally, you can filter to brands with >= N samples to stabilize; uncomment next two lines)
    # vc = pd.Series(y_brand).value_counts()
    # keep = vc[vc>=3].index; sel = np.isin(y_brand, keep); y_brand, idx_brand = y_brand[sel], idx_brand[sel]
    y_pred_brand, acc, f1m = cv_classification(X, y_brand, idx_brand, model=MODELS["brand"], label="BRAND")
    results["brand_cv_acc"] = acc

# Package
if idx_package is not None:
    y_pred_package, acc, f1m = cv_classification(X, y_package, idx_package, model=MODELS["package"], label="PACKAGE")
    results["package_cv_acc"] = acc

# Unit
if idx_unit is not None:
    y_pred_unit, acc, f1m = cv_classification(X, y_unit, idx_unit, model=MODELS["unit"], label="UNIT")
    results["unit_cv_acc"] = acc

# Quantity value (regression)
if idx_qty is not None:
    # Only keep positive finite targets
    ok = np.isfinite(df.loc[idx_qty, COL_QTY_VAL].values) & (df.loc[idx_qty, COL_QTY_VAL].values > 0)
    idx_qty_ok = idx_qty[ok]
    y_qty_ok = df.loc[idx_qty_ok, COL_QTY_VAL].values.astype(float)
    y_pred_qty, mae, within10 = cv_regression(X, y_qty_ok, idx_qty_ok, model=MODELS["qty_reg"], label="QTY_VALUE")
    results["qty_mae"] = mae
    results["qty_within10"] = within10

print("\n==== SUMMARY ====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# -----------------------------
# 9) Final fit on FULL data (to deploy)
# -----------------------------
trained = {}

def fit_and_store(label, idx, y, model):
    if idx is None: 
        return
    m = model
    m.fit(X[idx], y)
    trained[label] = m
    print(f"Trained final model: {label}")

if idx_sector is not None: fit_and_store("sector", idx_sector, y_sector, MODELS["sector"])
if idx_category is not None: fit_and_store("category", idx_category, y_category, MODELS["category"])
if idx_brand is not None: fit_and_store("brand", idx_brand, y_brand, MODELS["brand"])
if idx_package is not None: fit_and_store("package", idx_package, y_package, MODELS["package"])
if idx_unit is not None: fit_and_store("unit", idx_unit, y_unit, MODELS["unit"])
if idx_qty is not None:
    ok = np.isfinite(df.loc[idx_qty, COL_QTY_VAL].values) & (df.loc[idx_qty, COL_QTY_VAL].values > 0)
    trained["qty_reg"] = MODELS["qty_reg"].fit(X[idx_qty[ok]], np.log1p(df.loc[idx_qty[ok], COL_QTY_VAL].values.astype(float)))
    print("Trained final model: qty_reg (on log target)")

# -----------------------------
# 10) Inference function
# -----------------------------
def predict_products(dataframe: pd.DataFrame):
    """Return predictions for all 5 heads on a new dataframe with the same schema."""
    D = dataframe.copy()
    # clean fields
    for col in [COL_SECTOR, COL_CATEGORY, COL_BRAND, COL_PACKAGE, COL_QTY_UNIT]:
        if col is not None and col in D.columns:
            D[col] = D[col].map(clean_label)
    Xt = text_featurizer.transform(D[COL_DESC].fillna(""))
    Xn = PresenceStats().transform(D)
    if cat_cols:
        Xc = ohe.transform(D[cat_cols])
        Xall = sparse.hstack([Xt, Xn, Xc], format="csr")
    else:
        Xall = sparse.hstack([Xt, Xn], format="csr")
    out = {}
    if "sector" in trained: out["sector"] = trained["sector"].predict(Xall)
    if "category" in trained: out["category"] = trained["category"].predict(Xall)
    if "brand" in trained: out["brand"] = trained["brand"].predict(Xall)
    if "package" in trained: out["package"] = trained["package"].predict(Xall)
    if "unit" in trained: out["unit"] = trained["unit"].predict(Xall)
    if "qty_reg" in trained:
        out["qty_value_pred"] = np.expm1(trained["qty_reg"].predict(Xall))
    return pd.DataFrame(out, index=D.index)

print("\nModels are trained and `predict_products(df)` is ready for inference.")

































# ==========================================================
# LightGBM heads for all tasks (reusing X and target arrays)
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

np.random.seed(42)

# ---------- Helpers ----------
def lgbm_cls_params(num_class=None, is_binary=False):
    if is_binary:
        return dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            n_estimators=2000,
            min_data_in_leaf=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            n_jobs=-1
        )
    elif num_class and num_class > 2:
        return dict(
            objective="multiclass",
            num_class=num_class,
            learning_rate=0.07,
            num_leaves=127,
            max_depth=-1,
            n_estimators=3000,
            min_data_in_leaf=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            n_jobs=-1
        )
    else:
        # Fallback to binary if called without flag
        return lgbm_cls_params(is_binary=True)

def lgbm_reg_params():
    return dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        n_estimators=3000,
        min_data_in_leaf=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        n_jobs=-1
    )

def cv_lgbm_multiclass(X, y_str, idx, label, early_stopping_rounds=100, n_splits=5):
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    oof_pred = np.zeros_like(y)
    best_ests = []

    for fold, (tr, va) in enumerate(skf.split(idx, y)):
        tr_idx, va_idx = idx[tr], idx[va]
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr], y[va]

        clf = lgb.LGBMClassifier(**lgbm_cls_params(num_class=len(le.classes_)))
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        yhat = clf.predict(Xva)
        oof_pred[va] = yhat
        acc = accuracy_score(yva, yhat)
        f1m = f1_score(yva, yhat, average="macro")
        accs.append(acc); f1s.append(f1m)
        best_ests.append(clf.best_iteration_)
        print(f"[{label}] Fold {fold+1}: acc={acc:.4f} f1_macro={f1m:.4f} best_iter={clf.best_iteration_}")

    print(f"[{label}] CV mean: acc={np.mean(accs):.4f} ±{np.std(accs):.4f} | f1={np.mean(f1s):.4f}")
    # Final fit on full data at mean best_iteration
    final_clf = lgb.LGBMClassifier(**lgbm_cls_params(num_class=len(le.classes_)))
    final_clf.set_params(n_estimators=int(np.round(np.mean(best_ests))))
    final_clf.fit(X[idx], y)
    return {"encoder": le, "model": final_clf, "cv_acc": float(np.mean(accs)), "cv_f1": float(np.mean(f1s))}

def cv_lgbm_binary(X, y_str, idx, label, early_stopping_rounds=100, n_splits=5):
    # binary labels → encode to {0,1}
    y_series = pd.Series(y_str)
    # if labels are strings like "Envasado"/"Suelta", this will encode to 0/1
    le = LabelEncoder()
    y = le.fit_transform(y_series)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    best_ests = []

    for fold, (tr, va) in enumerate(skf.split(idx, y)):
        tr_idx, va_idx = idx[tr], idx[va]
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr], y[va]

        clf = lgb.LGBMClassifier(**lgbm_cls_params(is_binary=True))
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        yhat = clf.predict(Xva)
        acc = accuracy_score(yva, yhat)
        f1m = f1_score(yva, yhat, average="binary")
        accs.append(acc); f1s.append(f1m); best_ests.append(clf.best_iteration_)
        print(f"[{label}] Fold {fold+1}: acc={acc:.4f} f1={f1m:.4f} best_iter={clf.best_iteration_}")

    print(f"[{label}] CV mean: acc={np.mean(accs):.4f} ±{np.std(accs):.4f} | f1={np.mean(f1s):.4f}")
    final_clf = lgb.LGBMClassifier(**lgbm_cls_params(is_binary=True))
    final_clf.set_params(n_estimators=int(np.round(np.mean(best_ests))))
    final_clf.fit(X[idx], y)
    return {"encoder": le, "model": final_clf, "cv_acc": float(np.mean(accs)), "cv_f1": float(np.mean(f1s))}

def cv_lgbm_regression(X, y_float, idx, label, early_stopping_rounds=100, n_splits=5):
    y_log = np.log1p(y_float.astype(float))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, within10s = [], []
    best_ests = []

    for fold, (tr, va) in enumerate(kf.split(idx)):
        tr_idx, va_idx = idx[tr], idx[va]
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y_log[tr], y_log[va]

        reg = lgb.LGBMRegressor(**lgbm_reg_params())
        reg.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        yhat = np.expm1(reg.predict(Xva))
        mae = mean_absolute_error(np.expm1(yva), yhat)
        pct_err = np.abs(np.expm1(yva) - yhat) / (np.expm1(yva) + 1e-9)
        within10 = (pct_err <= 0.10).mean()
        maes.append(mae); within10s.append(within10); best_ests.append(reg.best_iteration_)
        print(f"[{label}] Fold {fold+1}: MAE={mae:.3f}, within10%={within10:.3f}, best_iter={reg.best_iteration_}")

    print(f"[{label}] CV mean: MAE={np.mean(maes):.3f} ±{np.std(maes):.3f} | within10%={np.mean(within10s):.3f}")
    final_reg = lgb.LGBMRegressor(**lgbm_reg_params())
    final_reg.set_params(n_estimators=int(np.round(np.mean(best_ests))))
    final_reg.fit(X[idx], y_log)
    return {"model": final_reg, "cv_mae": float(np.mean(maes)), "cv_within10": float(np.mean(within10s))}

# ---------- Train each head ----------
lgb_models = {}

# Sector
if idx_sector is not None:
    lgb_models["sector"] = cv_lgbm_multiclass(X, y_sector, idx_sector, label="LGB-Sector")

# Category
if idx_category is not None:
    lgb_models["category"] = cv_lgbm_multiclass(X, y_category, idx_category, label="LGB-Category")

# Package (binary / small multi)
if idx_package is not None:
    # detect binary vs multi
    if pd.Series(y_package).nunique() <= 2:
        lgb_models["package"] = cv_lgbm_binary(X, y_package, idx_package, label="LGB-Package")
    else:
        lgb_models["package"] = cv_lgbm_multiclass(X, y_package, idx_package, label="LGB-Package")

# Unit
if idx_unit is not None:
    lgb_models["unit"] = cv_lgbm_multiclass(X, y_unit, idx_unit, label="LGB-Unit")

# Brand (frequency threshold to tame class explosion)
if idx_brand is not None:
    BRAND_MIN_COUNT = 5  # tune: 3/5/10
    y_brand_series = pd.Series(y_brand)
    vc = y_brand_series.value_counts()
    keep = set(vc[vc >= BRAND_MIN_COUNT].index)
    y_brand_collapse = y_brand_series.where(y_brand_series.isin(keep), other="__OTHER__").values
    print(f"Brand classes kept: {len(keep)} (+1 OTHER) out of {vc.size}")
    lgb_models["brand"] = cv_lgbm_multiclass(X, y_brand_collapse, idx_brand, label="LGB-Brand (collapsed)")

# Quantity value (regression)
if idx_qty is not None:
    ok = np.isfinite(df.loc[idx_qty, COL_QTY_VAL].values) & (df.loc[idx_qty, COL_QTY_VAL].values > 0)
    idx_qty_ok = idx_qty[ok]
    y_qty_ok = df.loc[idx_qty_ok, COL_QTY_VAL].values.astype(float)
    lgb_models["qty_value"] = cv_lgbm_regression(X, y_qty_ok, idx_qty_ok, label="LGB-QtyValue")

print("\n==== LightGBM Summary ====")
for k, v in lgb_models.items():
    keys = [kk for kk in v.keys() if kk.startswith("cv_")]
    if keys:
        print(k, {kk: round(v[kk], 4) for kk in keys})

# ---------- Inference using LightGBM ----------
def predict_products_lgbm(dataframe: pd.DataFrame):
    D = dataframe.copy()
    Xt = text_featurizer.transform(D[COL_DESC].fillna(""))
    Xn = PresenceStats().transform(D)
    if cat_cols:
        Xc = ohe.transform(D[cat_cols])
        Xall = sparse.hstack([Xt, Xn, Xc], format="csr")
    else:
        Xall = sparse.hstack([Xt, Xn], format="csr")

    out = {}
    if "sector" in lgb_models:
        model = lgb_models["sector"]["model"]; le = lgb_models["sector"]["encoder"]
        out["sector"] = le.inverse_transform(model.predict(Xall))

    if "category" in lgb_models:
        model = lgb_models["category"]["model"]; le = lgb_models["category"]["encoder"]
        out["category"] = le.inverse_transform(model.predict(Xall))

    if "package" in lgb_models:
        model = lgb_models["package"]["model"]; le = lgb_models["package"]["encoder"]
        out["package"] = le.inverse_transform(model.predict(Xall))

    if "unit" in lgb_models:
        model = lgb_models["unit"]["model"]; le = lgb_models["unit"]["encoder"]
        out["unit"] = le.inverse_transform(model.predict(Xall))

    if "brand" in lgb_models:
        model = lgb_models["brand"]["model"]; le = lgb_models["brand"]["encoder"]
        pred = le.inverse_transform(model.predict(Xall))
        out["brand"] = pred  # note: may include "__OTHER__"

    if "qty_value" in lgb_models:
        reg = lgb_models["qty_value"]["model"]
        out["qty_value_pred"] = np.expm1(reg.predict(Xall))

    return pd.DataFrame(out, index=D.index)

print("LightGBM heads trained. Use `predict_products_lgbm(df)` for inference.")

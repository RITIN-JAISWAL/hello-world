# =====================================================
# Multi-task pipeline with K-Fold CV + Confusion Matrices
# =====================================================

import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, classification_report, mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict

# ---------------------------
# Feature engineering helpers
# ---------------------------
def strip_accents(s: str) -> str:
    if s is None or pd.isna(s): return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def clean_desc(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).replace("\u00A0"," ")
    s = strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_attr_pair(x):
    if pd.isna(x): return (None, None)
    s = strip_accents(str(x))
    parts = s.split(":", 1)
    if len(parts) == 1: return (parts[0].strip().lower(), None)
    return (parts[0].strip().lower(), parts[1].strip())

# Target-like labels to avoid leakage
TARGET_LABELS = {"sector","categoria","marca","envase","unidad","contenido",
                 "gramos","mililitros","litros","ml","gr"}

def build_text_input(df):
    base_desc = df["1"].astype(str).map(clean_desc) if "1" in df.columns else df["desc"].astype(str).map(clean_desc)
    attr_val_cols = [c for c in df.columns if str(c).lower().startswith("attribute_value")]
    safe_attrs = []
    for i in df.index:
        tokens = []
        for col in attr_val_cols:
            left, right = split_attr_pair(df.at[i, col])
            if left is None: continue
            if any(left.startswith(t) for t in TARGET_LABELS): continue
            if right: tokens.append(f"{left}:{right}")
            else: tokens.append(left)
        safe_attrs.append(" ".join(tokens))
    return (base_desc + " " + pd.Series(safe_attrs, index=df.index)).str.strip()

# ---------------------------
# Multi-task model
# ---------------------------
class MultiTaskModel:
    def __init__(self, targets):
        self.targets = targets
        self.heads = {}
        self.pre = None

    def fit(self, X_df, y_df):
        num_cols = [c for c in X_df.columns if c != "text_input"]
        pre = ColumnTransformer([
            ("txt", TfidfVectorizer(ngram_range=(1,3), max_features=300000, min_df=3, strip_accents="unicode"), "text_input"),
            ("num", StandardScaler(with_mean=False), num_cols)
        ], sparse_threshold=0.3)
        Xs = pre.fit_transform(X_df)

        for t in self.targets:
            if t not in y_df.columns: continue
            mask = y_df[t].notna()
            if mask.sum() == 0: continue

            if t == "Contenido":
                y_log = np.log1p(y_df.loc[mask, t].astype(float))
                est = Ridge(alpha=1.0)
                est.fit(Xs[mask], y_log)
            else:
                est = LogisticRegression(max_iter=2000, class_weight="balanced")
                est.fit(Xs[mask], y_df.loc[mask, t])
            self.heads[t] = est

        self.pre = pre
        return self

    def predict(self, X_df):
        Xs = self.pre.transform(X_df)
        out = {}
        for t, est in self.heads.items():
            if t == "Contenido":
                y_log_pred = est.predict(Xs)
                out[t] = np.expm1(y_log_pred)
            else:
                out[t] = est.predict(Xs)
        return pd.DataFrame(out, index=X_df.index)

    def evaluate(self, X_df, y_df, fold=None):
        preds = self.predict(X_df)
        results = {}
        for t in self.targets:
            if t not in preds.columns or t not in y_df.columns: continue
            mask = y_df[t].notna()
            if mask.sum() == 0: continue
            y_true, y_pred = y_df.loc[mask, t], preds.loc[mask, t]

            if t == "Contenido":
                rmse = mean_squared_error(y_true.astype(float), y_pred, squared=False)
                results[t] = {"RMSE": rmse}
                print(f"[{t}] RMSE: {rmse:.2f}")
            else:
                f1 = f1_score(y_true, y_pred, average="weighted")
                results[t] = {"F1_weighted": f1}
                print(f"\n[{t}] weighted F1: {f1:.3f}")
                print(classification_report(y_true, y_pred, digits=3))

                # Confusion Matrix plot
                cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
                plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
                plt.title(f"Confusion Matrix - {t} (Fold {fold if fold else ''})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.show()
        return results

    def save(self, path):
        joblib.dump({"pre": self.pre, "heads": self.heads, "targets": self.targets}, path)

    @classmethod
    def load(cls, path):
        bundle = joblib.load(path)
        obj = cls(bundle["targets"])
        obj.pre = bundle["pre"]
        obj.heads = bundle["heads"]
        return obj

# ---------------------------
# K-Fold CV training & evaluation
# ---------------------------
def run_kfold_pipeline(df, n_splits=5):
    df["text_input"] = build_text_input(df)
    targets = ["Sector","Categoria","Marca","Envase","Unidad","Contenido"]
    y_df = df[targets].copy() if set(targets).intersection(df.columns) else pd.DataFrame(index=df.index)
    X_df = pd.DataFrame({"text_input": df["text_input"]})

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_scores = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        print(f"\n========== Fold {fold}/{n_splits} ==========")
        X_tr, X_va = X_df.loc[train_idx], X_df.loc[val_idx]
        y_tr, y_va = y_df.loc[train_idx], y_df.loc[val_idx]

        m = MultiTaskModel(targets)
        m.fit(X_tr, y_tr)
        fold_results = m.evaluate(X_va, y_va, fold=fold)

        for t, scores in fold_results.items():
            for metric, val in scores.items():
                all_scores[f"{t}_{metric}"].append(val)

    print("\n===== Cross-Validation Results =====")
    for k, vals in all_scores.items():
        print(f"{k}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    print("\nTraining final model on full dataset...")
    final_model = MultiTaskModel(targets)
    final_model.fit(X_df, y_df)

    return final_model, all_scores

# ---------------------------
# Example call
# ---------------------------
# final_model, scores = run_kfold_pipeline(final_df, n_splits=5)
# final_model.save("multitask_model_cv.joblib")

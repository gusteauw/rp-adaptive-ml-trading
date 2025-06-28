# ============================================================
# 📁 Script: hybrid_classification_pipeline.py
# ⚡ Model Type: Hybrid Classifier
# 🧠 Features: Combined multi-source inputs
# 🔁 CV: Walk-forward Cross-Validation
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from importlib import import_module
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.multiclass import unique_labels

from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature_registry import feature_registry
from walkforward_cv import walk_forward_split
from config.paths import RESULTS_DIR

# --- CONFIGURATION --------------------------
FEATURE_SOURCES = ["technical_features", "daily_macro_features", "options_features"]
LABEL = "y_up_5d"
MODEL_TYPE = "logistic"  # Options: "logistic", "rf"

# --- LOAD & MERGE FEATURES ------------------
merged_df = None
for source_key in FEATURE_SOURCES:
    cfg = feature_registry[source_key]
    mod = import_module(cfg["script_path"].replace(".py", ""))
    X, y = mod.get_features_and_labels()

    if cfg.get("features"):
        X = X[["date"] + cfg["features"]]
    if y is not None and LABEL in y.columns:
        y = y[["date", LABEL]]
    else:
        y = None

    df = X.copy()
    if y is not None:
        df = pd.merge(df, y, on="date", how="left")

    merged_df = df if merged_df is None else pd.merge(merged_df, df, on="date", how="outer")

merged_df = merged_df.sort_values("date").dropna(subset=[LABEL]).reset_index(drop=True)
dates = merged_df["date"]
y = merged_df[LABEL]
X = merged_df.drop(columns=["date", LABEL])

print(f"\n✅ Loaded hybrid features from: {FEATURE_SOURCES}")
print(f"📐 X shape: {X.shape}, y shape: {y.shape}")

# --- SCALING -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- MODEL SELECTION -----------------------
if MODEL_TYPE == "logistic":
    model = LogisticRegression(max_iter=1000)
elif MODEL_TYPE == "rf":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Unsupported model type.")

# --- WALK-FORWARD CV -----------------------
results = []
n_splits = 5
test_size = 0.2

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X_scaled, n_splits, test_size)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": acc,
        "roc_auc": auc
    })

# --- REPORTS -------------------------------
res_df = pd.DataFrame(results)
print("\n📊 Cross-Validation Results:")
print(res_df)

print("\n📝 Classification Report (Last Fold):")
print(classification_report(y_test, y_pred))

# --- SAVE OUTPUTS --------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(RESULTS_DIR, exist_ok=True)

fname_base = f"hybrid_{MODEL_TYPE}_{LABEL}_{timestamp}"
res_df.to_csv(os.path.join(RESULTS_DIR, f"{fname_base}_cv_results.csv"), index=False)

report_str = classification_report(y_test, y_pred, target_names=[str(lbl) for lbl in unique_labels(y_test, y_pred)])
with open(os.path.join(RESULTS_DIR, f"{fname_base}_classification_report.txt"), "w") as f:
    f.write(report_str)

preds_df = pd.DataFrame({
    "date": d_test.values,
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba if y_proba is not None else np.nan
})
preds_df.to_csv(os.path.join(RESULTS_DIR, f"{fname_base}_predictions_last_fold.csv"), index=False)

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(os.path.join(RESULTS_DIR, f"{fname_base}_feature_importances.csv"))
    print("\n🌟 Top 10 Feature Importances:")
    print(importances.head(10))

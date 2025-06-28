# ============================================================
# 📁 Script: linear_classification_pipeline.py
# 🧠 Model Type: Linear Classification
# 📊 Models: Logistic Regression, Ridge Classifier, etc.
# 🔁 CV: Walk-forward or expanding window CV
# ============================================================

# --- CONFIG --------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from sklearn.preprocessing import StandardScaler
from importlib import import_module
from walkforward_cv import walk_forward_split
from config.paths import BASE_DIR, RESULTS_DIR


import os


# --- MODE SELECTION -------------------
# Choose one of the modes from the feature registry
MODE = "tech_momentum_regime"  # This is the 'mode' value
LABEL = "y_ret_5d"              # Can switch easily to 'y_up_1d', etc.
MODEL = "logistic"             # 'logistic' or 'ridge'

# --- IMPORT FEATURES ------------------
from feature_registry import feature_registry

# Lookup by 'mode' field inside registry
entry = next((cfg for cfg in feature_registry.values() if cfg["mode"] == MODE), None)
if entry is None:
    raise ValueError(f"MODE '{MODE}' not found in feature_registry. Check spelling or registry definition.")

features_script = entry["script_path"].replace(".py", "")
feature_cols = entry.get("features", [])
label_cols = entry["label"]

# Dynamically import the script as a module
mod = import_module(features_script)

# Call get_features_and_labels() function
X, y = mod.get_features_and_labels()

# --- FILTER FEATURES ------------------
if feature_cols:
    X = X[["date"] + feature_cols]
y = y[["date", LABEL]]

# Drop NA and align
df = pd.merge(X, y, on="date").dropna()
print(f"\n✅ Loaded dataset for mode: {MODE}")
print(f"📐 Shape: {df.shape} | 🎯 Target: {LABEL}")

# --- PREP DATA -------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- CHOOSE MODEL ----------------------
if MODEL == "logistic":
    model = LogisticRegression(max_iter=1000)
elif MODEL == "ridge":
    model = RidgeClassifier()
else:
    raise ValueError("Unsupported model")

# --- WALK FWD CROSS-VALIDATION ------
n_splits = 5
test_size = 0.2  # or a fixed integer like 60
results = []

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X_scaled, n_splits=n_splits, test_size=test_size)):

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": acc,
        "roc_auc": auc
    })

# --- RESULTS ---------------------------
res_df = pd.DataFrame(results)
print("\n Cross-Validation Results:")
print(res_df)

print("\n Classification Report (Last Fold):")
print(classification_report(y_test, y_pred))

#save results
output_path = os.path.join(RESULTS_DIR, f"{MODE}_{LABEL}_{MODEL}_cv_results.csv")
res_df.to_csv(output_path, index=False)
print(f"\n Results saved to: {output_path}")




# ============================================================
# Script: linear_regression_pipeline.py
# Model Type: Linear Regression
# Models: OLS, Ridge, Lasso
# CV: Walk-forward Cross-Validation
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from importlib import import_module
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from walkforward_cv import walk_forward_split
from feature_registry import feature_registry
from config.paths import RESULTS_DIR

# --- CONFIG ----------------------------------
MODE = "tech_momentum_regime"
LABEL = "y_ret_5d"
MODEL = "ridge"  # 'ols', 'ridge', 'lasso'

# --- LOAD FEATURE CONFIG ---------------------
entry = next((cfg for cfg in feature_registry.values() if cfg["mode"] == MODE), None)
if entry is None:
    raise ValueError(f"MODE '{MODE}' not found in feature_registry.")

features_script = entry["script_path"].replace(".py", "")
feature_cols = entry.get("features", [])
label_cols = entry["label"]

# --- IMPORT FEATURES -------------------------
mod = import_module(features_script)
X, y = mod.get_features_and_labels()

if feature_cols:
    X = X[["date"] + feature_cols]
y = y[["date", LABEL]]

df = pd.merge(X, y, on="date").dropna()
print(f"\n Loaded dataset for mode: {MODE}")
print(f" Shape: {df.shape} | 🎯 Target: {LABEL}")

# --- DATA PREP -------------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- MODEL SELECTION -------------------------
if MODEL == "ols":
    model = LinearRegression()
elif MODEL == "ridge":
    model = Ridge()
elif MODEL == "lasso":
    model = Lasso()
else:
    raise ValueError("Unsupported regression model")

# --- WALK-FORWARD CV -------------------------
n_splits = 5
test_size = 0.2
results = []

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X_scaled, n_splits=n_splits, test_size=test_size)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    })

    # Save last fold predictions
    if fold == n_splits - 1:
        preds_df = pd.DataFrame({
            "date": d_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "residual": y_test.values - y_pred
        })

# --- OUTPUT ----------------------------------
res_df = pd.DataFrame(results)
print("\n Cross-Validation Results:")
print(res_df)

# --- SAVE ------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
mode_dir = os.path.join(RESULTS_DIR, safe_mode)
os.makedirs(mode_dir, exist_ok=True)

res_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_cv_results.csv"), index=False)
preds_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_predictions_last_fold.csv"), index=False)

report_path = os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_summary.txt")
with open(report_path, "w") as f:
    f.write("Cross-Validation Results:\n")
    f.write(res_df.to_string(index=False))
    f.write("\n\nLast Fold Prediction Sample:\n")
    f.write(preds_df.head(10).to_string(index=False))

print(f"\n Results saved to: {mode_dir}")

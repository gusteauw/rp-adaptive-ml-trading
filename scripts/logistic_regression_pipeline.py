# ============================================================
# Script: logistic_regression_pipeline.py
# Model Type: Logistic Regression (L1/L2 penalties)
# CV: Walk-forward Cross-Validation
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import random
random.seed(42)
import numpy as np
np.random.seed(42)

import pandas as pd
from importlib import import_module
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from config.paths import RESULTS_DIR
from walkforward_cv import walk_forward_split
import optuna
from datetime import datetime

# --- CONFIGURATION --------------------------
MODE = "daily_macro_features"    # Any mode from feature_registry
LABEL = "direction_5d"           # Binary target
PENALTY = "l2"                   # "l1", "l2"

# --- LOAD FEATURE CONFIG --------------------
from feature_registry import feature_registry
entry = next((cfg for cfg in feature_registry.values() if cfg["mode"] == MODE), None)
if entry is None:
    raise ValueError(f"MODE '{MODE}' not found in feature_registry.")

features_script = entry["script_path"].replace(".py", "")
feature_cols = entry.get("features", [])
label_cols = entry["label"]

# --- IMPORT FEATURES ------------------------
mod = import_module(features_script)
X, y = mod.get_features_and_labels()

if feature_cols:
    X = X[["date"] + feature_cols]
y = y[["date", LABEL]]

df = pd.merge(X, y, on="date").dropna()
print(f"\n Loaded data for mode: {MODE}")
print(f" Data shape: {df.shape} | Target: {LABEL}")

# --- PREPARE DATA ---------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- HYPERPARAMETER OPTIMIZATION ------------
def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 10.0)
    solver = "liblinear" if PENALTY == "l1" else "lbfgs"
    model = LogisticRegression(penalty=PENALTY, C=C, solver=solver, max_iter=1000, random_state=42)
    return cross_val_score(model, X_scaled, y, cv=3, scoring="roc_auc").mean()

print("\n Running Optuna hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n Best Parameters: {study.best_params}")
C = study.best_params["C"]
solver = "liblinear" if PENALTY == "l1" else "lbfgs"
model = LogisticRegression(penalty=PENALTY, C=C, solver=solver, max_iter=1000, random_state=42)

# --- CROSS-VALIDATION -----------------------
results = []
n_splits = 5
test_size = 0.2

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X_scaled, n_splits, test_size)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    })

# --- REPORT & SAVE --------------------------
res_df = pd.DataFrame(results)
print("\n Cross-Validation Results:")
print(res_df)

print("\n Classification Report (Last Fold):")
print(classification_report(y_test, y_pred))

# --- SAVE ARTIFACTS --------------------------
safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
mode_dir = os.path.join(RESULTS_DIR, safe_mode)
os.makedirs(mode_dir, exist_ok=True)

res_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_logreg_{timestamp}_cv_results.csv"), index=False)
study.trials_dataframe().to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_logreg_{timestamp}_optuna_trials.csv"), index=False)

report_str = classification_report(y_test, y_pred)
with open(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_logreg_{timestamp}_classification_report.txt"), "w") as f:
    f.write(report_str)

preds_df = pd.DataFrame({
    "date": d_test.values,
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba
})
preds_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_logreg_{timestamp}_predictions_last_fold.csv"), index=False)

# ============================================================
# Script: mlp_classification_pipeline.py
# Model Type: Multi-Layer Perceptron (MLP) Classifier
# Task: Binary Classification
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from datetime import datetime
import optuna

from config.paths import RESULTS_DIR
from walkforward_cv import walk_forward_split
from feature_registry import feature_registry

# --- CONFIGURATION --------------------------
MODE = "tech_momentum_regime"
LABEL = "y_up_5d"
MODEL = "mlp"

# --- LOAD FEATURE CONFIG --------------------
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
print(f"\nLoaded data for mode: {MODE}")
print(f"Data shape: {df.shape} | Target: {LABEL}")

# --- PREPARE DATA ---------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

# --- OPTUNA HYPERPARAMETER TUNING -----------
def objective(trial):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple([trial.suggest_int("n_units_l%d" % i, 32, 256) for i in range(trial.suggest_int("n_layers", 1, 3))]),
            alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            max_iter=500,
            random_state=42
        ))
    ])
    return cross_val_score(pipeline, X, y, cv=3, scoring="roc_auc").mean()

print("\n Running Optuna hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n Best Parameters: {study.best_params}")

# --- FINAL MODEL SETUP ----------------------
hidden_layers = tuple([study.best_params[k] for k in sorted(study.best_params) if k.startswith("n_units")])
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        alpha=study.best_params["alpha"],
        learning_rate_init=study.best_params["lr"],
        max_iter=500,
        random_state=42
    ))
])

# --- WALK-FORWARD CV ------------------------
results = []
n_splits = 5
test_size = 0.2

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X, n_splits=n_splits, test_size=test_size)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": acc,
        "roc_auc": auc
    })

# --- SAVE RESULTS ---------------------------
res_df = pd.DataFrame(results)
print("\n Cross-Validation Results:")
print(res_df)

print("\n Classification Report (Last Fold):")
report_str = classification_report(y_test, y_pred, target_names=[str(l) for l in unique_labels(y_test, y_pred)])
print(report_str)

# --- FILE SAVING ----------------------------
safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
mode_dir = os.path.join(RESULTS_DIR, safe_mode)
os.makedirs(mode_dir, exist_ok=True)

res_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_cv_results.csv"), index=False)
with open(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_classification_report.txt"), "w") as f:
    f.write(report_str)

study.trials_dataframe().to_csv(
    os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_optuna_trials.csv"),
    index=False
)

# Save predictions from last fold
y_proba = y_proba if y_proba is not None else np.full(len(y_pred), np.nan)
preds_df = pd.DataFrame({
    "date": d_test.values,
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba
})
preds_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_predictions_last_fold.csv"), index=False)

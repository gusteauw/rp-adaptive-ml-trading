# ============================================================
# Script: tree_classification_pipeline.py
# Model Type: Tree-Based Classification
# Models: Random Forest, Gradient Boosting, etc.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from config.paths import RESULTS_DIR
from walkforward_cv import walk_forward_split
import optuna


# --- CONFIGURATION --------------------------
MODE = "tech_momentum_regime"   
LABEL = "y_up_5d"               # e.g. y_up_1d, valuation_regime, etc.
MODEL = "rf"                    # "rf", "gb"

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
    if MODEL == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42,
        )
    elif MODEL == "gb":
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=42,
        )
    else:
        raise ValueError("Unsupported MODEL.")

    score = cross_val_score(model, X_scaled, y, cv=3, scoring="roc_auc").mean()
    return score

print("\n Running Optuna hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Best params & model
print(f"\n Best Parameters: {study.best_params}")
best_params = study.best_params
if MODEL == "rf":
    model = RandomForestClassifier(**best_params, random_state=42)
elif MODEL == "gb":
    model = GradientBoostingClassifier(**best_params, random_state=42)

# --- CROSS-VALIDATION -----------------------
results = []
n_splits = 5
test_size = 0.2

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

# --- RESULTS REPORT -------------------------
res_df = pd.DataFrame(results)
print("\n Cross-Validation Results:")
print(res_df)

print("\n Classification Report (Last Fold):")
print(classification_report(y_test, y_pred))

safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
# Classification Report
from sklearn.utils.multiclass import unique_labels

report_str = classification_report(y_test, y_pred, target_names=[str(label) for label in unique_labels(y_test, y_pred)])
with open(os.path.join(RESULTS_DIR, f"{safe_mode}_{safe_label}_{MODEL}_classification_report.txt"), "w") as f:
    f.write(report_str)

# Feature Importances
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(os.path.join(RESULTS_DIR, f"{safe_mode}_{safe_label}_{MODEL}_feature_importances.csv"))


preds_df = pd.DataFrame({
    "date": d_test.values,
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_proba": y_proba if y_proba is not None else np.nan
})
preds_df.to_csv(os.path.join(RESULTS_DIR, f"{safe_mode}_{safe_label}_{MODEL}_predictions_last_fold.csv"), index=False)


# --- FEATURE IMPORTANCES  ---------
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n🌟 Top 10 Feature Importances:")
    print(importances.sort_values(ascending=False).head(10))

# --- SAVE -----------------------------------
mode_dir = os.path.join(RESULTS_DIR, safe_mode)
os.makedirs(mode_dir, exist_ok=True)


from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

cv_filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_cv_results.csv"
optuna_filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_optuna_trials.csv"
report_filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_classification_report.txt"
importances_filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_feature_importances.csv"
preds_filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_predictions_last_fold.csv"

res_df.to_csv(os.path.join(mode_dir, cv_filename), index=False)
study.trials_dataframe().to_csv(os.path.join(mode_dir, optuna_filename), index=False)

with open(os.path.join(mode_dir, report_filename), "w") as f:
    f.write(report_str)

if hasattr(model, "feature_importances_"):
    importances.to_csv(os.path.join(mode_dir, importances_filename))

preds_df.to_csv(os.path.join(mode_dir, preds_filename), index=False)

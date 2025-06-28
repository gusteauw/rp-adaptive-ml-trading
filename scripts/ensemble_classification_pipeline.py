# ============================================================
# 📁 Script: ensemble_classification_pipeline.py
# 🌍 Model Type: Ensemble Classifier
# 📊 Models: Voting (Soft), Blending (Stacked)
# 🔁 CV: Walk-forward Cross-Validation
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
from datetime import datetime
from importlib import import_module
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from config.paths import RESULTS_DIR
from walkforward_cv import walk_forward_split
from feature_registry import feature_registry

# --- CONFIGURATION --------------------------
MODE = "tech_momentum_regime"
LABEL = "y_up_5d"
ENSEMBLE_TYPE = "voting"  # "voting" or "stacking"

random.seed(42)
np.random.seed(42)

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
print(f"\n🚀 Loaded data for mode: {MODE}")
print(f"📊 Shape: {df.shape} | 🌟 Target: {LABEL}")

# --- PREPARE DATA ---------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- DEFINE BASE MODELS ---------------------
base_models = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
]

if ENSEMBLE_TYPE == "voting":
    model = VotingClassifier(estimators=base_models, voting="soft")
elif ENSEMBLE_TYPE == "stacking":
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )
else:
    raise ValueError("Unsupported ENSEMBLE_TYPE")

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
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": acc,
        "roc_auc": auc
    })

# --- REPORT ---------------------------------
res_df = pd.DataFrame(results)
print("\n📊 Cross-Validation Results:")
print(res_df)
print("\n📜 Classification Report (Last Fold):")
print(classification_report(y_test, y_pred))

# --- SAVE RESULTS ---------------------------
safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

mode_dir = os.path.join(RESULTS_DIR, safe_mode)
os.makedirs(mode_dir, exist_ok=True)

res_df.to_csv(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{ENSEMBLE_TYPE}_{timestamp}_cv_results.csv"), index=False)
with open(os.path.join(mode_dir, f"{safe_mode}_{safe_label}_{ENSEMBLE_TYPE}_{timestamp}_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred))

print(f"\n📅 Results saved to: {mode_dir}")

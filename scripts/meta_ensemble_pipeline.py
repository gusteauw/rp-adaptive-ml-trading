# ============================================================
# Script: meta_ensemble_pipeline.py
# Model Type: Meta-Ensemble (Stacked Generalization)
# Base Models: Linear, Tree, RL, etc.
# Meta-Model: Logistic Regression (default)
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.paths import RESULTS_DIR

# --- CONFIGURATION --------------------------
PRED_FILES = [
    "tech_momentum_regime_y_up_5d_rf_predictions_last_fold.csv",
    "price_volatility_regime_y_up_5d_logistic_predictions_last_fold.csv",
    "tech_momentum_regime_y_up_5d_ppo_rl_preds.csv"
]
META_MODEL_TYPE = "logistic"  # 'logistic', 'gb'

# --- LOAD PREDICTIONS -----------------------
dfs = []
for fname in PRED_FILES:
    fpath = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(fpath)[["date", "y_pred"]].rename(columns={"y_pred": fname.split("_")[0]})
    dfs.append(df)

meta_df = dfs[0]
for df in dfs[1:]:
    meta_df = pd.merge(meta_df, df, on="date")

# ground truth
y_file = os.path.join(RESULTS_DIR, PRED_FILES[0])
y_df = pd.read_csv(y_file)[["date", "y_true"]]
meta_df = pd.merge(meta_df, y_df, on="date").dropna()

# --- PREPARE DATA ---------------------------
X = meta_df.drop(columns=["date", "y_true"])
y = meta_df["y_true"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- META MODEL TRAINING --------------------
if META_MODEL_TYPE == "logistic":
    meta_model = LogisticRegression()
elif META_MODEL_TYPE == "gb":
    from sklearn.ensemble import GradientBoostingClassifier
    meta_model = GradientBoostingClassifier()
else:
    raise ValueError("Unsupported meta model type.")

meta_model.fit(X_scaled, y)
y_pred = meta_model.predict(X_scaled)
y_proba = meta_model.predict_proba(X_scaled)[:, 1] if hasattr(meta_model, "predict_proba") else None

# --- METRICS -------------------------------
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba) if y_proba is not None else np.nan

print(f"\n Meta-Ensemble Accuracy: {acc:.4f}, AUC: {auc:.4f}")
print("\n Classification Report:")
print(classification_report(y, y_pred))

# --- SAVE RESULTS --------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(RESULTS_DIR, exist_ok=True)

base_name = f"meta_ensemble_{META_MODEL_TYPE}_y_up_5d_{timestamp}"
meta_df["meta_pred"] = y_pred
meta_df["meta_proba"] = y_proba

meta_df.to_csv(os.path.join(RESULTS_DIR, f"{base_name}_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, f"{base_name}_report.txt"), "w") as f:
    f.write(classification_report(y, y_pred, target_names=[str(l) for l in unique_labels(y, y_pred)]))

print(f"\n Meta-ensemble results saved to: {base_name}_results.csv")

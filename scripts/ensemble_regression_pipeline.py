# ============================================================
# 📁 Script: ensemble_regression_pipeline.py
# 🧠 Model Type: Ensemble Regression
# 📊 Models: VotingRegressor, StackingRegressor
# 🔁 CV: Walk-forward Cross-Validation
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from importlib import import_module

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from walkforward_cv import walk_forward_split
from feature_registry import feature_registry
from config.paths import RESULTS_DIR

# --- CONFIGURATION --------------------------
MODE = "price_volatility_regime"
LABEL = "fwd_return_5d"
MODEL = "stack"  # Options: 'vote', 'stack'

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

merged = pd.merge(X, y, on="date").dropna()
dates = merged["date"]
X = merged.drop(columns=["date", LABEL])
y = merged[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ENSEMBLE MODEL SETUP -------------------
base_learners = [
    ("ridge", Ridge()),
    ("gbr", GradientBoostingRegressor()),
    ("rf", RandomForestRegressor()),
    ("svr", SVR())
]

if MODEL == "vote":
    model = VotingRegressor(estimators=base_learners)
elif MODEL == "stack":
    model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
else:
    raise ValueError("MODEL must be 'vote' or 'stack'")

# --- WALK-FORWARD CV ------------------------
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

# --- RESULTS & SAVE -------------------------
res_df = pd.DataFrame(results)
print("\n📊 Cross-Validation Results:")
print(res_df)

safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_ensemble_reg_results.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)
res_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

print(f"\n📁 Results saved to: {filename}")

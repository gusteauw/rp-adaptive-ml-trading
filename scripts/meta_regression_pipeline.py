# ============================================================
# Script: meta_regression_pipeline.py
# Model Type: Meta-Ensemble (Regression)
# Base Models: Linear, Tree, RL, etc.
# Meta-Model: Ridge (default)
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.paths import RESULTS_DIR

# --- CONFIGURATION --------------------------
PRED_FILES = [
    "price_volatility_regime_ret_5d_ridge_predictions_last_fold.csv",
    "price_volatility_regime_ret_5d_rf_predictions_last_fold.csv"
]
META_MODEL_TYPE = "ridge"  # 'ridge', 'lasso', 'gb', 'ols'

# --- LOAD BASE MODEL PREDICTIONS ------------
dfs = []
for fname in PRED_FILES:
    fpath = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(fpath)[["date", "y_pred"]].rename(columns={"y_pred": fname.split("_")[0]})
    dfs.append(df)

meta_df = dfs[0]
for df in dfs[1:]:
    meta_df = pd.merge(meta_df, df, on="date")

# Load y_true from first file (all models used same data)
y_df = pd.read_csv(os.path.join(RESULTS_DIR, PRED_FILES[0]))[["date", "y_true"]]
meta_df = pd.merge(meta_df, y_df, on="date").dropna()

# --- PREPARE FEATURES & TARGET --------------
X = meta_df.drop(columns=["date", "y_true"])
y = meta_df["y_true"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- META MODEL TRAINING --------------------
if META_MODEL_TYPE == "ridge":
    meta_model = Ridge()
elif META_MODEL_TYPE == "lasso":
    meta_model = Lasso()
elif META_MODEL_TYPE == "ols":
    meta_model = LinearRegression()
elif META_MODEL_TYPE == "gb":
    meta_model = GradientBoostingRegressor()
else:
    raise ValueError("Unsupported meta model type.")

meta_model.fit(X_scaled, y)
y_pred = meta_model.predict(X_scaled)

# --- METRICS -------------------------------
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n Meta-Ensemble Regression Results:")
print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# --- SAVE ----------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
base_name = f"meta_ensemble_{META_MODEL_TYPE}_regression_{timestamp}"

meta_df["meta_pred"] = y_pred

os.makedirs(RESULTS_DIR, exist_ok=True)
meta_df.to_csv(os.path.join(RESULTS_DIR, f"{base_name}_results.csv"), index=False)

metrics_path = os.path.join(RESULTS_DIR, f"{base_name}_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\n MAE: {mae:.4f}\n")

print(f"\n Saved ensemble predictions and metrics to: {base_name}_results.csv")

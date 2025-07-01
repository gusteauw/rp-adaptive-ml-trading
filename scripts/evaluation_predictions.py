# ============================================================
# Script: evaluate_predictions.py
# Description: Compute evaluation metrics for model outputs
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import empyrical as emp

from config.paths import RESULTS_DIR

# --- CONFIG -------------------------------------
PREDICTION_FILE = "price_volatility_regime_ret_5d_ridge_20250628_1530_predictions_last_fold.csv"
MODEL_TAG = "ridge"

# --- LOAD PREDICTIONS ---------------------------
fpath = os.path.join(RESULTS_DIR, PREDICTION_FILE)
df = pd.read_csv(fpath)

if not {"date", "y_true", "y_pred"}.issubset(df.columns):
    raise ValueError("CSV must contain 'date', 'y_true', and 'y_pred' columns.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").dropna()

# --- BASIC METRICS ------------------------------
r2 = r2_score(df["y_true"], df["y_pred"])
rmse = mean_squared_error(df["y_true"], df["y_pred"], squared=False)
mae = mean_absolute_error(df["y_true"], df["y_pred"])

# --- DIRECTIONAL HIT RATE -----------------------
df["direction_true"] = np.sign(df["y_true"])
df["direction_pred"] = np.sign(df["y_pred"])
hit_rate = (df["direction_true"] == df["direction_pred"]).mean()

# --- STRATEGY RETURNS & EMPYRICAL METRICS -------
df["strategy_return"] = df["y_pred"].shift(1) * df["y_true"]  # simulate lagged prediction
df = df.dropna()

daily_returns = df["strategy_return"]
cumulative_return = (1 + daily_returns).prod() - 1
sharpe_ratio = emp.sharpe_ratio(daily_returns)
sortino_ratio = emp.sortino_ratio(daily_returns)
max_drawdown = emp.max_drawdown(daily_returns)

# --- OUTPUT SUMMARY -----------------------------
print("\n Performance Metrics Summary")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Directional Hit Rate: {hit_rate:.2%}")
print(f"Cumulative Return: {cumulative_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# --- SAVE TO FILE -------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"eval_{MODEL_TAG}_{timestamp}.txt"
out_path = os.path.join(RESULTS_DIR, fname)

with open(out_path, "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"Model: {MODEL_TAG}\n")
    f.write(f"File: {PREDICTION_FILE}\n\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"Directional Hit Rate: {hit_rate:.4f}\n")
    f.write(f"Cumulative Return: {cumulative_return:.4f}\n")
    f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
    f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
    f.write(f"Max Drawdown: {max_drawdown:.4f}\n")

print(f"\nMetrics saved to {fname}")

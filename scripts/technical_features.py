# ===========================================
# Script: technical_features.py
# MODE: tech_momentum_regime
# LABEL: return_class_{horizon} or vol_spike
# FEATURES: momentum, volatility, price behavior
# Frequency: Daily
# MODEL_TYPE: classification or regression
# ===========================================

# --- SETUP CONFIG ---------------------------
MODE = "tech_momentum_regime"
LABEL = "y_ret_5d"  # or "y_ret_5d", "y_vol_5d"
HORIZONS = [1, 5, 20]  # Lookahead for returns
MODEL_TYPE = "classification"

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np

import os
import pandas as pd

from config.paths import RAW_DIR
import os


# --- LOAD ---------------------------
file_path = os.path.join(RAW_DIR, "AAPL", "AAPL_2014_2024_technical_cleaned.csv")
df = pd.read_csv(file_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- CLEAN COLUMN NAMES ---------------------
df.columns = (
    df.columns
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
    .str.strip("_")
)

# --- LABEL CONSTRUCTION ---------------------
df["y_ret_1d"] = df["close"].pct_change().shift(-1)
df["y_ret_5d"] = df["close"].pct_change(5).shift(-5)
df["y_ret_20d"] = df["close"].pct_change(20).shift(-20)

df["y_up_1d"] = (df["y_ret_1d"] > 0).astype(int)
df["y_up_5d"] = (df["y_ret_5d"] > 0).astype(int)
df["y_up_20d"] = (df["y_ret_20d"] > 0).astype(int)

df["y_vol_5d"] = df["close"].pct_change().rolling(5).std().shift(-5)

# --- FEATURE SELECTION ----------------------
# Use smart keyword-based filtering for TA indicators
keywords = [
    "rsi", "macd", "signal", "hist", "momentum", "obv", "roc", "osc", "trix",
    "williams", "cci", "std", "vol", "ulcer", "trend", "psar", "mass",
    "klinger", "money", "trix", "qstick", "mfi", "boll", "band", "vortex",
    "ultimate", "stochastic", "twiggs", "acc"
]

# Heuristic: select numeric columns that match keywords and are well-formed
candidate_features = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])
    and any(k in col for k in keywords)
    and df[col].isna().mean() < 0.3
]
# Fallback to only highly complete ones
final_features = [col for col in candidate_features if df[col].isna().mean() < 0.1]

# --- FINALIZE -------------------------------
X = df[["date"] + final_features].dropna().reset_index(drop=True)
y = df[["date", "y_up_1d", "y_up_5d", "y_up_20d", "y_ret_1d", "y_ret_5d", "y_vol_5d"]]

# Align y to X
y = y[y["date"].isin(X["date"])].reset_index(drop=True)

# --- OUTPUT CHECK ---------------------------
print(f"{MODE} features and labels prepared.")
print(f"Feature matrix shape: {X.shape}")
print(f"Label matrix shape:   {y.shape}")
print(f"Selected {len(final_features)} technical features.")

# === API for pipeline use ===
def get_features_and_labels():
    return X, y

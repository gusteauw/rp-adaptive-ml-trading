# ===========================================
# 📁 Script: daily_macro_features.py
# 🧠 MODE: macro_sentiment_regime
# 🎯 LABEL: macro_vol_switch or future_mkt_return
# 📈 FEATURES: Fama-French factors, VIX, VXN, VXO, volatility spreads
# 🧪 MODEL_TYPE: classification or regression
# ===========================================

# --- SETUP CONFIG ---------------------------
MODE = "macro_sentiment_regime"
LABEL = "macro_vol_switch"
HORIZON = 5  # days ahead to detect shift
MODEL_TYPE = "classification"

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np
import os
from config.paths import RAW_DIR

# --- LOAD ----------------------------------
file_path = os.path.join(RAW_DIR, "AAPL", "merged_daily_factors_final.csv")
df = pd.read_csv(file_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- FEATURE ENGINEERING -------------------

ff_factors = [
    "ff3_mkt_rf", "ff3_smb", "ff3_hml", "ff3_mom", "ff5_rmw", "ff5_cma"
]
volatility_indices = [
    "vix_close", "vix_high", "vix_low",
    "vxn", "vxnh", "vxnl", "vxnd",
    "vxo", "vxoh", "vxol", "vxdo", "vxdh", "vxdl"
]

# 1. Volatility spreads & skew
df["vix_range"] = df["vix_high"] - df["vix_low"]
df["vxn_vxo_spread"] = df["vxn"] - df["vxo"]
df["vxd_vxn_skew"] = df["vxdh"] - df["vxnh"]

# 2. Lagged momentum in macro
for col in ff_factors:
    df[f"{col}_1d"] = df[col].shift(1)
    df[f"{col}_5d_chg"] = df[col] - df[col].shift(5)

# 3. Volatility change features
df["vix_5d_chg"] = df["vix_close"] - df["vix_close"].shift(5)
df["vol_spread_chg"] = df["vxn_vxo_spread"] - df["vxn_vxo_spread"].shift(5)

# --- LABELING -------------------------------

# Example: macro volatility regime switch (VIX jump > 10% over 5 days)
df[LABEL] = ((df["vix_close"].pct_change(HORIZON) > 0.10).astype(int)).shift(-HORIZON)

# Optional regression label:
# df[LABEL] = df["ff3_mkt_rf"].shift(-HORIZON)  # forward return of market

# --- FINALIZATION -------------------------------


feature_cols = (
    ff_factors +
    [f"{col}_1d" for col in ff_factors] +
    [f"{col}_5d_chg" for col in ff_factors] +
    volatility_indices +
    ["vix_range", "vxn_vxo_spread", "vxd_vxn_skew", "vix_5d_chg", "vol_spread_chg"]
)

df_final = df.dropna(subset=feature_cols + [LABEL]).reset_index(drop=True)

X = df_final[["date"] + feature_cols].copy()
y = df_final[["date", LABEL]].copy()

# --- OUTPUT CHECK ---------------------------
print(f"✅ {MODE} features/labels ready.")
print(f"📐 Shape: {df_final.shape}")
print(f"🧾 Columns: {df_final.columns.tolist()}")

# === API for pipelines ===
def get_features_and_labels():
    return X, y
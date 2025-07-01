# ===========================================
# Script: merged_daily_macro_features.py
# MODE: macro_regime
# LABEL: macro_volatility_regime or fwd_macro_return
# FEATURES: FF factors, vol indices, derived risk metrics
# Frequency: Daily
# MODEL_TYPE: classification (regime) or regression
# ===========================================

# --- SETUP CONFIG ---------------------------
MODE = "macro_regime"
LABEL = "vol_regime"
HORIZONS = [5, 10, 20]  # Forward windows
MODEL_TYPE = "classification"

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np
from config.paths import RAW_DIR
import os

# --- LOAD ---------------------------
file_path = os.path.join(RAW_DIR, "macro", "merged_daily_factors_final.csv")
df = pd.read_csv(file_path, parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)

# --- CLEANING ---------------------------
df.rename(columns={"adj_close": "adj_close"}, inplace=True)

# Dropping rows with missing essential factor data
ff_cols = [col for col in df.columns if "ff" in col]
df = df.dropna(subset=ff_cols)

# --- FEATURE ENGINEERING ---------------------------
# 1. Rolling vol on key FF factors
for col in ["ff3_mkt_rf", "ff3_mom", "ff5_mkt_rf", "ff5_mom"]:
    df[f"{col}_vol_5d"] = df[col].rolling(5).std()
    df[f"{col}_vol_20d"] = df[col].rolling(20).std()

# 2. Spread features (disagreement across models)
df["ff3_vs_ff5_mkt_diff"] = df["ff3_mkt_rf"] - df["ff5_mkt_rf"]
df["ff3_vs_ff5_mom_diff"] = df["ff3_mom"] - df["ff5_mom"]

# 3. Volatility indices: daily changes, spreads
for base in ["vix", "vxo", "vxn", "vxd"]:
    base = base.lower()
    try:
        df[f"{base}_range"] = df[f"{base}_high"] - df[f"{base}_low"]
        df[f"{base}_change"] = df[f"{base}_close"].pct_change()
    except:
        pass  

# 4. Optional: volatility spreads across indices
if all(col in df.columns for col in ["vix_close", "vxn", "vxo"]):
    df["vol_spread_vix_vxo"] = df["vix_close"] - df["vxo"]
    df["vol_spread_vix_vxn"] = df["vix_close"] - df["vxn"]

# 5. Macro momentum (forward-looking)
df["mkt_mom_5d"] = df["ff3_mkt_rf"].rolling(5).sum().shift(-5)
df["mkt_mom_20d"] = df["ff3_mkt_rf"].rolling(20).sum().shift(-20)

# --- LABEL CONSTRUCTION ---------------------------
# Label: volatility regime shift based on VIX z-score thresholds
vix_z = (df["vix_close"] - df["vix_close"].rolling(20).mean()) / df["vix_close"].rolling(20).std()
df[LABEL] = 0
df.loc[vix_z >= 1.0, LABEL] = 1  # High vol regime
df.loc[vix_z <= -1.0, LABEL] = -1  # Low vol regime

# --- FINALIZATION ---------------------------
macro_feature_cols = [
    col for col in df.columns
    if any(k in col for k in [
        "ff3_", "ff5_", "vix_", "vxo", "vxn", "vxd",
        "vol_", "range", "mom", "spread"
    ])
    and df[col].dtype in [np.float64, np.float32]
    and df[col].isna().mean() < 0.3
]

X = df[["date"] + macro_feature_cols].dropna().reset_index(drop=True)
y = df[["date", LABEL]].dropna().reset_index(drop=True)

# --- OUTPUT CHECK ---------------------------
print(f"{MODE} features and labels ready.")
print(f"Feature matrix shape: {X.shape}")
print(f"Label matrix shape:   {y.shape}")
print(f"Feature count: {len(macro_feature_cols)}")


# --- API FUNCTION -----------------------------
def get_features_and_labels():
    return X, y
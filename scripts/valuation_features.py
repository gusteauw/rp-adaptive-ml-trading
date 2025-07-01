# ===========================================
# Script: valuation_features.py
# MODE: valuation_regime
# LABEL: valuation_regime (based on z-score threshold events)
# FEATURES: valuation levels, deltas, z-scores
# Frequency: Monthly
# MODEL_TYPE: classification (regime detection) or regression
# ===========================================

# --- SETUP CONFIG ---------------------------
MODE = "valuation_regime"
LABEL = "valuation_regime"
HORIZONS = [21]  # Forward-looking 1-month return
MODEL_TYPE = "classification"  # Regime change detection
NORMALIZE = False  # if needed

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np
import os
from config.paths import RAW_DIR

# --- LOAD & CLEAN ---------------------------
file_path = os.path.join(RAW_DIR, "AAPL", "AAPL_valuations.csv")
df = pd.read_csv(file_path, skiprows=1)

# Dropping metadata and summary rows
df = df[df["date"].str.lower() != "ttm"]
df = df.iloc[:, :-2]  # Drop last two mostly-NaN columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Stripping commas and converting to float
for col in df.columns:
    if col not in ["date", "ticker"]:
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False)
              .replace("", np.nan)
              .astype(float)
        )

df = df.sort_values("date").reset_index(drop=True)

# --- FEATURE ENGINEERING --------------------
valuation_cols = [
    "MarketCap", "EnterpriseValue", "PeRatio", "ForwardPeRatio", "PegRatio",
    "PsRatio", "PbRatio"
]

# 1. Log scale for size metrics
df["log_marketcap"] = np.log(df["MarketCap"].replace(0, np.nan))
df["log_enterprise"] = np.log(df["EnterpriseValue"].replace(0, np.nan))

# 2. Short-term change in ratios
for col in valuation_cols:
    df[f"{col}_chg"] = df[col].pct_change()

# 3. Z-score features: 12-month rolling normalization
for col in valuation_cols:
    df[f"{col}_zscore"] = (
        (df[col] - df[col].rolling(12).mean()) / df[col].rolling(12).std()
    )

# --- LABEL CONSTRUCTION ---------------------
# Regime changes defined as z-score thresholds for valuation dislocation
df[LABEL] = 0
for zcol in ["PeRatio_zscore", "PsRatio_zscore", "PbRatio_zscore"]:
    df[LABEL] |= ((df[zcol].shift(1) < 1) & (df[zcol] >= 1))  # upwards regime shift
    df[LABEL] |= ((df[zcol].shift(1) > -1) & (df[zcol] <= -1))  # downward shift
df[LABEL] = df[LABEL].astype(int)

# --- FINALIZATION ---------------------------
features = [
    "log_marketcap", "log_enterprise"
] + [f"{col}_chg" for col in valuation_cols] + [f"{col}_zscore" for col in valuation_cols]

df_final = df.dropna(subset=features + [LABEL]).reset_index(drop=True)


X = df_final[["date"] + features].copy()
y = df_final[["date", LABEL]].copy()
# --- OUTPUT ---------------------------
print(f"{MODE} features/labels ready.")
print(f"Shape: {df_final.shape}")
print(f"Features: {len(features)}")
print(f"Columns: {df_final.columns.tolist()}")

# === API for pipelines ===
def get_features_and_labels():
    return X, y
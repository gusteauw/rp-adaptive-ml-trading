# ===========================================
# 📁 Script: price_features.py
# 🧠 MODE: market_regime_price
# 🎯 LABEL: fwd_return_{HORIZON}d (normalized or raw)
# 📈 FEATURES: log_return_1d, realized_vol
# 🧪 MODEL_TYPE: regression or classification (flexible downstream)
# ===========================================

# --- SETUP CONFIG ---------------------------
MODE = "market_regime_price"
LABEL = "fwd_return"
HORIZONS = [5, 10, 21]  # 1W, 2W, 1M
MODEL_TYPE = "regression"
FEATURES = ["log_return_1d", "volatility"]
NORMALIZE_RETURNS = True

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np
from config.paths import RAW_DIR
import os

# --- LOAD & CLEAN ---------------------------
file_path = os.path.join(RAW_DIR, "AAPL", "AAPL.csv")
df = pd.read_csv(file_path, parse_dates=["Date"])
df.rename(columns={"Date": "date", "Adj Close": "adj_close"}, inplace=True)
df = df[["date", "adj_close"]].dropna()

# --- FEATURE ENGINEERING --------------------
df["log_return_1d"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
df["volatility"] = df["log_return_1d"].rolling(21).std()

# --- LABEL CONSTRUCTION ---------------------
for h in HORIZONS:
    label = f"{LABEL}_{h}d"
    raw_fwd_return = np.log(df["adj_close"].shift(-h) / df["adj_close"])
    if NORMALIZE_RETURNS:
        df[label] = raw_fwd_return / df["volatility"]
    else:
        df[label] = raw_fwd_return

# --- FINALIZE -------------------------------
labels = [f"{LABEL}_{h}d" for h in HORIZONS]
df_final = df.dropna(subset=labels).reset_index(drop=True)

# --- OUTPUT CHECK ---------------------------
print(f"✅ {MODE} features/labels ready.")
print(f"📐 Shape: {df_final.shape}")
print(f"🧾 Columns: {df_final.columns.tolist()}")

df.rename(columns={
    "fwd_return_5d": "ret_5d",
    "fwd_return_10d": "ret_10d",
    "fwd_return_21d": "ret_21d"
}, inplace=True)


# === API for pipeline use ===
def get_features_and_labels():
    X = df[["date"] + ["adj_close", "log_return_1d", "volatility"]].dropna().reset_index(drop=True)
    y = df[["date", "ret_5d", "ret_10d", "ret_21d"]].dropna().reset_index(drop=True)

    # Align by date
    y = y[y["date"].isin(X["date"])].reset_index(drop=True)
    X = X[X["date"].isin(y["date"])].reset_index(drop=True)

    return X, y

# ===========================================
# 📁 Script: options_features.py
# 🧠 MODE: options_sentiment_regime
# 🎯 LABEL: to be joined later with market data
# 📈 FEATURES: iv surface, flow asymmetries, call/put bias
# 🧪 MODEL_TYPE: classification / hybrid
# ===========================================


# NOTE: Labels are external (e.g., from price_features), joined in pipeline via 'date'

# --- SETUP CONFIG ---------------------------
MODE = "options_sentiment_regime"
HORIZON = 5
MODEL_TYPE = "classification"

# --- DEPENDENCIES ---------------------------
import pandas as pd
import numpy as np
from dateutil import parser
import os
from config.paths import RAW_DIR

# --- LOAD OPTIONS DATA ----------------------
file_path = os.path.join(RAW_DIR, "AAPL", "AAPL_options_v3.csv")
opt = pd.read_csv(file_path)

# --- CLEAN COLUMN HEADERS -------------------
opt.columns = (
    opt.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^\w]+", "_", regex=True)
    .str.replace("_+", "_", regex=True)
    .str.strip("_")
)

# --- PARSE AND CLEAN DATES ------------------
def parse_mixed_datetime(val):
    if pd.isna(val):
        return pd.NaT
    try:
        return pd.to_datetime(val, format='%m/%d/%Y %I:%M %p')
    except Exception:
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

opt["last_trade_date"] = opt["last_trade_date"].apply(parse_mixed_datetime)
opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce")

opt = opt.dropna(subset=["last_trade_date", "expiry_date"])
opt["date"] = opt["last_trade_date"].dt.floor("D")

# --- DERIVED FEATURES -----------------------

# Call/put tagging
opt["option_type"] = opt["contract_name"].str.extract(r"(\d+[CP])")[0].str[-1].map({"C": "call", "P": "put"})

# DTE & bucketing
opt["dte"] = (opt["expiry_date"] - opt["last_trade_date"]).dt.days
def tag_dte_bucket(dte):
    if 0 <= dte <= 10:
        return "short"
    elif 20 <= dte <= 40:
        return "medium"
    elif 50 <= dte <= 120:
        return "long"
    return None
opt["dte_bucket"] = opt["dte"].apply(tag_dte_bucket)
opt = opt[opt["dte_bucket"].notnull()]

# Convert numerics
opt["open_interest"] = pd.to_numeric(opt["open_interest"], errors="coerce")
opt["volume"] = pd.to_numeric(opt["volume"], errors="coerce")
opt["implied_volatility"] = (
    pd.to_numeric(opt["implied_volatility"].astype(str).str.replace("%", ""), errors="coerce") / 100
)

# --- PIVOT TO DAILY BUCKETS -----------------
agg = opt.groupby(["date", "dte_bucket", "option_type"]).agg({
    "open_interest": "sum",
    "volume": "sum",
    "implied_volatility": "mean"
}).unstack(["dte_bucket", "option_type"])

# Flatten column names
agg.columns = [f"{metric}_{bucket}_{otype}" for metric, bucket, otype in agg.columns]

# Reset index to make 'date' a column
options_features = agg.reset_index()

# --- OUTPUT CHECK ---------------------------
print(f"✅ {MODE} features ready.")
print(f"📐 Shape: {options_features.shape}")
print(f"🧾 Columns: {options_features.columns.tolist()}")


# --- API ------------------------------------
def get_features_and_labels():
    X = options_features.copy()
    y = X[["date"]].copy()
    y["dummy_label"] = np.nan  # placeholder, overwritten in pipeline
    return X, y

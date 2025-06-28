# build_feature_panel_clean.py
import pandas as pd
import numpy as np
from pathlib import Path

PRICE_PATH = "data/market_data/price_panel.csv"
FEATURE_PATH = "data/features/full_feature_panel_dynamic.csv"

print("🔍 Loading price panel...")
prices = pd.read_csv(PRICE_PATH, parse_dates=["date"])
prices = prices.sort_values(["ticker", "date"])

print("🛠️ Generating features...")
def generate_features(df):
    df = df.copy()

    # Rolling returns
    df["return_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["return_10d"] = df.groupby("ticker")["close"].pct_change(10)
    df["return_21d"] = df.groupby("ticker")["close"].pct_change(21)

    # Volatility
    df["volatility_5d"] = df.groupby("ticker")["close"].pct_change().rolling(5).std()
    df["volatility_10d"] = df.groupby("ticker")["close"].pct_change().rolling(10).std()
    df["volatility_21d"] = df.groupby("ticker")["close"].pct_change().rolling(21).std()

    # Momentum
    df["momentum_5d"] = df.groupby("ticker")["close"].apply(lambda x: x / x.shift(5) - 1)
    df["momentum_10d"] = df.groupby("ticker")["close"].apply(lambda x: x / x.shift(10) - 1)
    df["momentum_21d"] = df.groupby("ticker")["close"].apply(lambda x: x / x.shift(21) - 1)

    # Volume z-score
    df["volume_zscore_5d"] = df.groupby("ticker")["volume"].transform(lambda x: (x - x.rolling(5).mean()) / x.rolling(5).std())
    df["volume_zscore_10d"] = df.groupby("ticker")["volume"].transform(lambda x: (x - x.rolling(10).mean()) / x.rolling(10).std())

    # Behavioral triggers
    df["overreaction_5d"] = np.where(df["return_5d"] > 2 * df["return_5d"].rolling(252).std(), 1, 0)
    df["underreaction_5d"] = np.where(df["return_5d"] < -2 * df["return_5d"].rolling(252).std(), 1, 0)

    return df

features = generate_features(prices)

# Target Construction (5d future return based signal)
features = features.sort_values(["ticker", "date"])
future_return = features.groupby("ticker")["close"].shift(-5) / features["close"] - 1

def label_return(x):
    if x > 0.01:
        return "buy"
    elif x < -0.01:
        return "sell"
    else:
        return "hold"

features["target"] = future_return.apply(label_return)

# Save
features.to_csv(FEATURE_PATH, index=False)
print(f"✅ Feature panel saved to {FEATURE_PATH}")

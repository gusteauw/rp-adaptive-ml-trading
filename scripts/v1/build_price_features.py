import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/prices")
OUTPUT_PATH = Path("data/features/price_feature_panel.csv")
ROLLING_WINDOWS = [5, 10, 21]  # ~1wk, 2wk, 1mo


def compute_price_features(df, ticker):
    df = df.copy()
    df["ticker"] = ticker
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Volume": "volume"})
    df = df.sort_values("date")

    # Basic returns
    df["return_1d"] = df["adj_close"].pct_change()
    for w in ROLLING_WINDOWS:
        df[f"return_{w}d"] = df["adj_close"].pct_change(w)

    # Lagged returns
    for w in ROLLING_WINDOWS:
        df[f"return_lag_{w}d"] = df["return_1d"].shift(w)

    # Momentum
    for w in ROLLING_WINDOWS:
        df[f"momentum_{w}d"] = df["adj_close"] / df["adj_close"].shift(w) - 1

    # Rolling volatility
    for w in ROLLING_WINDOWS:
        df[f"volatility_{w}d"] = df["return_1d"].rolling(w).std()

    # Z-scores
    for w in ROLLING_WINDOWS:
        mean = df["return_1d"].rolling(w).mean()
        std = df["return_1d"].rolling(w).std()
        df[f"zscore_return_{w}d"] = (df["return_1d"] - mean) / std

    # Volume features
    for w in ROLLING_WINDOWS:
        df[f"volume_avg_{w}d"] = df["volume"].rolling(w).mean()
        df[f"volume_change_{w}d"] = df["volume"] / df["volume"].shift(w) - 1
        df[f"volume_zscore_{w}d"] = (df["volume"] - df["volume"].rolling(w).mean()) / df["volume"].rolling(w).std()

    # Drop rows with rolling NaNs
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    all_features = []
    for file in sorted(DATA_DIR.glob("*.csv")):
        ticker = file.stem.upper()
        print(f"\n🔍 Processing {ticker}...")
        df = pd.read_csv(file, parse_dates=["Date"])
        if df.shape[0] < max(ROLLING_WINDOWS) + 5:
            print(f"⚠️ Skipping {ticker}, not enough data.")
            continue
        df_feat = compute_price_features(df, ticker)
        all_features.append(df_feat)

    final = pd.concat(all_features).sort_values(["date", "ticker"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved feature panel to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

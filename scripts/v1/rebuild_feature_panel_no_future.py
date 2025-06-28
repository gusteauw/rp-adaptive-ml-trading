# rebuild_feature_panel_no_future.py

import pandas as pd
from pathlib import Path

# --- Paths ---
INPUT_FEATURE_PATH = "data/features/full_feature_panel_with_alpha_beta.csv"
OUTPUT_FEATURE_PATH = "data/features/full_feature_panel_no_future.csv"

# --- Load feature panel ---
print("\U0001F50D Loading original feature panel...")
df = pd.read_csv(INPUT_FEATURE_PATH, parse_dates=["date"])

# --- Drop future-looking features ---
future_leakage_features = [
    'return_5d_price', 'return_1d_price', 'return_10d', 'return_21d',
    'zscore_return_5d', 'zscore_return_10d', 'zscore_return_21d',
    'return_1d_sentiment', 'return_5d_sentiment', 'sentiment_return_1d'
]

print(f"\U0001F9F9 Dropping future-leaking columns: {future_leakage_features}")
columns_to_drop = [col for col in df.columns if col in future_leakage_features]

df_clean = df.drop(columns=columns_to_drop)

print(f"\U0001F4BE Saving clean feature panel to: {OUTPUT_FEATURE_PATH}")
Path(OUTPUT_FEATURE_PATH).parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_FEATURE_PATH, index=False)

print("\u2705 Feature panel cleaned successfully.")

# build_dataset_no_leakage.py
import pandas as pd
from pathlib import Path

INPUT_FILE = "data/features/full_feature_panel_with_alpha_beta.csv"
OUTPUT_FILE = "data/features/full_feature_panel_no_leakage.csv"

# Suspect leakage-related terms
leak_terms = [
    "return_5d", "return_1d", "return_10d", "return_21d",
    "return_lag_5d", "return_lag_10d", "return_lag_21d",
    "zscore_return_5d", "zscore_return_10d", "zscore_return_21d",
    "return_1d_sentiment", "return_5d_sentiment", "sentiment_return_1d"
]

print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Drop leaky columns
print("🧹 Dropping suspect columns...")
leaky_cols = [col for col in df.columns if any(term in col for term in leak_terms)]
print(f"⚠️ Removing columns: {leaky_cols}\n")
df_clean = df.drop(columns=leaky_cols)

# Save cleaned version
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved no-leakage version to: {OUTPUT_FILE}")

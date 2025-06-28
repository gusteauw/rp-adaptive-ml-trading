import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_PATH = Path("data/features/full_feature_panel.csv")
OUTPUT_PATH = Path("data/features/full_feature_panel_ready.csv")

# --- Load data ---
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df.set_index(["date", "ticker"], inplace=True)

# --- Clean step 1: Drop rows with NaNs ---
df.dropna(inplace=True)

# --- Clean step 2: Check for outliers using Z-scores ---
scaler = StandardScaler()
df_z = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Drop extreme outliers (|z| > 10 is likely broken data)
outlier_mask = (df_z.abs() > 10).any(axis=1)
df_clean = df[~outlier_mask].copy()

# --- Clean step 3: Create classification target ---
# Using return_5d_price → label as quantiles: sell (low), hold (mid), buy (high)
df_clean = df_clean[df_clean["return_5d_price"].notnull()]
df_clean["target"] = pd.qcut(
    df_clean["return_5d_price"], q=3, labels=["sell", "hold", "buy"]
)

# --- Final save ---
df_clean.to_csv(OUTPUT_PATH)

print(f"✅ Feature panel cleaned and saved to: {OUTPUT_PATH}")
print(df_clean["target"].value_counts(normalize=True))

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/factors/F-F_Research_Data_5_Factors_2x3_daily.csv")
OUTPUT_PATH = Path("data/factors/fama_french_factors_daily.csv")

# Load raw FF data
print("🔍 Loading raw Fama-French factor data...")
df_raw = pd.read_csv(INPUT_PATH, skiprows=3)

# Clean column names and drop footer rows
df_raw.columns = [col.strip() for col in df_raw.columns]
df_raw = df_raw.rename(columns={"Unnamed: 0": "date"})
df_raw = df_raw[df_raw["date"].str.match(r"^\d{8}$")]  # Keep only valid date rows

# Parse date and convert to datetime
df_raw["date"] = pd.to_datetime(df_raw["date"], format="%Y%m%d")

# Convert factors to numeric and divide by 100
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
df_raw[factor_cols] = df_raw[factor_cols].apply(pd.to_numeric, errors="coerce") / 100

# Compute market return (Mkt = Mkt-RF + RF)
df_raw["Mkt"] = df_raw["Mkt-RF"] + df_raw["RF"]

# Set index for merging
ff_factors = df_raw.set_index("date")

# Save cleaned factor dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ff_factors.to_csv(OUTPUT_PATH)
print(f"✅ Fama-French daily factors saved to: {OUTPUT_PATH}")
print(ff_factors.head())

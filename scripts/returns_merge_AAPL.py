import os
import pandas as pd

# === CONFIGURATION ===
AAPL_PATH = "data/raw/AAPL/AAPL.csv" 
UNIFIED_DIR = "results/unified_predictions/"  # directory with model predictions
OUTPUT_DIR = "results/merged_predictions"  # where to save merged files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 1: Load AAPL and compute daily returns ===
df_price = pd.read_csv(AAPL_PATH, parse_dates=["Date"])
df_price = df_price.sort_values("Date").reset_index(drop=True)

# Defensive check
if "Adj Close" not in df_price.columns:
    raise ValueError("The 'Adj Close' column is required in AAPL.csv")

# Compute daily return
df_price["daily_return"] = df_price["Adj Close"].pct_change()
df_return = df_price[["Date", "daily_return"]].dropna()
df_return = df_return.rename(columns={"Date": "date"})

# === STEP 2: Load each predictions CSV and merge with return ===
files = os.listdir(UNIFIED_DIR)
print(f"\nFiles in unified_predictions: {files}")

for file in files:
    if not file.lower().endswith((".csv", "_csv")):
        continue

    pred_path = os.path.join(UNIFIED_DIR, file)
    try:
        df_pred = pd.read_csv(pred_path, parse_dates=["date"])
    except Exception as e:
        print(f" Could not read {file}: {e}")
        continue


    # Merge with return
    df_merged = pd.merge(df_pred, df_return, on="date", how="inner")

    # Save merged file
    output_name = file.replace(".csv", "_merged.csv")
    df_merged.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
    print(f" Saved merged file: {output_name}")

print("\nAll model predictions have been merged with returns.")

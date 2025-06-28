# scripts/build_price_panel_from_raw.py

import pandas as pd
import glob
from pathlib import Path

RAW_DATA_PATH = "data/market_data/raw/"
PRICE_PANEL_PATH = "data/market_data/price_panel.csv"

print("🔍 Aggregating raw price data files...")

all_files = glob.glob(RAW_DATA_PATH + "*.csv")

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df.columns = [col.lower() for col in df.columns]  # lower-case all columns

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {file}")

    df["date"] = pd.to_datetime(df["date"])

    ticker = Path(file).stem.upper()
    df["ticker"] = ticker

    # Keep standard columns
    needed_cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
    df = df[[col for col in needed_cols if col in df.columns]]  # keep available

    df_list.append(df)

# Merge all
price_panel = pd.concat(df_list)
price_panel = price_panel.sort_values(["ticker", "date"])

print(f"✅ Saving unified price panel to {PRICE_PANEL_PATH}")
price_panel.to_csv(PRICE_PANEL_PATH, index=False)

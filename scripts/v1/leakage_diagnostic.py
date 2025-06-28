# leakage_diagnostic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FEATURE_FILE = "data/features/full_feature_panel_with_alpha_beta.csv"

print("🔍 Loading feature panel...")
df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
df = df.set_index(["date", "ticker"])

print("🔎 Checking for target leakage via direct features...")
leak_terms = ["return_5d", "return"]
suspect_cols = [col for col in df.columns if any(term in col for term in leak_terms)]

print(f"⚠️ Suspect columns containing leakage keywords: {suspect_cols}\n")

if "return_5d" in df.columns:
    print("⚠️ 'return_5d" \
    "' is in features! This is likely causing direct leakage.\n")
else:
    print("✅ 'return_5d' correctly excluded from feature set.\n")

print("📈 Calculating correlation of all features with target...")
if "target" in df.columns:
    df["target_encoded"] = df["target"].map({"buy": 0, "hold": 1, "sell": 2})
    corr = df.drop(columns=["target", "target_encoded"]).corrwith(df["target_encoded"])
    high_corr = corr[abs(corr) > 0.8].sort_values(ascending=False)

    if not high_corr.empty:
        print("⚠️ Features with very high correlation to target (|corr| > 0.8):")
        print(high_corr)
    else:
        print("✅ No suspiciously high correlations found between features and target.")
else:
    print("⚠️ 'target' column not found in feature set. Cannot compute correlation.")

print("✅ Leakage diagnostic completed.")

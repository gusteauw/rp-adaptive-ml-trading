import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

FEATURE_PATH = "data/features/full_feature_panel_with_alpha_beta.csv"


# --- Load and prep data ---
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.dropna(subset=["return_5d_price"])
df["target"] = pd.qcut(df["return_5d_price"], q=3, labels=["sell", "hold", "buy"])

# 🔹 Label encode (ensure it's available)
if "target_encoded" not in df.columns:
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["target"])

print("\n🔹 Global Label Distribution:")
print(df["target"].value_counts(normalize=True))

# --- Fold analysis ---
def fold_label_check(df_tkr, ticker):
    print(f"\nTicker: {ticker}")
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (_, test_idx) in enumerate(tscv.split(df_tkr)):
        y_test = df_tkr.iloc[test_idx]["target"]
        proportions = y_test.value_counts(normalize=True).to_dict()
        print(f"Fold {fold+1}: {proportions}")

def feature_correlation(df, ticker):
    df_tkr = df[df["ticker"] == ticker].copy()
    features = [c for c in df_tkr.columns if (c.startswith("sentiment_") or c.startswith("momentum_") or c.startswith("return_"))]
    df_corr = df_tkr[features].corrwith(df_tkr["target_encoded"].astype(int)).dropna()

    plt.figure(figsize=(10, 6))
    df_corr.sort_values().plot(kind="barh", title=f"Feature Correlation with Target (Encoded) — {ticker}")
    plt.tight_layout()
    plt.show()

print("\n🔹 Fold-Level Class Distribution Check")
for ticker in df["ticker"].unique():
    df_tkr = df[df["ticker"] == ticker].copy()
    if len(df_tkr) < 300:
        continue
    fold_label_check(df_tkr, ticker)

print("\n🤓 Feature Correlation Heatmap — AAPL")
feature_correlation(df, ticker="AAPL")

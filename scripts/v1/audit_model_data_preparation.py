# audit_model_data_preparation.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Paths
FEATURE_PATH = "data/features/full_feature_panel_no_future.csv"

print("🔍 Loading dataset...")
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.set_index(["date", "ticker"])

# Encode target for classification
df = df[df["target"].isin(["buy", "hold", "sell"])]
df["target_encoded"] = df["target"].map({"buy": 0, "hold": 1, "sell": 2})

# Confirm no future leakage in preprocessing
def audit_feature_window(df):
    """Check for suspicious feature relationships or lookaheads"""
    print("\n🔎 Checking feature windows and lags...")
    suspicious = [col for col in df.columns if "lead" in col or "future" in col]
    if suspicious:
        print("⚠️ Potential lookahead features:", suspicious)
    else:
        print("✅ No explicit future-looking features detected.")

audit_feature_window(df)

# Simulate baseline model on shuffled labels to evaluate overfitting
def run_null_model_simulation(df):
    print("\n🔎 Running null model (shuffled targets) as a sanity check...")
    df_shuffled = df.copy()
    df_shuffled["target_encoded"] = np.random.permutation(df_shuffled["target_encoded"].values)

    features = df_shuffled.drop(columns=["target", "target_encoded"]).columns.tolist()
    tickers = df_shuffled.index.get_level_values("ticker").unique()
    results = []

    for ticker in tickers:
        df_tkr = df_shuffled.xs(ticker, level="ticker").sort_index()
        X = df_tkr[features].fillna(0).values
        y = df_tkr["target_encoded"].values

        tscv = TimeSeriesSplit(n_splits=5)
        accs = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accs.append(accuracy_score(y_test, preds))

        avg_acc = np.mean(accs)
        print(f"Ticker: {ticker} - Avg Acc (null model): {avg_acc:.3f}")
        results.append(avg_acc)

    print(f"\n📉 Mean null-model accuracy across tickers: {np.mean(results):.3f}")

run_null_model_simulation(df)

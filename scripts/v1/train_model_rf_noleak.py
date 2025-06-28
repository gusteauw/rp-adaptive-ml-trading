import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter

# --- Config ---
FEATURE_PATH = "data/features/full_feature_panel_no_leakage.csv"
TICKERS = ["AAPL", "AMZN", "AVGO", "GOOGL", "MSFT", "NVDA", "TSLA", "WMT", "GOOG"]
N_SPLITS = 5
SEED = 42

print("🧠 Label mapping: {'buy': 0, 'hold': 1, 'sell': 2}")
LABEL_MAP = {"buy": 0, "hold": 1, "sell": 2}

# --- Load data ---
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.dropna()
df["target_encoded"] = df["target"].map(LABEL_MAP)

# --- Train per ticker ---
for ticker in TICKERS:
    print(f"\n📊 Training RF (no leakage) for ticker: {ticker}")
    df_tkr = df[df["ticker"] == ticker].sort_values("date")

    X = df_tkr.drop(columns=["date", "ticker", "target", "target_encoded"])
    y = df_tkr["target_encoded"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(f"\n✅ Fold {fold+1} - Train acc: {model.score(X_train, y_train):.2f}, Test acc: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, digits=2))

# train_model_rf_clean.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

FEATURE_PATH = "data/features/full_feature_panel_no_future.csv"

print("\U0001F9E0 Label mapping: {'buy': 0, 'hold': 1, 'sell': 2}")

# Load data
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.set_index(["date", "ticker"])
df = df[df["target"].isin(["buy", "hold", "sell"])]
df["target_encoded"] = df["target"].map({"buy": 0, "hold": 1, "sell": 2})

features = df.drop(columns=["target", "target_encoded"]).columns.tolist()
tickers = df.index.get_level_values("ticker").unique()

# Train RF
for ticker in tickers:
    print(f"\n\U0001F4CA Training Random Forest for ticker: {ticker}")

    df_tkr = df.xs(ticker, level="ticker").sort_index()
    X = df_tkr[features].fillna(0).values
    y = df_tkr["target_encoded"].values

    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        print(f"\n✅ Fold {fold} - Train acc: {accuracy_score(y_train, clf.predict(X_train)):.2f}, Test acc: {accuracy_score(y_test, preds):.2f}")
        print(classification_report(y_test, preds))
        fold += 1

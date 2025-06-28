# scripts/train_model_rf.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import json

FEATURE_PATH = "data/features/full_feature_panel_with_alpha_beta.csv"
REPORTS_PATH = "outputs/model_reports"

MODEL_NAME = "rf"
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    min_samples_split=10,
    random_state=42
)

def load_data():
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["target"])
    return df

def encode_labels(df):
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["target"])
    print("\U0001f9e0 Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    return df, le

def train_per_ticker(df):
    Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)
    tickers = df["ticker"].unique()
    for ticker in tickers:
        print(f"\n\U0001f4ca Training RF for ticker: {ticker}")
        df_ticker = df[df["ticker"] == ticker].copy()
        if len(df_ticker) < 100:
            print(f"⚠️ Skipping {ticker}, too few samples: {len(df_ticker)}")
            continue

        features = [c for c in df_ticker.columns if c not in ["date", "ticker", "target", "target_encoded"]]
        df_ticker = df_ticker.dropna(subset=features + ["target_encoded"])

        tscv = TimeSeriesSplit(n_splits=5)
        all_reports = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_ticker)):
            train, test = df_ticker.iloc[train_idx], df_ticker.iloc[test_idx]
            X_train, y_train = train[features], train["target_encoded"]
            X_test, y_test = test[features], test["target_encoded"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc_train = accuracy_score(y_train, model.predict(X_train_scaled))
            acc_test = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            print(f"\n✅ Fold {fold + 1} - Train acc: {acc_train:.2f}, Test acc: {acc_test:.2f}")
            print(classification_report(y_test, y_pred))

            all_reports.append({
                "fold": fold + 1,
                "train_accuracy": acc_train,
                "test_accuracy": acc_test,
                "report": report
            })

        with open(f"{REPORTS_PATH}/report_{ticker}_{MODEL_NAME}.json", "w") as f:
            json.dump(all_reports, f, indent=2)

if __name__ == "__main__":
    df = load_data()
    df, _ = encode_labels(df)
    train_per_ticker(df)

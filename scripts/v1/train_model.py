import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json

FEATURE_PATH = "data/features/sentiment_feature_panel.csv"
REPORTS_PATH = "outputs/model_reports"

MODELS = {
    "xgb": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    ),
    "rf": RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        max_features="sqrt",
        min_samples_split=10
    ),
    "lr": LogisticRegression(max_iter=500)
}


def load_data():
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["sentiment_score", "return_5d"])
    df["target"] = pd.qcut(df["return_5d"], q=3, labels=["sell", "hold", "buy"])
    return df

def encode_labels(df):
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["target"])
    print("\U0001f9e0 Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    return df, le

def train_per_ticker(df, models):
    Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)
    tickers = df["ticker"].unique()
    for ticker in tickers:
        print(f"\n\U0001f4ca Training models for ticker: {ticker}")
        df_ticker = df[df["ticker"] == ticker].copy()
        if len(df_ticker) < 100:
            print(f"⚠️ Skipping {ticker}, too few samples: {len(df_ticker)}")
            continue

        features = [c for c in df_ticker.columns if c.startswith("sentiment_") or c == "sentiment_score"]
        df_ticker = df_ticker.dropna(subset=features + ["target_encoded"])

        tscv = TimeSeriesSplit(n_splits=5)
        report_summary = {}

        for name, model in models.items():
            print(f"\n🔍 Model: {name}")
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
                report_summary.setdefault(name, []).append({
                    "fold": fold+1,
                    "train_accuracy": acc_train,
                    "test_accuracy": acc_test,
                    "report": report
                })

                print(f"\n✅ Train acc: {acc_train:.2f}, Test acc: {acc_test:.2f}")
                print(f"\n📅 Fold {fold+1} Report:\n", classification_report(y_test, y_pred))

        with open(f"{REPORTS_PATH}/report_{ticker}.json", "w") as f:
            json.dump(report_summary, f, indent=2)

if __name__ == "__main__":
    df = load_data()
    df, _ = encode_labels(df)
    train_per_ticker(df, MODELS)

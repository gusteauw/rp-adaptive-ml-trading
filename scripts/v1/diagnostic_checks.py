import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import itertools

FEATURE_PATH = "data/features/sentiment_feature_panel.csv"
REPORTS_PATH = "outputs/model_reports"

MODELS = {
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
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
    return df

def analyze_folds_for_leakage(df, ticker):
    df_ticker = df[df["ticker"] == ticker].copy()
    features = [c for c in df_ticker.columns if c.startswith("sentiment_") or c == "sentiment_score"]
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_ticker)):
        train = df_ticker.iloc[train_idx]
        test = df_ticker.iloc[test_idx]

        # Print label distribution
        print(f"\n📅 Fold {fold + 1} label distribution:")
        print(train["target"].value_counts(normalize=True))

        # Correlation checks
        corr = train[features + ["return_5d"]].corr()
        leak_corrs = corr["return_5d"].drop("return_5d").abs()
        print(f"\n🔍 Correlation with return_5d (Fold {fold+1}):")
        print(leak_corrs.sort_values(ascending=False))

if __name__ == "__main__":
    df = load_data()
    df = encode_labels(df)

    
    analyze_folds_for_leakage(df, ticker="")

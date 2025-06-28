# strategy_simulator.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt

# Load inputs
def load_data():
    features = pd.read_csv("data/sentiment/final_sentiment_panel.csv", parse_dates=["date"])
    prices = pd.read_csv("data/raw/all_stocks_ohlcv.csv", parse_dates=["Date"])
    labels = pd.read_csv("data/labels/master_labels.csv", parse_dates=["Date"])
    features = features.rename(columns={"date": "Date"})
    df = features.merge(prices, on=["Date"], how="left")
    df = df.merge(labels, on=["Date", "ticker"], how="left")
    return df.dropna()

# Simulate trading

def simulate_pnl(df):
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    model = joblib.load("models/xgb_model.pkl")
    feature_cols = [col for col in df.columns if col.endswith("_z")]

    df = df.sort_values("Date")
    test = df["Date"] >= "2023-01-01"
    df_test = df[test].copy()
    df_test["pred_label"] = model.predict(df_test[feature_cols])
    df_test["pred_action"] = le.inverse_transform(df_test["pred_label"])

    df_test["position"] = df_test["pred_action"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
    df_test["prev_position"] = df_test.groupby("ticker")["position"].shift(1).fillna(0)
    df_test["traded"] = df_test["position"] != df_test["prev_position"]
    df_test["transaction_cost"] = df_test["traded"].astype(int) * 0.001  # 10 bps

    df_test["daily_return"] = df_test["position"] * df_test["future_return"] - df_test["transaction_cost"]

    pnl_curve = df_test.groupby("Date")["daily_return"].sum().cumsum()
    pnl_curve.to_csv("results/strategy_cumulative_pnl.csv")

    plt.figure(figsize=(10, 5))
    pnl_curve.plot(title="Simulated Strategy Cumulative PnL (2023–2024)")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.tight_layout()
    Path("results").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/strategy_pnl_curve.png")
    print("Saved PnL curve and CSV to results/")

if __name__ == "__main__":
    df = load_data()
    simulate_pnl(df)

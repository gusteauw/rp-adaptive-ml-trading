# scripts/build_sentiment_features.py

import pandas as pd
import numpy as np
from pathlib import Path

# Config paths
SENTIMENT_PATH = "data/sentiment/hf_twitter_news_sentiment.csv"
PRICE_PATH = "data/market_data/price_panel.csv"
OUTPUT_PATH = "data/features/sentiment_feature_panel.csv"

def load_sentiment():
    df = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])
    df = df.dropna(subset=["sentiment_score"])
    return df

def load_prices():
    df = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])
    return df

def build_features():
    sentiment = load_sentiment()
    prices = load_prices()

    panel = prices.merge(sentiment, on=["date", "ticker"], how="left")
    panel = panel.sort_values(["ticker", "date"])

    # Strictly past-looking sentiment features
    panel["sentiment_score_lag_1"] = panel.groupby("ticker")["sentiment_score"].shift(1)
    panel["sentiment_score_lag_3"] = panel.groupby("ticker")["sentiment_score"].shift(3)
    panel["sentiment_score_mean_3d"] = (
        panel.groupby("ticker")["sentiment_score"].shift(1).rolling(window=3).mean().reset_index(0, drop=True)
    )
    panel["sentiment_score_mean_5d"] = (
        panel.groupby("ticker")["sentiment_score"].shift(1).rolling(window=5).mean().reset_index(0, drop=True)
    )
    panel["sentiment_score_std_5d"] = (
        panel.groupby("ticker")["sentiment_score"].shift(1).rolling(window=5).std().reset_index(0, drop=True)
    )
    panel["sentiment_score_z_5d"] = (
        (panel["sentiment_score"] - panel["sentiment_score_mean_5d"]) / (panel["sentiment_score_std_5d"] + 1e-6)
    )
    panel["sentiment_delta_1d"] = panel["sentiment_score"] - panel["sentiment_score_lag_1"]
    panel["sentiment_return_1d"] = panel.groupby("ticker")["sentiment_score"].pct_change()
    panel["sentiment_momentum_5d"] = (
        panel["sentiment_score"] - panel.groupby("ticker")["sentiment_score"].shift(5)
    )

    # 🚫 DROP potential leakage or redundant features
    panel.drop(columns=["sentiment_return_5d"], errors="ignore", inplace=True)

    # Final data hygiene
    panel = panel.dropna(subset=["sentiment_score_lag_1", "return_5d"])
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()

    Path("data/features").mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features saved to {OUTPUT_PATH} — final shape: {panel.shape}")

if __name__ == "__main__":
    build_features()

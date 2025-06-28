# wallstreetbets_ingest.py

import pandas as pd
import numpy as np
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Load FinBERT
finbert = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone"),
    tokenizer=AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
)

def score_sentiment(text):
    try:
        if isinstance(text, str) and len(text.strip()) > 5:
            result = finbert(text[:512])[0]
            label = result["label"]
            score = result["score"]
            return score if label == "positive" else -score if label == "negative" else 0.0
        return 0.0
    except:
        return 0.0

# Define keyword mapping to tickers
ticker_map = {
    "AAPL": ["apple"], "MSFT": ["microsoft"], "TSLA": ["tesla"],
    "AMZN": ["amazon"], "META": ["meta", "facebook"], "GOOG": ["google"],
    "GOOGL": ["google"], "NVDA": ["nvidia"], "BRK.B": ["berkshire"], "AVGO": ["broadcom"]
}

# Load WallStreetBets posts from CSV
def load_wsb_csv(file_path="data/sentiment/wsb_posts.csv"):
    try:
        df = pd.read_csv(file_path, parse_dates=["date"])
        return df
    except Exception as e:
        print("❌ Failed to load WallStreetBets data:", e)
        return pd.DataFrame()

def extract_sentiment(df):
    sentiment_log = defaultdict(list)

    for _, row in df.iterrows():
        text = str(row.get("title", "")) + " " + str(row.get("body", ""))
        timestamp = row.get("date")
        if pd.isna(timestamp):
            continue

        dt = pd.to_datetime(timestamp).strftime("%Y-%m-%d")

        for ticker, keywords in ticker_map.items():
            if any(re.search(rf"\\b{k}\\b", text.lower()) for k in keywords):
                score = score_sentiment(text)
                sentiment_log[(dt, ticker)].append(score)
                break

    daily_sentiment = [(dt, tkr, np.mean(scores)) for (dt, tkr), scores in sentiment_log.items() if scores]
    return pd.DataFrame(daily_sentiment, columns=["date", "ticker", "wsb_sentiment"])

if __name__ == "__main__":
    df = load_wsb_csv()
    if not df.empty:
        df_sent = extract_sentiment(df)
        Path("data/sentiment").mkdir(parents=True, exist_ok=True)
        df_sent.to_csv("data/sentiment/wallstreetbets_sentiment.csv", index=False)
        print("✅ Saved WallStreetBets sentiment panel.")
    else:
        print("⚠️ No WallStreetBets data processed.")

# sentiment_engine.py

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import feedparser

# Load FinBERT
finbert = pipeline("sentiment-analysis", 
                   model=AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone"),
                   tokenizer=AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone"))

def score_text(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return 0.0
    try:
        result = finbert(text[:512])[0]
        score = result['score'] if result['label'] == 'positive' else -result['score'] if result['label'] == 'negative' else 0.0
        return score
    except:
        return 0.0

def fetch_stocktwits(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return []
        messages = response.json().get("messages", [])
        return [m["body"] for m in messages if "body" in m]
    except:
        return []

def process_stocktwits(ticker):
    texts = fetch_stocktwits(ticker)
    scores = [score_text(t) for t in texts]
    return np.mean(scores) if scores else 0.0

def fetch_rss_headlines():
    feeds = [
        "https://finance.yahoo.com/rss/topstories",
        "https://www.investing.com/rss/news_25.rss"
    ]
    headlines = []
    for feed in feeds:
        parsed = feedparser.parse(feed)
        headlines.extend([entry.get("title", "") for entry in parsed.entries])
    return headlines

def process_rss_sentiment():
    headlines = fetch_rss_headlines()
    scores = [score_text(h) for h in headlines if h]
    return np.mean(scores) if scores else 0.0

def load_huggingface_sentiment():
    try:
        df = pd.read_csv("data/sentiment/hf_twitter_news_sentiment.csv", parse_dates=["date"])
        return df
    except:
        print("HF sentiment file missing.")
        return pd.DataFrame()

def load_wsb_sentiment():
    try:
        df = pd.read_csv("data/sentiment/wallstreetbets_sentiment.csv", parse_dates=["date"])
        return df
    except:
        print("WSB sentiment file missing.")
        return pd.DataFrame()

def build_sentiment_panel(tickers, dates):
    hf_sent = load_huggingface_sentiment()
    wsb_sent = load_wsb_sentiment()
    panel = []
    for date in dates:
        row = {"date": date}
        for ticker in tickers:
            row[f"{ticker}_stocktwits"] = process_stocktwits(ticker)
            hf_score = hf_sent[(hf_sent["date"] == pd.to_datetime(date)) & (hf_sent["ticker"] == ticker)]
            row[f"{ticker}_hf_sentiment"] = hf_score["sentiment_score"].values[0] if not hf_score.empty else 0.0
            wsb_score = wsb_sent[(wsb_sent["date"] == pd.to_datetime(date)) & (wsb_sent["ticker"] == ticker)]
            row[f"{ticker}_wsb_sentiment"] = wsb_score["wsb_sentiment"].values[0] if not wsb_score.empty else 0.0
        row["rss_sentiment"] = process_rss_sentiment()
        panel.append(row)
    return pd.DataFrame(panel)

def normalize_scores(df):
    sentiment_cols = [col for col in df.columns if col.endswith("_stocktwits") or col.endswith("_hf_sentiment") or col.endswith("_wsb_sentiment") or col == "rss_sentiment"]
    for col in sentiment_cols:
        df[col + "_z"] = (df[col] - df[col].rolling(20).mean()) / (df[col].rolling(20).std() + 1e-6)
    return df

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    dates = pd.date_range("2024-01-01", "2024-01-15")
    sentiment_df = build_sentiment_panel(tickers, dates)
    sentiment_df = normalize_scores(sentiment_df)
    sentiment_df.to_csv("data/sentiment/multi_source_sentiment_panel.csv", index=False)
    print("Sentiment panel saved.")

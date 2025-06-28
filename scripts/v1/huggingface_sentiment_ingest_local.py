# huggingface_sentiment_ingest_local.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT
finbert = pipeline("sentiment-analysis", 
                   model=AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone"),
                   tokenizer=AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone"))

def score_text(text):
    try:
        if isinstance(text, str) and len(text.strip()) > 5:
            result = finbert(text[:512])[0]
            label = result["label"]
            score = result["score"]
            return score if label == "positive" else -score if label == "negative" else 0.0
        return 0.0
    except:
        return 0.0

records = []

# Load jppgks
try:
    df = pd.read_parquet("data/sentiment/jppgks.parquet")
    df["date"] = pd.to_datetime(["2021-01-01"] * len(df)).date
    df["ticker"] = "ALL"
    df["sentiment_score"] = df["instruction"].apply(score_text)
    records.append(df[["date", "ticker", "sentiment_score"]])
    print("✅ jppgks scored with FinBERT")
except Exception as e:
    print("❌ jppgks failed:", e)

# Load zeroshot
try:
    df = pd.read_csv("data/sentiment/zeroshot.csv")
    df["date"] = pd.to_datetime(["2021-01-01"] * len(df)).date
    df["ticker"] = "ALL"
    df["sentiment_score"] = df["label"].map({"bullish": 1.0, "neutral": 0.0, "bearish": -1.0})
    records.append(df[["date", "ticker", "sentiment_score"]])
    print("✅ zeroshot processed")
except Exception as e:
    print("❌ zeroshot failed:", e)

# Load sovai
try:
    df = pd.read_parquet("data/sentiment/sovai.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={"sentiment": "sentiment_score"})
    records.append(df[["date", "ticker", "sentiment_score"]])
    print("✅ sovai loaded")
except Exception as e:
    print("❌ sovai failed:", e)

# Load egupta.jsonl
try:
    df = pd.read_json("data/sentiment/egupta.jsonl", lines=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["ticker"] = "ALL"
    df["sentiment_score"] = df["instruction"].apply(score_text)
    records.append(df[["date", "ticker", "sentiment_score"]])
    print("✅ egupta scored with FinBERT")
except Exception as e:
    print("❌ egupta failed:", e)

# Combine and save
try:
    all_sent = pd.concat(records, axis=0)
    sentiment_daily = all_sent.groupby(["date", "ticker"])['sentiment_score'].mean().reset_index()
    Path("data/sentiment").mkdir(parents=True, exist_ok=True)
    sentiment_daily.to_csv("data/sentiment/hf_twitter_news_sentiment.csv", index=False)
    print("✅ Saved final HF sentiment to data/sentiment/hf_twitter_news_sentiment.csv")
except Exception as e:
    print("⚠️ Final save failed:", e)

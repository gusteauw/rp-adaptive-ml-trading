# main_pipeline.py

from sentiment_engine import build_sentiment_panel, normalize_scores
from huggingface_sentiment_ingest import *
from labeling_engine import main as generate_labels
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA"]
DATES = pd.date_range("2024-01-01", "2024-01-15")

def ensure_dirs():
    for sub in ["data/raw", "data/sentiment", "data/labels", "models", "results"]:
        Path(sub).mkdir(parents=True, exist_ok=True)

def run_pipeline():
    ensure_dirs()

    print("Step 1: Running Hugging Face ingestion...")
    try:
        exec(open("scripts/huggingface_sentiment_ingest.py").read())
    except Exception as e:
        print("HF ingestion failed:", e)

    print("Step 2: Skipping Reddit ingestion (Pushshift deprecated).")

    print("Step 3: Building sentiment panel...")
    sentiment_df = build_sentiment_panel(TICKERS, DATES)
    sentiment_df = normalize_scores(sentiment_df)
    sentiment_df.to_csv("data/sentiment/final_sentiment_panel.csv", index=False)

    print("Step 4: Generating labels...")
    generate_labels()

    print("✅ Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()

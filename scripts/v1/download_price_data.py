# download_price_data.py

import yfinance as yf
import pandas as pd
from pathlib import Path

# Config
tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "META", "AVGO", "NVDA", "GOOG", "GOOGL", "BRK-B"]
start_date = "2014-01-01"
end_date = "2024-12-31"

def fetch_price_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[["Adj Close"]].rename(columns={"Adj Close": "close"})
    df["ticker"] = ticker
    df.index.name = "date"
    return df.reset_index()

def build_panel():
    all_data = []
    for ticker in tickers:
        print(f"📥 Downloading {ticker}...")
        df = fetch_price_data(ticker)
        all_data.append(df)
    full_df = pd.concat(all_data)
    full_df.sort_values(by=["ticker", "date"], inplace=True)
    full_df["return_1d"] = full_df.groupby("ticker")["close"].pct_change().shift(-1)
    full_df["return_5d"] = full_df.groupby("ticker")["close"].pct_change(periods=5).shift(-5)
    Path("data/market_data").mkdir(parents=True, exist_ok=True)
    full_df.to_csv("data/market_data/price_panel.csv", index=False)
    print("✅ Saved price panel to data/market_data/price_panel.csv")

if __name__ == "__main__":
    build_panel()

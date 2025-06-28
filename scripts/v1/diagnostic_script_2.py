import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

FEATURE_PATH = "data/features/sentiment_feature_panel.csv"

# Load data
print("\n🔍 Loading feature panel from:", FEATURE_PATH)
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.dropna(subset=["return_5d"])

# Ternary classification target
df["target"] = pd.qcut(df["return_5d"], q=3, labels=["sell", "hold", "buy"])

# Helper: Visualize target distribution by fold
def plot_fold_target_distribution(df_ticker, ticker):
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\n📊 Visualizing target distribution for: {ticker}")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_ticker)):
        y_test = df_ticker.iloc[test_idx]["target"]
        counts = y_test.value_counts(normalize=True)
        counts.plot(kind="bar", title=f"Fold {fold+1} Target Distribution ({ticker})")
        plt.xlabel("Class")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.show()

# Helper: Visualize multicollinearity (correlation heatmap)
def plot_feature_correlation(df_ticker, features, ticker):
    print(f"\n🧠 Correlation heatmap for {ticker}")
    corr = df_ticker[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap — {ticker}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for ticker in df["ticker"].unique():
        df_ticker = df[df["ticker"] == ticker].copy()
        if len(df_ticker) < 300:
            continue
        features = [c for c in df_ticker.columns if c.startswith("sentiment_") or c == "sentiment_score"]
        plot_fold_target_distribution(df_ticker, ticker)
        plot_feature_correlation(df_ticker, features, ticker)

# ============================================================
# Script: analyze_logreg_pred.py
# Purpose: Evaluate all merged logistic regression predictions
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

MERGED_DIR = "results/merged_predictions"

def compute_sharpe(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    std = excess_returns.std()
    return np.sqrt(252) * excess_returns.mean() / std if std > 1e-6 else np.nan

all_results = []
summary_rows = []

for fname in os.listdir(MERGED_DIR):
    if not fname.endswith("_merged.csv"):
        continue

    path = os.path.join(MERGED_DIR, fname)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if not {"daily_return", "y_pred", "y_proba"}.issubset(df.columns):
        print(f"⚠️ Skipping {fname} — missing required columns.")
        continue

    # Clean and ensure numeric
    df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    initial_len = len(df)
    df = df.dropna(subset=["daily_return", "y_pred", "y_proba"])

    if df.empty:
        print(f"⚠️ Skipping {fname} — all {initial_len} rows dropped after cleaning.")
        continue

    # Strategy returns
    df["strategy_return"] = df["daily_return"] * (df["y_pred"] == 1)
    df["linear_return"] = df["daily_return"] * df["y_proba"]

    # Cumulative returns
    df["market_cum"] = (1 + df["daily_return"]).cumprod()
    df["strategy_cum"] = (1 + df["strategy_return"]).cumprod()
    df["linear_cum"] = (1 + df["linear_return"]).cumprod()

    # Sharpe ratios
    sharpe_market = compute_sharpe(df["daily_return"])
    sharpe_strategy = compute_sharpe(df["strategy_return"])
    sharpe_linear = compute_sharpe(df["linear_return"])

    # Final values
    final_market = df["market_cum"].iloc[-1]
    final_strategy = df["strategy_cum"].iloc[-1]
    final_linear = df["linear_cum"].iloc[-1]

    # Label from filename
    label_base = fname.replace("_merged.csv", "")
    
    all_results.append({
        "date": df["date"],
        "market": df["market_cum"],
        f"{label_base} (logreg)": df["strategy_cum"],
        f"{label_base} (linear)": df["linear_cum"]
    })

    summary_rows.append({
        "Strategy": label_base,
        "Sharpe (Market)": round(sharpe_market, 2),
        "Sharpe (LogReg)": round(sharpe_strategy, 2),
        "Sharpe (Linear)": round(sharpe_linear, 2),
        "Final Market Return": round(final_market, 4),
        "Final LogReg Return": round(final_strategy, 4),
        "Final Linear Return": round(final_linear, 4),
        "Start Date": df["date"].min().date(),
        "End Date": df["date"].max().date(),
        "Days": len(df)
    })

# --- Plot cumulative returns ---
plot_df = pd.DataFrame()
for result in all_results:
    label_logreg = list(result.keys())[2]
    label_linear = list(result.keys())[3]

    df_plot = pd.DataFrame({
        "date": result["date"],
        "Market": result["market"],
        f"{label_logreg} (sharpe={summary_rows[-1]['Sharpe (LogReg)']:.2f})": result[label_logreg],
        f"{label_linear} (sharpe={summary_rows[-1]['Sharpe (Linear)']:.2f})": result[label_linear]
    })
    df_plot = df_plot.melt(id_vars="date", var_name="Strategy", value_name="Cumulative Return")
    plot_df = pd.concat([plot_df, df_plot], ignore_index=True)

plt.figure(figsize=(14, 6))
sns.lineplot(data=plot_df, x="date", y="Cumulative Return", hue="Strategy", linewidth=1.8)
plt.title("Cumulative Returns: Logistic Regression vs Linear Weighted vs Market")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(title="Strategy", loc="best")
plt.tight_layout()
plt.show()

# --- Display Summary Table ---
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("Sharpe (LogReg)", ascending=False)
print("\n Summary Performance Table:")
print(summary_df.to_string(index=False))

# audit_autocorr_temporal_featureimportance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

FEATURE_PATH = "data/features/full_feature_panel_no_future.csv"
WINDOW = 5

df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
df = df.set_index(["date", "ticker"])

print("\n🔁 Checking autocorrelation of target...")
acorr_by_ticker = []
for ticker, group in df.groupby("ticker"):
    group = group.sort_index()
    if "target" not in group.columns:
        continue
    le = LabelEncoder()
    y = le.fit_transform(group["target"])
    y_lag = pd.Series(y).shift(WINDOW).dropna()
    acorr = np.corrcoef(y[WINDOW:], y_lag)[0, 1]
    acorr_by_ticker.append((ticker, acorr))
    print(f"Ticker: {ticker}, {WINDOW}-lag autocorr: {acorr:.3f}")

print("\n🧠 Assessing potential temporal leakage in folds...")
ts_split = TimeSeriesSplit(n_splits=5)
le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df["target"])

for ticker, group in df.groupby("ticker"):
    X = group.drop(columns=["target", "target_encoded"])
    y = group["target_encoded"]
    for i, (train_idx, test_idx) in enumerate(ts_split.split(X)):
        test_dates = X.iloc[test_idx].index.get_level_values("date")
        train_max = X.iloc[train_idx].index.get_level_values("date").max()
        test_min = test_dates.min()
        if test_min <= train_max:
            print(f"⚠️ Temporal overlap in fold {i+1} for ticker {ticker}: train max {train_max}, test min {test_min}")

print("\n🌲 Running RandomForest for feature importance (sample ticker)...")
sample_ticker = df.index.get_level_values("ticker").unique()[0]
sample = df.loc[pd.IndexSlice[:, sample_ticker], :].dropna()
X = sample.drop(columns=["target", "target_encoded"])
y = sample["target_encoded"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n📊 Top 10 Features by Importance:")
print(importances.head(10))

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind="barh")
plt.title(f"Top 10 Feature Importances - {sample_ticker}")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

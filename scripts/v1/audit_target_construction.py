# audit_target_construction.py
import pandas as pd
from pathlib import Path

FEATURE_PATH = "data/features/full_feature_panel_no_future.csv"

print("🔍 Loading dataset...")
df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])

# Sort for safe lag checks
df = df.sort_values(by=["ticker", "date"])

# Step 1: Check if 'target' is derived from any future-looking columns
# We'll see how many days ahead the return that defines target spans
print("\n🔎 Checking target distribution and construction logic...")
print(df["target"].value_counts(normalize=True), "\n")

# Step 2: Inspect if the target correlates with *future* returns
# That would suggest it leaks test-set-like info
future_returns = [col for col in df.columns if "return_" in col and "lag" not in col]
print(f"🔎 Checking correlation of target with future-oriented returns: {future_returns}")

# Encode target numerically
df["target_encoded"] = df["target"].map({"buy": 0, "hold": 1, "sell": 2})

correlations = df[future_returns].corrwith(df["target_encoded"])
print("\n📈 Correlation of target with future returns:")
print(correlations.sort_values(ascending=False))

# Step 3: Verify if target label was constructed from one of these future-looking returns
# Check strong correlations (e.g., above 0.6)
suspect_corrs = correlations[abs(correlations) > 0.6]
if not suspect_corrs.empty:
    print("\n⚠️ High correlation found with future return(s):")
    print(suspect_corrs)
    print("\n🚨 This may indicate that target construction used future info — investigate the return definition!")
else:
    print("\n✅ No suspiciously high correlations between future returns and target. Target construction seems safe.")

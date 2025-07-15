import os
import sys
import pandas as pd
import numpy as np
from importlib import import_module

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.feature_registry import feature_registry


def check_label_presence(df, labels):
    return [label for label in labels if label not in df.columns]


def check_correlation_leakage(df, labels, threshold=0.75):  
    leakage_flags = []
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    for label in labels:
        if label not in numeric_df.columns:
            continue
        correlations = numeric_df.corrwith(numeric_df[label]).drop(label)
        strong_corrs = correlations[correlations.abs() > threshold]
        if not strong_corrs.empty:
            leakage_flags.append((label, strong_corrs.sort_values(ascending=False)))

    return leakage_flags


def check_leakage_for_config(name):
    config = feature_registry[name]
    script_path = config["script_path"].replace(".py", "")
    labels = config["label"]
    features = config.get("features", [])

    print(f"\n🔍 Checking for leakage in: {name} (script: {script_path})")

    try:
        mod = import_module(f"scripts.{script_path}")
        X, y = mod.get_features_and_labels()
    except Exception as e:
        print(f"    Error importing or running get_features_and_labels(): {e}")
        return

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        print("    Feature or label output is not a DataFrame.")
        return

    if features:
        # Guard against missing columns
        missing_feats = [f for f in features if f not in X.columns]
        if missing_feats:
            print(f"   Warning: some specified features missing in DataFrame: {missing_feats}")
        selected = ["date"] + [f for f in features if f in X.columns]
        X = X[selected]

    if "date" not in X.columns or "date" not in y.columns:
        print("    'date' column required in both X and y for merging.")
        return

    try:
        df = pd.merge(X, y, on="date").dropna()
    except Exception as e:
        print(f"   Error merging features and labels: {e}")
        return

    missing = check_label_presence(df, labels)
    if missing:
        print(f"   Missing labels: {missing}")
        print(f"   Available columns: {df.columns.tolist()[:10]}... (+{len(df.columns)-10} more)" if len(df.columns) > 10 else df.columns.tolist())
        return

    print("   All labels present.")
    print("   Running correlation check...")

    leak_corrs = check_correlation_leakage(df, labels)
    if leak_corrs:
        for label, corrs in leak_corrs:
            print(f"   Potential leakage: Features highly correlated with '{label}' (> 0.95):")
            print(corrs.to_string())
    else:
        print("   No strong feature-label correlations detected.")


if __name__ == "__main__":
    print("\n Starting leakage checks for all feature configurations...")
    for config_name in feature_registry:
        check_leakage_for_config(config_name)

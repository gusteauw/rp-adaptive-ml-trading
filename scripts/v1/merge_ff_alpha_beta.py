import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

# --- Paths ---
FEATURE_FILE = "data/features/full_feature_panel_ready.csv"
FF_FILE = "data/factors/fama_french_factors_daily.csv"
OUTPUT_FILE = "data/features/full_feature_panel_with_alpha_beta.csv"

# --- Load feature panel ---
df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
df = df.set_index(["date", "ticker"])

# --- Load Fama-French factors ---
ff = pd.read_csv(FF_FILE, parse_dates=["date"])
ff = ff.set_index("date")
ff = ff.dropna()

# --- Merge with features ---
df = df.reset_index().merge(ff, on="date", how="left").set_index(["date", "ticker"])

# --- Rolling regression per ticker ---
WINDOW = 252

def run_rolling_ff5(group):
    group = group.sort_index()
    if group.shape[0] < WINDOW:
        return pd.DataFrame(index=group.index)

    y = group["return_5d_price"]
    X = group[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)

    model = RollingOLS(endog=y, exog=X, window=WINDOW)
    rres = model.fit()
    
    params = rres.params
    out = pd.DataFrame(index=group.index)
    out["alpha"] = params["const"]
    out["beta_mkt"] = params["Mkt-RF"]
    out["beta_smb"] = params["SMB"]
    out["beta_hml"] = params["HML"]
    out["beta_rmw"] = params["RMW"]
    out["beta_cma"] = params["CMA"]
    return out

# Run per ticker
results = df.groupby("ticker", group_keys=False).apply(run_rolling_ff5)
df = df.join(results)

# Save
print(f"✅ Saving enhanced panel with alpha and FF5 betas to: {OUTPUT_FILE}")
df.reset_index().to_csv(OUTPUT_FILE, index=False)

# ============================================================
# Script: batch_run_linear_reg.py
# Runs linear_regression_pipeline.py for all combinations
# ============================================================

# import subprocess

# print("\n Starting batch runs for linear regression pipeline...\n")

# MODE = "options_sentiment_regime"
# LABELS = ["ret_5d", "vol_5d", "direction_5d"]
# MODELS = ["ols", "ridge", "lasso"]

# for label in LABELS:
#     for model in MODELS:
#         print(f"▶ Running: MODE={MODE}, LABEL={label}, MODEL={model}")
#         try:
#             subprocess.run(
#                 [
#                     "python",
#                     "scripts/linear_regression_pipeline.py",
#                     f"--mode={MODE}",
#                     f"--label={label}",
#                     f"--model={model}"
#                 ],
#                 check=True
#             )
#         except subprocess.CalledProcessError:
#             print(f" Failed run: {label}-{model}")

# print("\n All runs completed.\n")


# ============================================================
# Script: batch_run_linear_reg.py
# Updated: Only run valid regression-compatible feature modes
# ============================================================

import subprocess

print("\n🚀 Starting batch runs for linear regression pipeline...\n")

# --- VALID CONFIGS ONLY ---
CONFIGS = [
    {
        "mode": "price_volatility_regime",
        "labels": ["ret_5d", "ret_10d", "ret_21d"],
        "models": ["ols", "ridge", "lasso"]
    },
    {
        "mode": "valuation_regime",
        "labels": ["valuation_regime"],
        "models": ["ols", "ridge", "lasso"]
    }
]

for config in CONFIGS:
    MODE = config["mode"]
    for label in config["labels"]:
        for model in config["models"]:
            print(f"▶ Running: MODE={MODE}, LABEL={label}, MODEL={model}")
            try:
                subprocess.run(
                    [
                        "python",
                        "scripts/linear_regression_pipeline.py",
                        f"--mode={MODE}",
                        f"--label={label}",
                        f"--model={model}"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"❌ Failed run: {label}-{model}")

print("\n✅ All valid runs completed.\n")

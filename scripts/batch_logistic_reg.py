# ============================================================
# Script: batch_logistic_reg.py
# Batch runs for logistic_regression_pipeline.py
# ============================================================

import subprocess

CONFIGS = [
    {
        "mode": "tech_momentum_regime",
        "labels": ["y_up_1d", "y_up_5d", "y_up_20d"]
    },
    {
        "mode": "macro_sentiment_regime",
        "labels": ["macro_vol_switch"]
    },
    {
        "mode": "valuation_regime",
        "labels": ["valuation_regime"]
    }
]

PENALTIES = ["l1", "l2"]  # Regularization types

print("\n Starting batch runs for logistic regression pipeline...\n")

for cfg in CONFIGS:
    mode = cfg["mode"]
    for label in cfg["labels"]:
        for penalty in PENALTIES:
            print(f"▶ Running: MODE={mode}, LABEL={label}, PENALTY={penalty}")
            try:
                subprocess.run(
                    [
                        "python",
                        "scripts/logistic_regression_pipeline.py",
                        f"--mode={mode}",
                        f"--label={label}",
                        f"--penalty={penalty}"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f" Failed run: {mode}-{label}-{penalty}")

print("\n All logistic regression runs completed.\n")

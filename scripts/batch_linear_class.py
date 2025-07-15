# ============================================================
# Script: batch_run_linear_class.py
# Runs linear_classification_pipeline.py for all combinations
# ============================================================

import subprocess

print("\n🚀 Starting batch runs for linear classification pipeline...\n")

# --- VALID CONFIGS --------------------
CONFIGS = [
    {
        "mode": "valuation_regime",
        "labels": ["valuation_regime"],
        "models": ["logistic", "ridge"]
    },
    {
        "mode": "tech_momentum_regime",
        "labels": [ "y_up_1d"],
        "models": ["logistic", "ridge"]
    },
    # {
    #     "mode": "options_sentiment_regime",  # if we add valid classification labels
    #     "labels": ["direction_5d"],
    #     "models": ["logistic", "ridge"]
    # }
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
                        "scripts/linear_classification_pipeline.py",
                        f"--mode={MODE}",
                        f"--label={label}",
                        f"--model={model}"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"❌ Failed run: {label}-{model}")

print("\n✅ All valid runs completed.\n")

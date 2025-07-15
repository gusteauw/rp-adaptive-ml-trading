# batch_tree_reg.py

import subprocess

CONFIGS = [
    {
        "mode": "price_volatility_regime",
        "labels": ["ret_5d", "ret_10d", "ret_21d"]
    },
    {
        "mode": "valuation_regime",
        "labels": ["fwd_return_21d"]
    },

]

MODELS = ["rf", "gb"]  # Random Forest and Gradient Boosting

print("\n Starting Tree-Based Regression Batch Runs...\n")

for cfg in CONFIGS:
    mode = cfg["mode"]
    for label in cfg["labels"]:
        for model in MODELS:
            print(f"▶ Running MODE={mode}, LABEL={label}, MODEL={model}")
            try:
                subprocess.run(
                    [
                        "python",
                        "scripts/tree_regression_pipeline.py",
                        f"--mode={mode}",
                        f"--label={label}",
                        f"--model={model}"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f" Failed run: {mode}-{label}-{model}")

print("\n All Tree Regression Runs Completed.\n")

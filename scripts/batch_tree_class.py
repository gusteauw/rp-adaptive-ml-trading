# batch_tree_class.py

import subprocess

CONFIGS = [
    {
        "mode": "valuation_regime",
        "labels": ["valuation_regime"]
    },
    {
        "mode": "tech_momentum_regime",
        "labels": ["y_up_1d", "y_up_5d"]
    },

]

MODELS = ["rf", "gb"]  # Random Forest, Gradient Boosting

print("\n Starting Tree-Based Classification Batch Runs...\n")

for cfg in CONFIGS:
    mode = cfg["mode"]
    for label in cfg["labels"]:
        for model in MODELS:
            print(f"▶ Running MODE={mode}, LABEL={label}, MODEL={model}")
            try:
                subprocess.run(
                    [
                        "python",
                        "scripts/tree_classification_pipeline.py",
                        f"--mode={mode}",
                        f"--label={label}",
                        f"--model={model}"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f" Failed run: {mode}-{label}-{model}")

print("\n All Tree-Based Runs Completed.\n")

# batch_mlp_class.py
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

print("\n🚀 Starting MLP classification batch runs...\n")

for cfg in CONFIGS:
    mode = cfg["mode"]
    for label in cfg["labels"]:
        print(f"▶ Running MLP for MODE={mode}, LABEL={label}")
        try:
            subprocess.run(
                ["python", "scripts/mlp_classification_pipeline.py", f"--mode={mode}", f"--label={label}"],
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"❌ Failed run: {mode}-{label}")

print("\n✅ All MLP runs completed.\n")

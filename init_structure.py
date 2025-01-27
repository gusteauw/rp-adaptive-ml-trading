import os

def create_structure():
    # Directories to create
    dirs = [
        "data_processing",
        "supervised_learning",
        "reinforcement_learning",
        "backtesting",
        "utils"
    ]
    
    # Files to create (excluding README.md)
    files = [
        "requirements.txt",
        "config.py",
        "main.py",
        "data_processing/market_data.py",
        "data_processing/feature_engineering.py",
        "data_processing/eda.py",
        "supervised_learning/model_training.py",
        "supervised_learning/model_evaluation.py",
        "reinforcement_learning/trading_env.py",
        "reinforcement_learning/dqn_agent.py",
        "reinforcement_learning/train_rl.py",
        "backtesting/backtest.py",
        "backtesting/performance_metrics.py",
        "utils/logger.py",
        "utils/plotting.py",
        "utils/data_utils.py"
    ]

    # Create directories if they don't exist
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Create files
    for f in files:
        if not os.path.exists(f):
            with open(f, "w") as fp:
                # Just creating an empty file
                fp.write("")
    
    print("Project structure initialized (excluding README.md)!")

if __name__ == "__main__":
    create_structure()

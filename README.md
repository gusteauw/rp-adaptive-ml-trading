# rp-adaptive-ml-trading
Research Project Paper - 2025

This repository contains the Adaptive Machine Learning for Behavioral Trading Signals project, an academic research initiative aiming to combine behavioral finance insights with reinforcement learning (RL) techniques to develop adaptive trading strategies. The project is designed to be both reproducible and modular, separating data processing, feature engineering, supervised learning benchmarks, and RL environments into distinct modules.

Collaboration of **Nadiy** (nadiy.abdel-karim-dit-aramouni@hec.edu) & **Auguste** (auguste.piromalli@hec.edu) 
Towards the completion of MSc. Data Science & AI @ Ecole Polytechnique and MSc. in Finance @ HEC undertook by both authors  
Under the supervision and with guidance of **Professor Augustin Landier**

## keywords

Data Retrieval & Preprocessing
  Scripts to load market data (prices, volumes) and perform initial cleaning.
  Integration of news or social media sentiment data.

Feature Engineering
  Technical indicators (e.g., RSI, MACD, Bollinger Bands).
  Behavioral features (e.g., herding scores, sentiment analysis).

Supervised Learning Benchmarks
  Predictive models (e.g., Random Forest, XGBoost) for next-day market movement.
  Baseline comparisons for RL performance.

Reinforcement Learning Agents
  Custom Gym environment to simulate trading actions (Buy, Sell, Hold).
  DQN, PPO, A2C, and other RL algorithms to learn adaptive strategies.

Backtesting & Evaluation
  Sharpe ratio, maximum drawdown, and alpha calculations.
  Tools for reproducible historical simulations and robust performance metrics.


## project structure

adaptive_ml_trading/
│── README.md                     # Project documentation
│── requirements.txt              # Python dependencies
│── config.py                     # Configuration settings (paths, hyperparameters)
│
├── data_processing/
│   ├── market_data.py            # Data loading and preprocessing
│   ├── feature_engineering.py    # Technical, sentiment, and behavioral feature creation
│   ├── eda.py                    # Exploratory Data Analysis scripts
│
├── supervised_learning/
│   ├── model_training.py         # Training pipeline for supervised models
│   ├── model_evaluation.py       # Evaluation metrics and functions
│
├── reinforcement_learning/
│   ├── trading_env.py            # Custom OpenAI Gym environment
│   ├── dqn_agent.py              # DQN agent implementation
│   ├── train_rl.py               # RL training loop
│
├── backtesting/
│   ├── backtest.py               # Backtesting pipeline
│   ├── performance_metrics.py    # Financial metrics (Sharpe, drawdown, etc.)
│
├── utils/
│   ├── logger.py                 # Logging utilities
│   ├── plotting.py               # Visualization utilities
│   ├── data_utils.py             # Additional data handling helpers
│
└── main.py                       # Main entry point to run the entire project



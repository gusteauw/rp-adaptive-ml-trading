# 📈 RP Adaptive ML Trading

A modular, extensible machine learning framework for regime-aware modeling, backtesting, and signal generation in financial markets — built using walk-forward cross-validation and feature-script orchestration.

---

## 📁 Project Structure

rp-adaptive-ml-trading/
├── config/ # Path configurations and environment constants
├── scripts/ # Modular ML pipelines (classification, regression, RL, ensemble)
├── results/ # Model outputs, Optuna trials, performance metrics
├── data/
│ └── raw/ # Raw input data (prices, options, macro, valuations)
├── requirements.txt # Python dependencies


---

## 🧠 Pipelines & Models

Each pipeline is modular and supports walk-forward cross-validation and Optuna hyperparameter tuning.

| Script                           | Model Type                  | Purpose                                 |
|----------------------------------|-----------------------------|-----------------------------------------|
| `tree_classification_pipeline.py` | RF, Gradient Boosting       | Regime or signal classification         |
| `tree_regression_pipeline.py`     | RF, GB Regressor            | Return/volatility regression            |
| `linear_regression_pipeline.py`   | OLS, Ridge, Lasso           | Baseline linear modeling                |
| `logistic_regression_pipeline.py` | Logistic, RidgeClassifier   | Probabilistic classification            |
| `ensemble_classification_pipeline.py` | Voting/Stacking Classifier | Combine multiple classifiers            |
| `rl_classification_pipeline.py`   | PPO, A2C, DQN               | RL for directional policy decisions     |
| `rl_regression_pipeline.py`       | PPO with continuous rewards | RL for return optimization              |

---

## 🏗 Features

Feature engineering scripts are registered in `feature_registry.py`, each with:
- `mode` identifier (e.g. `"tech_momentum_regime"`)
- Associated labels (e.g. `"y_up_5d"`, `"ret_5d"`)
- Source data files (preprocessed CSVs)
- Python logic to extract and clean features

**Current modes:**
- `tech_momentum_regime` – technical indicators
- `valuation_regime` – valuation signals
- `macro_sentiment_regime` – macroeconomic daily indicators
- `options_sentiment_regime` – options IV & flow features
- `price_volatility_regime` – OHLCV-based return & volatility

---

## 🔁 Walk-Forward Cross-Validation

All pipelines are evaluated using **walk-forward cross-validation**, preserving temporal structure for realistic testing.

Benefits:
- Robust forward-looking evaluation
- Prevents lookahead bias
- Fold-by-fold metric logging (accuracy, R², AUC, etc.)

---

## 🔬 Hyperparameter Optimization

Each pipeline includes **Optuna integration** to optimize model hyperparameters.

- Trials are saved as CSVs in `/results`
- Easily configurable search spaces
- Model-agnostic objective functions

Example output:
results/tech_momentum_regime_y_up_5d_rf_20250628_2310_optuna_trials.csv

---

## 📦 Setup

```bash
# 1. Create environment
conda create -n rp-adaptive-ml python=3.10
conda activate rp-adaptive-ml

# 2. Install dependencies
pip install -r requirements.txt


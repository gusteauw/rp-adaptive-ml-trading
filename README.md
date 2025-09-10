# RP Adaptive ML Trading


A compact, extensible ML framework for cross-sectional equity forecasting with walk-forward evaluation, lag & horizon ensembles, and portfolio backtesting. Designed for realistic, paper-ready results (lead–lag, low-lag emphasis).

---
## Project Structure
The notebook lives in scripts/.  Market data is under data/tickers
```
rp-adaptive-ml-trading/
├─ config/                   # (optional) extra config files
├─ scripts/
│  └─ leadlag_wf.ipynb       #  main notebook 
├─ results/                  # (optional) figures / exports
├─ data/
│  └─ tickers_x.zip/               #  prices + features (tidy, by date,ticker)
├─ requirements.txt
```
---

## Pipelines
What this notebook does:
### Parameters & Methods:
- Ingest & QC: prices (close) + engineered features for a stable universe.
- Targets: forward returns at 5D and 21D (trading days).
- As-of alignment: pre-lag features to prevent look-ahead.
- Walk-forward CV: expanding train, monthly steps, embargo = max horizon.
- Lag ensemble: train per as-of lag (e.g., [1,3] or [1,3,5,10]) and weight by train-only IC with shrinkage → equal.
- Per-date neutralization vs prior 1D return: removes market whipsaw exposure and stabilizes ranks.
- Horizon ensemble: blend 5D & 21D via train-IC weights with shrinkage.
### Risk overlays:
- ADV eligibility (liquidity filter),
- Beta-neutralization overlay,
- Per-name cap and target gross leverage.
### Portfolio & costs:
- long-only or long–short, top-quantile, turnover, transaction costs in bps.
### Diagnostics:
- Rank-IC (daily), with sanity probes (XS shuffle, calendar shuffle).
### Backtest analytics:
- CAGR, Vol, Sharpe, Max DD, turnover, coverage, equity curves.
### Model variants:
- Ridge (prod), ElasticNet, Huber, HistGBR via a toggleable sweep.

---

## Key toggles & parameters
- USE_VARIANT_SWEEP: False (Ridge-only) or True (multi-model sweep).
- PROD_MODEL_KIND / PROD_MODEL_PARAMS: production model (default Ridge α=1.0).
- LAG_ENS: e.g., [1,3] for low-lag paper profile.
- USE_TRAIN_IC_WEIGHTS + IC_SHRINK_LAMBDA: train-only lag weights w/ shrink.
- USE_HORIZON_ENSEMBLE + HORIZON_LIST + HORIZON_IC_SHRINK_LAMBDA.
- Risk overlays: USE_ADV_ELIGIBILITY, ADV_PCT; USE_BETA_NEUTRAL, BETA_LOOKBACK_DAYS, BETA_NEUTRAL_STRENGTH; USE_PER_NAME_CAP, PER_NAME_CAP; GROSS_LEVERAGE.
- Portfolio knobs: REBALANCE_FREQ, TOP_QUANTILE, HOLDING_PERIOD, TCOST_BPS, LONG_SHORT.

---

## Some tested configs

P1 (conservative, low-lag, 5D only):
LAG_ENS=[1,3], USE_HORIZON_ENSEMBLE=False, TCOST_BPS=10–25, TOP_QUANTILE=0.2–0.3, LONG_SHORT=True or long-only; ADV filter on, beta overlay off.

P2 (balanced 5D+21D):
LAG_ENS=[1,3,5], USE_HORIZON_ENSEMBLE=True with moderate HORIZON_IC_SHRINK_LAMBDA≈0.4–0.6, ADV filter on, beta overlay optional.

P3 (robust, lower turnover):
REBALANCE_FREQ='M', HOLDING_PERIOD=10, TOP_QUANTILE=0.2, ADV filter on, consider beta-neutral overlay to dampen market drift exposure.

---

## Set-up

```bash
# 1. Create environment
conda create -n rp-adaptive-ml python=3.10
conda activate rp-adaptive-ml

# 2. Install dependencies
pip install -r requirements.txt
```
---

######################## Previous iteration of the project ########################

---


A modular, extensible machine learning framework for regime-aware modeling, backtesting, and signal generation in financial markets — built using walk-forward cross-validation and feature-script orchestration.

---

## Project Structure

rp-adaptive-ml-trading/
├── config/ # Path configurations and environment constants
├── scripts/ # Modular ML pipelines (classification, regression, RL, ensemble)
├── results/ # Model outputs, Optuna trials, performance metrics
├── data/
│ └── raw/ # Raw input data (prices, options, macro, valuations)
├── requirements.txt # Python dependencies


---

## Pipelines & Models

Each pipeline is modular and supports walk-forward cross-validation and Optuna hyperparameter tuning.

| Script                           | Model Type                  | Purpose                                 |
|----------------------------------|-----------------------------|-----------------------------------------|
| `tree_classification_pipeline.py` | RF, Gradient Boosting       | Regime or signal classification         |
| `tree_regression_pipeline.py`     | RF, GB Regressor            | Return/volatility regression            |
| `linear_regression_pipeline.py`   | OLS, Ridge, Lasso           | Baseline linear modeling                |
| `logistic_regression_pipeline.py` | Logistic, RidgeClassifier   | Probabilistic classification            |
| `ensemble_classification_pipeline.py` | Voting/Stacking Classifier | Combine multiple classifiers            |
| `rl_classification_pipeline.py`   | PPO, A2C               | RL for directional policy decisions     |
| `rl_regression_pipeline.py`       | PPO with continuous rewards | RL for return optimization              |

---

## Features

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

## Walk-Forward Cross-Validation

All pipelines are evaluated using **walk-forward cross-validation**, preserving temporal structure for realistic testing.

Benefits:
- Robust forward-looking evaluation
- Prevents lookahead bias
- Fold-by-fold metric logging (accuracy, R², AUC, etc.)

---

## Hyperparameter Optimization

Each pipeline includes **Optuna integration** to optimize model hyperparameters.

- Trials are saved as CSVs in `/results`
- Easily configurable search spaces
- Model-agnostic objective functions

Example output:
results/tech_momentum_regime_y_up_5d_rf_20250628_2310_optuna_trials.csv

---

## Setup

```bash
# 1. Create environment
conda create -n rp-adaptive-ml python=3.10
conda activate rp-adaptive-ml

# 2. Install dependencies
pip install -r requirements.txt


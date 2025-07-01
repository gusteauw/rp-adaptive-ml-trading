# =============================================
# Script: feature_registry.py
# Registry of feature/label script configs
# Used for automated model pipeline selection
# =============================================

feature_registry = {

    # === TECHNICAL FEATURES ===
    "technical_features": {
        "mode": "tech_momentum_regime",
        "label": ["y_up_1d", "y_up_5d", "y_up_20d", "y_ret_1d", "y_ret_5d", "y_vol_5d"],
        "frequency": "daily",
        "script_path": "technical_features.py",
        "feature_file": "AAPL_2014_2024_technical_cleaned.csv",
        "model_type": "classification"
    },

    # === VALUATION FEATURES ===
    "valuation_features": {
        "mode": "valuation_regime",
        "label": ["valuation_regime"],
        "frequency": "monthly",
        "script_path": "valuation_features.py",
        "feature_file": "AAPL_valuations.csv",
        "model_type": "classification"
    },

    # === PRICE FEATURES ===
    "price_features": {
        "mode": "price_volatility_regime",
        "label": ["fwd_return_5d", "fwd_return_10d", "fwd_return_21d"],
        "frequency": "daily",
        "script_path": "price_features.py",
        "feature_file": "AAPL.csv",
        "model_type": "regression"
    },

    # === DAILY MACRO FEATURES ===
    "daily_macro_features": {
        "mode": "macro_sentiment_regime",
        "label": ["macro_vol_switch"],
        "frequency": "daily",
        "script_path": "daily_macro_features.py",
        "feature_file": "merged_daily_factors_final.csv",
        "model_type": "classification"
    },

    # === MERGED MACRO FEATURES ===
    "merged_daily_macro_features": {
        "mode": "macro_regime",
        "label": ["vol_regime"],
        "frequency": "daily",
        "script_path": "merged_daily_macro_features.py",
        "feature_file": "merged_daily_factors_final.csv",
        "model_type": "classification"
    },

    # === OPTIONS FEATURES ===
    "options_features": {
        "mode": "options_sentiment_regime",
        "label": ["ret_5d", "vol_5d", "direction_5d"],  # may add more
        "frequency": "daily",
        "script_path": "options_features.py",
        "feature_file": "AAPL_options_v3.csv",
        "model_type": "classification"
    }
}


# === Utility ===
def get_feature_config(name):
    return feature_registry.get(name)


if __name__ == "__main__":
    for key, config in feature_registry.items():
        print(f" {key}: mode={config['mode']}, labels={config['label']}, file={config['feature_file']}")

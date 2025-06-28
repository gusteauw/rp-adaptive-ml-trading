# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
import talib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Paths
PRICE_FEATURES_PATH = Path("data/features/price_feature_panel.csv")
SENTIMENT_FEATURES_PATH = Path("data/features/sentiment_feature_panel.csv")
OUTPUT_PATH = Path("data/features/full_feature_panel.csv")


def load_price_features():
    print("\U0001F50D Loading price features...")
    df_price = pd.read_csv(PRICE_FEATURES_PATH, parse_dates=["date"])
    df_price = df_price.set_index(["date", "ticker"])
    return df_price


def load_sentiment_features():
    print("\U0001F4AC Loading sentiment features...")
    df_sentiment = pd.read_csv(SENTIMENT_FEATURES_PATH, parse_dates=["date"])
    df_sentiment = df_sentiment.set_index(["date", "ticker"])
    return df_sentiment


def merge_features(df_price, df_sentiment):
    df_price = df_price.reset_index()
    df_sentiment = df_sentiment.reset_index()

    print("🧪 Merging feature sets...")
    df_merged = pd.merge(
        df_price,
        df_sentiment,
        on=["date", "ticker"],
        how="inner",
        suffixes=("_price", "_sentiment")
    )
    df_merged = df_merged.set_index(["date", "ticker"])
    return df_merged


def save_full_feature_panel(df):
    print(f"\U0001F4BE Saving merged feature panel to: {OUTPUT_PATH}")
    df.reset_index().to_csv(OUTPUT_PATH, index=False)


class FeatureEngineer:
    """Feature engineering pipeline for market data"""
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        volatility_lookback: int = 20,
        momentum_lookback: int = 10,
        volume_lookback: int = 5
    ):
        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.volatility_lookback = volatility_lookback
        self.momentum_lookback = momentum_lookback
        self.volume_lookback = volume_lookback
        self.scaler = StandardScaler()
        
    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        
        # Trend indicators
        df['tech_sma'] = talib.SMA(close, timeperiod=20)
        df['tech_ema'] = talib.EMA(close, timeperiod=20)
        df['tech_macd'], df['tech_macd_signal'], _ = talib.MACD(close)
        
        # Momentum indicators
        df['tech_rsi'] = talib.RSI(close, timeperiod=14)
        df['tech_mom'] = talib.MOM(close, timeperiod=10)
        
        # Volatility indicators
        df['tech_atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['tech_bbands_upper'], df['tech_bbands_middle'], df['tech_bbands_lower'] = \
            talib.BBANDS(close, timeperiod=20)
        
        # Volume indicators
        df['tech_obv'] = talib.OBV(close, volume)
        df['tech_ad'] = talib.AD(high, low, close, volume)
        
        return df
    
    def _calculate_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral indicators"""
        returns = df['close'].pct_change()
        volume = df['volume']
        
        # Volatility clustering
        df['behav_vol_cluster'] = returns.rolling(self.volatility_lookback).std()
        df['behav_vol_regime'] = (
            df['behav_vol_cluster'] > df['behav_vol_cluster'].rolling(100).mean()
        ).astype(int)
        
        # Volume pressure
        df['behav_vol_pressure'] = volume / volume.rolling(self.volume_lookback).mean()
        
        # Price momentum and mean reversion
        df['behav_momentum'] = returns.rolling(self.momentum_lookback).mean()
        df['behav_mean_rev'] = (
            (df['close'] - df['close'].rolling(20).mean()) /
            df['close'].rolling(20).std()
        )
        
        # Sentiment features (if available)
        if self.sentiment_data is not None:
            df = df.merge(
                self.sentiment_data,
                left_index=True,
                right_index=True,
                how='left'
            )
            # Normalize sentiment scores
            sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
            for col in sentiment_cols:
                df[f'behav_{col}'] = self.scaler.fit_transform(
                    df[col].values.reshape(-1, 1)
                )
        
        # Herding behavior
        df['behav_herd_pressure'] = (
            (volume * abs(returns)) /
            (volume * abs(returns)).rolling(20).mean()
        )
        
        return df
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify market regimes using multiple metrics"""
        returns = df['close'].pct_change()
        
        # Trend regime
        df['regime_trend'] = np.where(
            df['tech_sma'] > df['tech_sma'].shift(1),
            1,  # Uptrend
            -1  # Downtrend
        )
        
        # Volatility regime
        vol = returns.rolling(20).std()
        df['regime_volatility'] = np.where(
            vol > vol.rolling(100).mean(),
            1,  # High volatility
            0   # Low volatility
        )
        
        # Volume regime
        vol_ma = df['volume'].rolling(20).mean()
        df['regime_volume'] = np.where(
            df['volume'] > vol_ma,
            1,  # High volume
            0   # Low volume
        )
        
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        df = self.price_data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Technical features
        df = self._calculate_technical_features(df)
        
        # Behavioral features
        df = self._calculate_behavioral_features(df)
        
        # Market regime features
        df = self._calculate_market_regime(df)
        
        # Clean up and forward fill any missing values
        df = df.fillna(method='ffill').fillna(0)
        
        return df


if __name__ == "__main__":
    df_price = load_price_features()
    df_sentiment = load_sentiment_features()
    df_full = merge_features(df_price, df_sentiment)
    save_full_feature_panel(df_full)

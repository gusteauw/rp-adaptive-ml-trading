import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class MarketState:
    price_features: np.ndarray  # OHLCV data
    technical_indicators: np.ndarray  # Technical analysis features
    behavioral_features: np.ndarray  # Sentiment, volatility clustering, etc.
    portfolio_state: np.ndarray  # Current position, cash, etc.

class MarketEnvironment(gym.Env):
    """Custom Market Environment that follows gym interface"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 20,
        transaction_cost: float = 0.001,
        initial_cash: float = 100000,
        action_type: str = "discrete"
    ):
        super(MarketEnvironment, self).__init__()
        
        self.data = data
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash
        self.action_type = action_type
        
        # Define action space
        if action_type == "discrete":
            self.action_space = gym.spaces.Discrete(3)  # Buy (1), Hold (0), Sell (-1)
        else:
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )  # Continuous allocation between -1 and 1
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'price_features': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(lookback_window, 5), dtype=np.float32
            ),
            'technical_indicators': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(lookback_window, 10), dtype=np.float32
            ),
            'behavioral_features': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(lookback_window, 5), dtype=np.float32
            ),
            'portfolio_state': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        })
        
        self.reset()
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on PnL and risk-adjusted metrics"""
        portfolio_return = self.current_position * self.data.iloc[self.current_step]['returns']
        transaction_cost = abs(action - self.current_position) * self.transaction_cost
        
        # Sharpe ratio component (using rolling window)
        returns_window = self.portfolio_returns[-self.lookback_window:]
        sharpe = np.mean(returns_window) / (np.std(returns_window) + 1e-6) * np.sqrt(252)
        
        # Combine immediate return with risk-adjusted metric
        reward = portfolio_return - transaction_cost + 0.1 * sharpe
        return reward
    
    def _get_observation(self) -> Dict:
        """Construct the observation state"""
        idx = self.current_step
        
        # Get historical price data
        price_features = self.data.iloc[idx-self.lookback_window:idx][
            ['open', 'high', 'low', 'close', 'volume']
        ].values
        
        # Get technical indicators
        technical_indicators = self.data.iloc[idx-self.lookback_window:idx][
            [col for col in self.data.columns if col.startswith('tech_')]
        ].values
        
        # Get behavioral features
        behavioral_features = self.data.iloc[idx-self.lookback_window:idx][
            [col for col in self.data.columns if col.startswith('behav_')]
        ].values
        
        # Portfolio state: [position, cash, equity]
        portfolio_state = np.array([
            self.current_position,
            self.cash,
            self.equity
        ])
        
        return {
            'price_features': price_features.astype(np.float32),
            'technical_indicators': technical_indicators.astype(np.float32),
            'behavioral_features': behavioral_features.astype(np.float32),
            'portfolio_state': portfolio_state.astype(np.float32)
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # Calculate reward before updating position
        reward = self._calculate_reward(action)
        
        # Update position and portfolio value
        old_position = self.current_position
        self.current_position = action if self.action_type == "discrete" else np.clip(action[0], -1, 1)
        
        # Calculate transaction costs
        transaction_cost = abs(self.current_position - old_position) * self.transaction_cost
        self.cash -= transaction_cost * self.equity
        
        # Update portfolio value
        self.equity *= (1 + reward)
        self.portfolio_returns.append(reward)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'portfolio_value': self.equity,
            'position': self.current_position,
            'transaction_cost': transaction_cost
        }
        
        return observation, reward, done, info
    
    def reset(self) -> Dict:
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.current_position = 0
        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.portfolio_returns = []
        
        return self._get_observation() 
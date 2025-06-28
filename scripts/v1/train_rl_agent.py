import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import yaml
from typing import Dict, Optional

from market_env import MarketEnvironment
from feature_engineering import FeatureEngineer

class RLTrainer:
    """Trainer class for RL trading agents"""
    
    def __init__(
        self,
        config_path: str = "config/rl_config.yaml",
        model_type: str = "ppo"
    ):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = model_type
        self.setup_paths()
        
    def setup_paths(self):
        """Create necessary directories"""
        for path in ['models', 'results', 'logs']:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self) -> pd.DataFrame:
        """Prepare and engineer features for training"""
        # Load price data
        price_data = pd.read_csv(
            self.config['data']['price_path'],
            parse_dates=['date']
        ).set_index('date')
        
        # Load sentiment data if available
        sentiment_data = None
        if self.config['data'].get('sentiment_path'):
            sentiment_data = pd.read_csv(
                self.config['data']['sentiment_path'],
                parse_dates=['date']
            ).set_index('date')
        
        # Engineer features
        engineer = FeatureEngineer(
            price_data=price_data,
            sentiment_data=sentiment_data,
            **self.config['feature_params']
        )
        
        return engineer.engineer_features()
    
    def create_env(
        self,
        data: pd.DataFrame,
        train: bool = True,
        seed: Optional[int] = None
    ) -> DummyVecEnv:
        """Create and wrap the trading environment"""
        def make_env():
            env = MarketEnvironment(
                data=data,
                **self.config['env_params']
            )
            if train:
                env = Monitor(env, f"logs/train_{seed}" if seed else "logs/train")
            return env
        
        return DummyVecEnv([make_env])
    
    def train(self):
        """Train the RL agent"""
        # Prepare data
        data = self.prepare_data()
        
        # Split data into train and validation
        train_cutoff = int(len(data) * 0.8)
        train_data = data.iloc[:train_cutoff]
        val_data = data.iloc[train_cutoff:]
        
        # Create environments
        train_env = self.create_env(train_data, train=True, seed=42)
        val_env = self.create_env(val_data, train=False)
        
        # Initialize model
        if self.model_type == "ppo":
            model = PPO(
                "MultiInputPolicy",
                train_env,
                verbose=1,
                **self.config['model_params']['ppo']
            )
        elif self.model_type == "dqn":
            model = DQN(
                "MultiInputPolicy",
                train_env,
                verbose=1,
                **self.config['model_params']['dqn']
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path="models/best_model",
            log_path="logs/eval",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        model.learn(
            total_timesteps=self.config['training']['total_timesteps'],
            callback=eval_callback
        )
        
        # Save the final model
        model.save(f"models/final_{self.model_type}_model")
        
    def evaluate(self, model_path: str):
        """Evaluate a trained model"""
        # Load test data
        data = self.prepare_data()
        test_data = data.iloc[int(len(data) * 0.8):]
        
        # Create test environment
        test_env = self.create_env(test_data, train=False)
        
        # Load model
        if self.model_type == "ppo":
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Run evaluation
        obs = test_env.reset()
        done = False
        rewards = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            rewards.append(reward)
        
        # Calculate metrics
        total_return = np.sum(rewards)
        sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-6) * np.sqrt(252)
        
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'rewards': rewards
        }

if __name__ == "__main__":
    trainer = RLTrainer(model_type="ppo")
    trainer.train()
    results = trainer.evaluate("models/best_model/best_model.zip") 
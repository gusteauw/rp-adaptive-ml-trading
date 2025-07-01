# ============================================================
# Script: rl_regression_pipeline.py
# Model Type: RL-Based Regression (Reward from returns)
# Environment: Gym-like Custom RL Env
# Goal: Minimize error by reinforcing accurate forecasts
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler
from importlib import import_module

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from feature_registry import feature_registry
from config.paths import RESULTS_DIR

# --- CONFIGURATION --------------------------
MODE = "price_volatility_regime"
LABEL = "fwd_return_5d"
MODEL = "ppo"  # 'ppo', 'a2c', 'dqn'
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

# --- LOAD FEATURES --------------------------
entry = next((cfg for cfg in feature_registry.values() if cfg["mode"] == MODE), None)
if entry is None:
    raise ValueError(f"MODE '{MODE}' not found in feature_registry.")

features_script = entry["script_path"].replace(".py", "")
feature_cols = entry.get("features", [])

mod = import_module(features_script)
X, y = mod.get_features_and_labels()

if feature_cols:
    X = X[["date"] + feature_cols]
y = y[["date", LABEL]]

# Merge and clean
df = pd.merge(X, y, on="date").dropna()
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
y_scaled = y.astype(np.float32).values

# --- DEFINE GYM ENV -------------------------
class RLRegressionEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.n = len(X)
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous forecast
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.X[self.current_step]

    def step(self, action):
        true = self.y[self.current_step]
        pred = action[0]
        reward = -abs(true - pred)  # Negative absolute error
        self.current_step += 1
        done = self.current_step >= self.n
        obs = self.X[self.current_step] if not done else self.X[-1]
        return obs, reward, done, {}

# --- ENV INSTANTIATION ----------------------
env = DummyVecEnv([lambda: RLRegressionEnv(X_scaled, y_scaled)])

model_dict = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
ModelClass = model_dict.get(MODEL)
if ModelClass is None:
    raise ValueError("Unsupported RL model.")

print(f"\n Training {MODEL.upper()} agent for regression...")
model = ModelClass("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)

# --- INFERENCE ------------------------------
state = env.reset()
y_preds = []
for _ in range(X_scaled.shape[0]):
    action, _ = model.predict(state, deterministic=True)
    y_preds.append(action[0][0])
    state, _, done, _ = env.step(action)
    if done:
        break

# --- EVALUATION -----------------------------
preds_df = pd.DataFrame({
    "date": dates.iloc[:len(y_preds)].values,
    "y_true": y.iloc[:len(y_preds)].values,
    "y_pred": y_preds
})

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(preds_df["y_true"], preds_df["y_pred"])
rmse = mean_squared_error(preds_df["y_true"], preds_df["y_pred"], squared=False)
r2 = r2_score(preds_df["y_true"], preds_df["y_pred"])

print(f"\n MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# --- SAVE RESULTS ---------------------------
safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
filename = f"{safe_mode}_{safe_label}_{MODEL}_{timestamp}_rl_regression.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)
preds_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
print(f"\n Saved predictions to: {filename}")

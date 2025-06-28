# ============================================================
# 📁 Script: rl_classification_pipeline.py
# 🧱 Model Type: Reinforcement Learning for Classification
# 📊 Goal: Optimize action policy based on feature signals
# 🔁 CV: Walk-forward CV adapted for RL setting
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces

from importlib import import_module
from config.paths import RESULTS_DIR
from feature_registry import feature_registry
from walkforward_cv import walk_forward_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

random.seed(42)
np.random.seed(42)

# --- CONFIGURATION --------------------------
MODE = "tech_momentum_regime"
LABEL = "y_up_5d"

# --- LOAD FEATURE CONFIG --------------------
entry = next((cfg for cfg in feature_registry.values() if cfg["mode"] == MODE), None)
if entry is None:
    raise ValueError(f"MODE '{MODE}' not found in feature_registry.")

features_script = entry["script_path"].replace(".py", "")
feature_cols = entry.get("features", [])
label_cols = entry["label"]

# --- IMPORT FEATURES ------------------------
mod = import_module(features_script)
X, y = mod.get_features_and_labels()

if feature_cols:
    X = X[["date"] + feature_cols]
y = y[["date", LABEL]]

df = pd.merge(X, y, on="date").dropna()
print(f"\n🚀 Loaded data for mode: {MODE}")
print(f"📀 Data shape: {df.shape} | 🌟 Target: {LABEL}")

# --- ENVIRONMENT CLASS ----------------------
class ClassificationEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y.reset_index(drop=True)
        self.n_samples = len(X)
        self.action_space = spaces.Discrete(2)  # binary classification
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        return self.X[self.current_idx], {}

    def step(self, action):
        reward = 1 if action == self.y[self.current_idx] else -1
        self.current_idx += 1
        done = self.current_idx >= self.n_samples
        obs = self.X[self.current_idx] if not done else np.zeros_like(self.X[0])
        return obs, reward, done, False, {}

# --- PREP DATA ------------------------------
dates = df["date"]
X = df.drop(columns=["date", LABEL])
y = df[LABEL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- WALK-FORWARD CV ------------------------
results = []
n_splits = 5
test_size = 0.2

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X_scaled, n_splits=n_splits, test_size=test_size)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    d_test = dates.iloc[test_idx]

    env = DummyVecEnv([lambda: ClassificationEnv(X_train, y_train)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=2000)

    preds = []
    for x in X_test:
        action, _ = model.predict(x, deterministic=True)
        preds.append(action)

    acc = accuracy_score(y_test, preds)
    results.append({
        "fold": fold,
        "start_date": d_test.min(),
        "end_date": d_test.max(),
        "accuracy": acc
    })

    print(f"\n📝 Fold {fold} Report:")
    print(classification_report(y_test, preds))

# --- SAVE -----------------------------------
res_df = pd.DataFrame(results)
print("\n📊 Cross-Validation Summary:")
print(res_df)

safe_mode = MODE.replace(".", "_")
safe_label = LABEL.replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(RESULTS_DIR, exist_ok=True)
res_df.to_csv(os.path.join(RESULTS_DIR, f"{safe_mode}_{safe_label}_rl_{timestamp}_results.csv"), index=False)

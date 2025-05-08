#!/usr/bin/env python3
"""
agent_template.py

Loads 1h data, builds the portfolio Gym env, inspects state/action,
and runs a quick training loop with PPO.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# adjust these imports to your project structure
from src.env.portfolio_env import PortfolioEnv

PRICES_PATH = os.path.join(os.getcwd(), 'data', 'raw', 'prices_1h.parquet')

def main():
    # --- 1) Load intraday price data ---
    # expects a MultiIndex parquet: (Ticker, {Open,High,Low,Close,Volume}) × Datetime
    df = pd.read_parquet(PRICES_PATH)  
    df_close = df.xs('Close', axis=1, level=1)
    df_close.columns = df_close.columns.get_level_values(0)
    
    # size T x N
    price_matrix = df_close.values

    # --- 2) Build the Gym environment ---
    # window=20 means the last 20 bars form the state history
    def make_env():
        return PortfolioEnv(price_matrix)
    env = DummyVecEnv([make_env])

    # --- 3) Inspect state & action spaces ---
    obs_space = env.observation_space
    act_space = env.action_space
    print(f"Observation shape: {obs_space.shape}")
    print(f"Action shape:      {act_space.shape}")
    print(f"  - obs low/high: {obs_space.low[:5]} … {obs_space.high[:5]}")
    print(f"  - act low/high:{act_space.low} … {act_space.high}")

    # sample a single step
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    print("Sample step → reward:", reward)

    # --- 4) Instantiate & train a PPO agent ---
    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99
    )
    # quick run – replace timesteps when you do full training
    model.learn(total_timesteps=10_000)
    model.save('outputs/ppo_quick_test')

    print("Training loop complete, model saved to outputs/ppo_quick_test.zip")

if __name__ == '__main__':
    main()

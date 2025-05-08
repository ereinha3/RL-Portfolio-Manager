from stable_baselines3 import PPO
import pandas as pd
from env.portfolio_env import PortfolioEnv
from utils.metrics import summary_stats
import matplotlib.pyplot as plt

# load env & model
env = PortfolioEnv()
model = PPO.load('outputs/ppo_model')

# simulate
obs = env.reset()
vals = []
for _ in range( env.T - env.window ):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    vals.append(env.cash + env.hold.dot(env.prices[env.t-1]))
    if done: break

series = pd.Series(vals)
stats = summary_stats(series)
print(stats)

# plot
series.div(series.iloc[0]).plot(); plt.show()
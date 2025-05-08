import joblib, pandas as pd, numpy as np
from config import INTRADAY_FILE, TICKERS_FILE, WINDOW
from features.engineering import make_features
from env.portfolio_env import PortfolioEnv

MODEL = 'outputs/ppo_model.zip'
model = joblib.load(MODEL)

# load latest
prices = pd.read_parquet(INTRADAY_FILE)
feats  = make_features(prices)
window = feats.iloc[-WINDOW:]
obs = np.concatenate([ window.values.flatten(), [1.0], np.zeros(len(window.columns.levels[0])) ])

action, _ = model.predict(obs, deterministic=True)
w = np.exp(action); w/=w.sum()
print("Target weights:", w)
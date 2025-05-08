import gym
import numpy as np
from gym import spaces
from src.config import WINDOW, COMMISSION, SPREAD_BPS, LAMBDA_RISK, INTRADAY_FILE, TICKERS_FILE
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self):
        # load data
        df = pd.read_parquet(INTRADAY_FILE)
        self.closes = df['Close']  # DataFrame (T × N)
        self.prices = self.closes.values
        self.T, self.N = self.prices.shape

        # init wallet
        self.window = WINDOW
        self.comm = COMMISSION
        self.spread = SPREAD_BPS / 10000
        self.lambda_risk = LAMBDA_RISK

        # spaces
        obs_dim = self.window * self.N + 1 + self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.N,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.t = self.window
        self.cash = 10000.0
        self.hold = np.zeros(self.N)
        self.last_w = np.zeros(self.N)
        return self._obs()

    def _obs(self):
        window_prices = self.prices[self.t-self.window:self.t]
        # log returns relative
        rel = np.log(window_prices/window_prices[0])
        obs = np.concatenate([
            rel.flatten(),
            [self.cash/(self.cash + self.hold.dot(self.prices[self.t-1]))],
            self.last_w
        ])
        return obs.astype(np.float32)

    def step(self, action):
        price = self.prices[self.t]
        V0 = self.cash + self.hold.dot(price)

        # target weights
        w = np.exp(action)
        w /= w.sum()

        target_value = w * V0
        current_value = price * self.hold
        trade = target_value - current_value

        # costs
        fee = self.comm * np.abs(trade).sum()
        spread = self.spread * np.abs(trade).sum()
        self.cash -= (fee + spread)

        # execute trades
        buy = trade.clip(min=0).sum()
        if buy > self.cash:
            scale = self.cash / buy
            trade = np.where(trade>0, trade*scale, trade)
        self.hold += trade/price
        self.cash -= trade.clip(min=0).sum()
        self.cash += -trade.clip(max=0).sum() * (1 - self.comm - self.spread)

        # reward
        V1 = self.cash + self.hold.dot(price)
        raw = V1 - V0
        reward = raw - self.lambda_risk * abs(raw)

        self.t += 1
        done = self.t >= self.T
        return (self._obs(), reward, done, {})
import gym
import numpy as np
from gym import spaces
import pandas as pd
from src.config import (
    WINDOW, COMMISSION, SPREAD_BPS,
    REWARD_SCALE, HOLD_BONUS, COST_SCALE,
    SLICE_LENGTH
)

class PortfolioEnv(gym.Env):
    def __init__(self, data_file):

        if 'TEST' in data_file:
            self.test_mode = True
        else:
            self.test_mode = False

        df = pd.read_parquet(data_file)

        closes = df.xs('Close', axis=1, level=1).ffill().bfill()
        self.prices = closes.values
        self.T, self.N = self.prices.shape

        # precompute returns
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = self.prices[1:] / self.prices[:-1] - 1

        if not self.test_mode:
            # training: fixed slice
            self.slice_len = min(SLICE_LENGTH, self.T - WINDOW)
            self.max_start = self.T - self.slice_len - 1
        else:
            # testing: one episode spanning everything after the WINDOW
            self.slice_len = self.T - WINDOW
            # we’ll start exactly at WINDOW
            self.max_start = WINDOW  

        # action & observation spaces
        obs_dim = WINDOW*self.N*2 + self.N + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, (self.N,), np.float32)

        # friction params
        self.comm   = COMMISSION
        self.spread = SPREAD_BPS/10_000

        self.reset()

    def reset(self):
        if not self.test_mode and self.max_start > WINDOW:
            # pick a random slice within [WINDOW, max_start]
            self.start = np.random.randint(WINDOW, self.max_start)
        else:
            # in test mode (or if your data is too short), always start at WINDOW
            self.start = WINDOW        
        
        self.t = self.start
        self.step_count = 0

        # wallet
        self.cash = 10_000.0
        self.hold = np.zeros(self.N, dtype=float)

        return self._obs()

    def _obs(self):
        # price & return windows
        p_win = self.prices[self.t-WINDOW:self.t]
        r_win = self.returns[self.t-WINDOW:self.t]
        norm_p = (p_win - p_win.mean(0))/(p_win.std(0)+1e-8)

        port_val   = self.cash + (self.hold*self.prices[self.t-1]).sum()
        cash_ratio = self.cash/port_val

        obs = np.concatenate([
            norm_p.flatten(),
            r_win.flatten(),
            self.hold / (self.hold.sum()+1e-8),
            [cash_ratio, port_val/10_000.0]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # 1) normalize target weights
        action = np.clip(action, 0,1)
        action = action/(action.sum()+1e-8)

        price = self.prices[self.t]
        V0    = self.cash + (self.hold*price).sum()

        # 2) desired vs current
        target_val  = action * V0
        current_val = self.hold * price
        trade        = target_val - current_val

        # 3) costs
        fee   = self.comm * abs(trade).sum()
        spread = self.spread * abs(trade).sum()
        cost_penalty = (fee+spread)/V0 * COST_SCALE

        # 4) adjust cash & holdings
        self.cash -= (fee+spread)
        self.cash = max(self.cash, 0.0)

        buy_amt = trade.clip(min=0).sum()
        if buy_amt>self.cash:
            scale= self.cash/buy_amt
            trade = np.where(trade>0, trade*scale, trade)

        self.hold += trade/price
        self.cash -= trade.clip(min=0).sum()
        self.cash += -trade.clip(max=0).sum()*(1-self.comm-self.spread)

        # 5) compute reward
        V1  = self.cash + (self.hold*price).sum()
        ret = (V1 - V0)/V0

        reward = (ret - cost_penalty)*REWARD_SCALE
        reward += HOLD_BONUS * (self.hold.sum()>0)

        # 6) advance time & check done
        self.t += 1
        self.step_count += 1
        done = (self.step_count >= self.slice_len)

        obs = self._obs()
        info = {'portfolio_value': V1, 'return': ret}
        return obs, reward, done, info

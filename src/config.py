import os

# Data paths
DATA_DIR      = os.path.join(os.getcwd(), 'data', 'raw')
INTRADAY_FILE = os.path.join(DATA_DIR, 'prices_1h.parquet')
TICKERS_FILE  = os.path.join(DATA_DIR, 'tickers_final.txt')

# RL hyperparameters
ALGO          = 'PPO'       # 'PPO', 'DQN', or 'SAC'
TIMESTEPS     = 500_000

# Environment settings
WINDOW        = 20          # look-back bars

# Cost & risk parameters
COMMISSION    = 0.0005       # 5 bps
SPREAD_BPS    = 2            # 2 bps half-spread
LAMBDA_RISK   = 0.5
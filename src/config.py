# config.py
import os

# ─── Data paths ─────────────────────────────────────────────-────────────────
ROOT_DIR            = os.path.dirname(os.path.dirname(__file__))
DATA_DIR            = os.path.join(ROOT_DIR, 'data')
TRAIN_FILE_HOURLY   = os.path.join(DATA_DIR, 'prices_train_1h.parquet')
EVAL_FILE_HOURLY    = os.path.join(DATA_DIR, 'prices_eval_1h.parquet')
TEST_FILE_HOURLY    = os.path.join(DATA_DIR, 'prices_test_1h.parquet')
TRAIN_FILE_DAILY    = os.path.join(DATA_DIR, 'prices_train_daily.parquet')
EVAL_FILE_DAILY     = os.path.join(DATA_DIR, 'prices_eval_daily.parquet')
TICKERS_FILE        = os.path.join(DATA_DIR, 'tickers_final.txt')
OUTPUT_DIR          = os.path.join(ROOT_DIR, 'outputs')
DAILY_CB_DIR        = os.path.join(OUTPUT_DIR, 'daily_cb')
DAILY_MODEL_PATH    = os.path.join(OUTPUT_DIR, 'daily_model')
HOURLY_CB_DIR       = os.path.join(OUTPUT_DIR, 'hourly_cb')
HOURLY_MODEL_PATH   = os.path.join(OUTPUT_DIR, 'hourly_model')
LOG_DIR             = os.path.join(OUTPUT_DIR, 'logs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DAILY_CB_DIR, exist_ok=True)
os.makedirs(HOURLY_CB_DIR, exist_ok=True)


# ─── Env settings ────────────────────────────────────────────────────────────
SLICE_LENGTH    = 500              # each episode lasts 500 env steps
WINDOW          = 20           # look-back bars
COMMISSION      = 0.0          # 0 with Alpaca and other brokers
SPREAD_BPS      = 0.5          # Highly liquid -> low spread
LAMBDA_RISK     = 0.1          # risk penalty multiplier
REWARD_SCALE    = 100.0
HOLDOUT_MONTHS  = 2
HOLDOUT_YEARS   = 2
EVAL_YEARS      = 1
INITIAL_CASH    = 10_000.0

# ─── SAC hyperparameters ────────────────────────────────────────────────────
GAMMA          = 0.95             # discount factor
ENT_COEF       = 0.01             # fixed entropy coefficient, encourages exploration
DAILY_LR       = 3e-4
HOURLY_LR      = 1e-4
BATCH_SIZE     = 256
BUFFER_SIZE    = 500_000
EVAL_FREQ      = 10_000
TIMESTEPS      = 500_000

# ─── Reward shaping ─────────────────────────────────────────────────────────
REWARD_SCALE   = 1_000.0          # multiply net return so gradients are large
HOLD_BONUS     = 0.01             # small per-step bonus for any non-zero position
COST_SCALE     = 1.0              # scale on cost penalty if you want to tweak
import pandas as pd
import yfinance as yf
from stable_baselines3 import SAC
from src.env.portfolio_env import PortfolioEnv
from src.config import (
    TEST_FILE_HOURLY,
    WINDOW,            
    INITIAL_CASH,
    HOURLY_MODEL_PATH,
    HOLDOUT_MONTHS,
    ROOT_DIR
)
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
# 1) Load your fine-tuned model
model = SAC.load(HOURLY_MODEL_PATH)

# 2) Load test set (with WINDOW bars of pad in front)
df_test = pd.read_parquet(TEST_FILE_HOURLY)
if df_test.index.tz is not None:
    df_test.index = df_test.index.tz_convert(None)
hourly_index = df_test.index.unique()

# 3) Initialize
pos = WINDOW   # skip the pad
n_rows = len(hourly_index)
results = []

# 4) Loop week-by-week
while True:
    if pos >= n_rows:
        break

    start_ts = hourly_index[pos]
    week_end_ts = start_ts + pd.Timedelta(days=7)

    # find the first index > week_end_ts
    end_pos = hourly_index.searchsorted(week_end_ts, side="right")
    # no more full weeks?
    if end_pos <= pos:
        break

    slice_len = end_pos - pos

    # 5) Run your agent on [pos … pos+slice_len)
    env = PortfolioEnv(data_file=TEST_FILE_HOURLY)
    # ─── Manually initialize the slice ─────────────────────────
    env.t           = pos     # jump to your chosen start
    env.slice_len   = slice_len     # force exactly this many steps
    env.step_count  = 0             # reset the step counter
    env.cash        = env.cash      # (keep initial cash=10k)
    env.hold        = env.hold      # (keep initial empty holdings)

    # first observation
    obs = env._obs()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)

    agent_roi = info['portfolio_value']/INITIAL_CASH - 1.0

    # 6) SPY benchmark over the calendar span [start_ts, week_end_ts]
    spy = yf.download(
        "SPY",
        start=start_ts.strftime("%Y-%m-%d"),
        end  =(week_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    bench_roi = (spy.iloc[-1]/spy.iloc[0] - 1.0).iloc[0]

    last_bar_idx = end_pos - 1
    results.append({
        "start":     start_ts.date(),
        "end":       hourly_index[last_bar_idx].date(),
        "agent_roi": agent_roi,
        "spy_roi":   bench_roi
    })

    # advance to next week
    pos = end_pos

# 7) Print line by line
for i, r in enumerate(results, 1):
    print(
      f"Week {i}: {r['start']} → {r['end']}, "
      f"Agent ROI: {100*r['agent_roi']:.2f}%, "
      f"SPY ROI:   {100*r['spy_roi']:.2f}%"
    )

# 8) Summary
avg_agent = sum(r['agent_roi'] for r in results)/len(results)
avg_spy   = sum(r['spy_roi']   for r in results)/len(results)
print(f"\nAverage weekly agent ROI: {100*avg_agent:.2f}%")
print(   f"Average weekly    SPY ROI: {100*avg_spy:.2f}%")

# 7) Turn into DataFrame for metrics & plotting
df = pd.DataFrame(results)

# Sharpe ratio (mean / std) on weekly returns
sr_agent = df['agent_roi'].mean() / (df['agent_roi'].std(ddof=0) + 1e-8)
sr_spy   = df['spy_roi'].mean()   / (df['spy_roi'].std(ddof=0)   + 1e-8)

# Max drawdown calculation
cum_agent    = (1 + df['agent_roi']).cumprod()
peak_agent   = cum_agent.cummax()
dd_agent     = (cum_agent - peak_agent) / peak_agent
maxdd_agent  = dd_agent.min()

cum_spy      = (1 + df['spy_roi']).cumprod()
peak_spy     = cum_spy.cummax()
dd_spy       = (cum_spy - peak_spy) / peak_spy
maxdd_spy    = dd_spy.min()

print("\n=== Summary Performance ===")
print(f"Agent   Sharpe ratio: {sr_agent:.2f}")
print(f"Agent   Max drawdown: {100*maxdd_agent:.2f}%")
print(f"SPY     Sharpe ratio: {sr_spy:.2f}")
print(f"SPY     Max drawdown: {100*maxdd_spy:.2f}%")

# 8) Plot weekly returns
plt.figure()
plt.plot(df.index + 1, df['agent_roi'], label="Agent")
plt.plot(df.index + 1, df['spy_roi'],   label="SPY")
plt.xlabel("Week #")
plt.ylabel("Weekly Return")
plt.title("Agent vs SPY Weekly ROI")
plt.legend()
plt.tight_layout()
save_path = os.path.join(ROOT_DIR, 'src', 'data', 'weekly_roi.png')
plt.savefig(save_path)


env = PortfolioEnv(data_file=TEST_FILE_HOURLY)
obs = env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = env.step(action)

agent_roi = info['portfolio_value']/INITIAL_CASH - 1.0
now = datetime.now()

end_ts = df_test.index.max()
start_ts = end_ts - DateOffset(months=HOLDOUT_MONTHS)

spy = yf.download(
        "SPY",
        start=start_ts.strftime("%Y-%m-%d"),
        end  =end_ts.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
bench_roi = (spy.iloc[-1]/spy.iloc[0] - 1.0).iloc[0]

print(f"\nAverage two-monthly agent ROI: {100*avg_agent:.2f}%")
print(f"Average two-monthly SPY ROI: {100*avg_spy:.2f}%")
import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from src.env.portfolio_env import PortfolioEnv
import src.config as config

def make_daily_train_env():
    return PortfolioEnv(config.TRAIN_FILE_DAILY)

def make_daily_eval_env():
    return PortfolioEnv(config.EVAL_FILE_DAILY)

def make_hourly_train_env():
    return PortfolioEnv(config.TRAIN_FILE_HOURLY)

def make_hourly_eval_env():
    return PortfolioEnv(config.EVAL_FILE_HOURLY)

def main():
    # print("\n=== Stage 1: Daily Pre-Training ===")

    # daily_env = DummyVecEnv([make_daily_train_env])
    # daily_env = VecMonitor(daily_env)

    # daily_eval_env = DummyVecEnv([make_daily_eval_env])
    # daily_eval_env = VecMonitor(daily_eval_env)

    # early_stop_daily  = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5,
    #                                                 min_evals=10,
    #                                                 verbose=1)

    # daily_eval_callback = EvalCallback(
    #     daily_eval_env, 
    #     callback_on_new_best=early_stop_daily,
    #     best_model_save_path=config.DAILY_CB_DIR,
    #     log_path=config.LOG_DIR, 
    #     eval_freq=config.EVAL_FREQ,
    #     n_eval_episodes=3, 
    #     deterministic=True,
    # )

    # model = SAC(
    #     policy='MlpPolicy',
    #     env=daily_env,
    #     learning_rate=config.DAILY_LR,
    #     buffer_size=config.BUFFER_SIZE,
    #     batch_size=config.BATCH_SIZE,
    #     gamma=config.GAMMA,
    #     ent_coef='auto',
    #     verbose=1,
    #     tensorboard_log=config.LOG_DIR,
    #     policy_kwargs=dict(net_arch=[256, 256]),
    # )

    # model.learn(total_timesteps=config.TIMESTEPS, callback=daily_eval_callback)
    # model.save(config.DAILY_MODEL_PATH)
    
    # print("Daily SAC model training complete and saved.")

    print("\n=== Stage 2: Hourly Fine-tuning ===")

    hourly_env = DummyVecEnv([make_hourly_train_env])
    hourly_env = VecMonitor(hourly_env)

    hourly_eval_env = DummyVecEnv([make_hourly_eval_env])
    hourly_eval_env = VecMonitor(hourly_eval_env)

    early_stop_hourly = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5,
                                                    min_evals=10,
                                                    verbose=1)

    hourly_eval_callback = EvalCallback(
        hourly_eval_env, 
        callback_on_new_best=early_stop_hourly,
        best_model_save_path=config.HOURLY_CB_DIR,
        log_path=config.LOG_DIR, 
        eval_freq=config.EVAL_FREQ, 
        deterministic=True,
    )

    model = SAC.load(
        path=config.DAILY_MODEL_PATH,
        env=hourly_env,
        custom_objects={
            "learning_rate": config.HOURLY_LR,
            "buffer_size":   config.BUFFER_SIZE,
            "batch_size":    config.BATCH_SIZE,
            "gamma":         config.GAMMA,
            "ent_coef":      config.ENT_COEF,
        }
    )

    model.learn(total_timesteps=config.TIMESTEPS, callback=hourly_eval_callback)
    model.save(config.HOURLY_MODEL_PATH)

    print("Hourly SAC model training complete and saved.")

if __name__ == '__main__':
    main()

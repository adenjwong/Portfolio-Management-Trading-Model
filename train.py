import os
import shutil

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from portfolio_env import PortfolioEnv

# === Clear existing models/logs if present ===
if os.path.exists("models"):
    shutil.rmtree("models")
# Recreate models directory (including the 'best' subfolder)
os.makedirs("models/best", exist_ok=True)

if os.path.exists("logs"):
    shutil.rmtree("logs")
os.makedirs("logs", exist_ok=True)

# === Load all price data ===
# Assumes data.csv exists in working directory
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# === Split data: train on everything except the last 250 days ===
train_data = data.iloc[:-250]

# === Environment setup for training ===
window_size = 50

def make_train_env():
    return PortfolioEnv(train_data, window_size=window_size)

vec_env = DummyVecEnv([make_train_env])

# === Callbacks ===
checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path="./models/",
    name_prefix="sac_portfolio"
)

# For simplicity, we use a copy of the training environment for evaluation during training.
eval_env = DummyVecEnv([make_train_env])
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    log_path="./logs/",
    eval_freq=5_000,
    deterministic=True,
    render=False
)

# === Initialize SAC model ===
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto"
)

# === Train on train_data only ===
timesteps = 200_000
model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, eval_cb])

# === Save final model ===
model.save("./models/sac_portfolio_final")
print("Training complete on train_data; model saved to ./models/sac_portfolio_final.zip")

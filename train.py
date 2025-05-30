import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from portfolio_env import PortfolioEnv

# === Load price data ===
# Assumes data.csv exists in working directory
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# === Environment ===
window_size = 50

def make_env():
    return PortfolioEnv(data, window_size=window_size)

vec_env = DummyVecEnv([make_env])

# === Callbacks ===
eval_env = DummyVecEnv([make_env])
checkpoint_cb = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="sac_portfolio")
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# === Initialize model ===
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

# === Train ===
timesteps = 200_000
model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, eval_cb])

# === Save final model ===
model.save("./models/sac_portfolio_final")
print("Training complete, model saved to ./models/sac_portfolio_final.zip")

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv

# === Load price data ===
# Use full dataset or split off last year for evaluation
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)
test_data = data.iloc[-250:]  # approx. last 1 year of trading days

# === Create environment for evaluation ===
window_size = 50
env = PortfolioEnv(test_data, window_size=window_size)

# === Load trained model ===
model = SAC.load("models/best/best_model")  # adjust path if needed

# === Rollout ===
obs = env.reset()
portfolio_vals = [env.portfolio_value]
for _ in range(len(test_data) - window_size):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_vals.append(info["portfolio_value"])
    if done:
        break

# === Plot results ===
plt.figure(figsize=(10, 6))
# RL agent equity curve
plt.plot(portfolio_vals, label="RL Agent")
# Buy-and-hold baseline on first asset
dates = test_data.index[window_size:window_size+len(portfolio_vals)]
baseline = test_data.iloc[window_size:, 0] / test_data.iloc[window_size, 0]
plt.plot(baseline.values, label="Buy & Hold")

plt.title("RL Agent vs. Buy & Hold Performance")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

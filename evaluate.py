import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv

# === Load price data ===
# Use full dataset or split off last year for evaluation
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)
test_data = data.iloc[-250:]   # approx. last 1 year of trading days

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

# === Compute performance metrics ===
vals = np.array(portfolio_vals)
# 1. Cumulative return
cum_return = vals[-1] / vals[0] - 1.0
# 2. Maximum drawdown
running_max = np.maximum.accumulate(vals)
drawdowns = 1.0 - vals / running_max
max_drawdown = drawdowns.max()
# 3. Volatility (std of log returns)
log_returns = np.log(vals[1:] / vals[:-1])
volatility = np.std(log_returns)

print(f"Cumulative Return: {cum_return*100:.2f}%")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
print(f"Volatility (std of log returns): {volatility:.4f}")

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

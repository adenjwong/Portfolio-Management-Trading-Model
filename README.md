# RL Portfolio Trading Bot

A simple reinforcement learning project for portfolio management. This repository contains:

- **`fetch_data.py`**: Download historical adjusted-close price data for selected tickers and save to `data.csv`.
- **`portfolio_env.py`**: Custom OpenAI Gym environment to simulate portfolio allocation over time with transaction costs.
- **`.gitignore`**: Exclude Conda envs, caches, and temporary files from version control.

## Prerequisites

- **Conda** (or Miniconda) installed on your system.
- **Python 3.9** (recommended for compatibility).

## Setup

1. **Create and activate the Conda environment**
   ```bash
   conda create -n portfolio-rl python=3.9 -y
   conda activate portfolio-rl
   ```

2. **Install required Python packages**
   ```bash
   pip install gym pandas numpy stable-baselines3 yfinance matplotlib
   ```

## Fetch Historical Data

Run `fetch_data.py` to download and save price data:

```bash
python fetch_data.py
```

- This creates `data.csv` with your selected tickers (default: 15 well-known S&P 500 stocks) for the period 2018–2023.
- The script prints the DataFrame shape and first few rows.

## Portfolio Environment

After generating `data.csv`, load and test the custom Gym environment:

```bash
python - <<EOF
import pandas as pd
from portfolio_env import PortfolioEnv

# Load price data
prices = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Create env with a 10-day window
env = PortfolioEnv(prices, window_size=10)
EOF
```

### Smoke Test

Validate stepping through a few random actions:

```bash
python - <<EOF
import pandas as pd
from portfolio_env import PortfolioEnv

prices = pd.read_csv("data.csv", index_col=0, parse_dates=True)
env = PortfolioEnv(prices, window_size=10)
obs = env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: Reward={reward:.4f}, Portfolio Value={info['portfolio_value']:.4f}")
    if done: break
EOF
```

If you see rewards and portfolio values printed without errors, the environment is working correctly.

## Project Structure

```text
├── fetch_data.py      # Download price data and save to CSV
├── portfolio_env.py   # Custom Gym environment
├── data.csv           # Generated price data
├── .gitignore
└── README.md          # Project overview and instructions
```

## Next Steps

1. **Wrap the environment** in a vectorized wrapper (`DummyVecEnv`) for Stable-Baselines3.
2. **Train an RL agent** (e.g., SAC or PPO) on the environment:
   ```bash
   python train.py
   ```
3. **Evaluate** performance against a buy-and-hold baseline and **visualize** results (equity curve, weight allocations).
4. **Iterate** on hyperparameters, additional assets, or feature enhancements (technical indicators, risk constraints).

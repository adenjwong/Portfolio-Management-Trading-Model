# RL Portfolio Trading Bot

A simple reinforcement learning project for portfolio management. This repository contains:

- **`fetch_data.py`**: Downloads historical adjusted-close price data for 15 well-known S&P 500 tickers (2018-01-01 to present) and saves cleaned `data.csv`.
- **`portfolio_env.py`**: Custom OpenAI Gym environment to simulate portfolio allocation over time with transaction costs.
- **`train.py`**: Trains a Soft Actor-Critic (SAC) agent on `PortfolioEnv`, with checkpointing and evaluation callbacks.
- **`.gitignore`**: Excludes Conda environments, caches, and temporary files.

## Prerequisites

- **Conda** (or Miniconda).
- **Python 3.9**.

## Setup

1. Create and activate the environment:
   ```bash
   conda create -n portfolio-rl python=3.9 -y
   conda activate portfolio-rl
   ```
2. Install required packages:
   ```bash
   pip install gym pandas numpy stable-baselines3 yfinance matplotlib shimmy
   ```

## Fetch Historical Data

Run the data-fetch script:
```bash
python fetch_data.py
```
- Generates `data.csv` with price history.
- Prints its shape and head.

## Portfolio Environment

Load and test your custom Gym env:
```bash
python - <<EOF
import pandas as pd
from portfolio_env import PortfolioEnv

prices = pd.read_csv("data.csv", index_col=0, parse_dates=True)
# 10-day window example
env = PortfolioEnv(prices, window_size=10)
obs = env.reset()
EOF
```

### Smoke Test

Step through a few random actions:
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
    print(f"Step {i+1}: Reward={reward:.4f}, Value={info['portfolio_value']:.4f}")
    if done: break
EOF
```

## Training

Launch SAC training:
```bash
python train.py
```
- Checkpoints saved in `./models/` every 10k steps.
- Best model stored under `./models/best/`.
- Final model at `./models/sac_portfolio_final.zip`.

## Project Structure

```text
├── fetch_data.py      # Data downloader
├── portfolio_env.py   # Gym environment
├── train.py           # Training script
├── data.csv           # Downloaded price data
├── models/            # Checkpoints and saved models
├── logs/              # Evaluation logs
├── .gitignore
└── README.md          # This file
```

## Next Steps

- Tune hyperparameters (window size, learning rate, batch size).
- Add/replace tickers or include technical indicators.
- Implement `evaluate.py` to compare agent vs. buy-and-hold.
- Experiment with other RL algorithms (PPO, TD3).

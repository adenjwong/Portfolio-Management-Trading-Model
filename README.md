## RL Portfolio Trading Bot

A simple reinforcement learning project for portfolio management. This repository contains:

* **`fetch_data.py`**: Script to download historical adjusted-close price data for selected assets.
* **`.gitignore`**: Excludes environment folders and cache files from version control.

### Prerequisites

* **Conda** (or Miniconda) installed on your system.
* **Python 3.9** (recommended).

### Setup

1. **Create and activate the Conda environment**

   ```bash
   conda create -n portfolio-rl python=3.9 -y
   conda activate portfolio-rl
   ```

2. **Install required Python packages**

   ```bash
   pip install gym pandas numpy stable-baselines3 yfinance matplotlib
   ```

### Fetch Historical Data

Run the data-fetch script to download price history for your assets:

```bash
python fetch_data.py
```

This will print the first few rows of your price DataFrame.

### Project Structure

```text
├── fetch_data.py    # Download and preview price data
├── .gitignore       # Ignore Conda envs, caches, etc.
└── README.md        # Project overview and setup
```

### Next Steps

* Implement the `PortfolioEnv` gym environment in `portfolio_env.py`.
* Train an RL agent (e.g., SAC or PPO) using Stable-Baselines3.
* Evaluate and visualize portfolio performance against a buy-and-hold baseline.

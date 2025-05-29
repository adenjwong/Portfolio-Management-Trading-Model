import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    """Custom Environment for portfolio management using RL.

    Observation:
        Type: Box(shape=(window_size * n_assets + n_assets + 1,), dtype=np.float32)
        Contains historical price window (flattened) + previous weights vector.

    Actions:
        Type: Box(shape=(n_assets + 1,), dtype=np.float32)
        Allocation weights for assets and cash, normalized to sum to 1.

    Reward:
        Portfolio return at next time-step minus transaction costs.

    Episode termination:
        When reaching the end of the price data.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, price_df: pd.DataFrame, window_size: int = 50, transaction_cost_pct: float = 0.001):
        super(PortfolioEnv, self).__init__()
        self.price_df = price_df
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct

        # Convert price_df to numpy for speed
        self.prices = self.price_df.values
        self.dates = self.price_df.index
        self.n_assets = self.prices.shape[1]

        # Observation space: window of prices + previous weights
        obs_dim = self.window_size * self.n_assets + (self.n_assets + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: allocation weights for assets + cash
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self):
        # Start at time index = window_size
        self.t = self.window_size
        # Start with full cash allocation
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # last position is cash
        # Initial portfolio value
        self.portfolio_value = 1.0
        return self._get_obs()

    def _get_obs(self):
        window = self.prices[self.t - self.window_size:self.t]
        obs = np.concatenate([window.flatten(), self.weights])
        return obs.astype(np.float32)

    def step(self, action):
        # Normalize allocations to sum=1
        weights = action / (action.sum() + 1e-8)
        prev_prices = self.prices[self.t - 1]
        curr_prices = self.prices[self.t]

        # Compute asset returns (exclude cash)
        asset_returns = curr_prices / prev_prices - 1.0
        # Append cash return = 0
        asset_returns = np.append(asset_returns, 0.0)

        # Portfolio return
        port_return = np.dot(weights, asset_returns)
        # Transaction cost
        cost = self.transaction_cost_pct * np.sum(np.abs(weights - self.weights))
        # Net reward
        reward = port_return - cost

        # Update portfolio value
        self.portfolio_value *= (1 + port_return)

        # Update state
        self.weights = weights
        self.t += 1

        # Check if done
        done = bool(self.t >= len(self.prices))

        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'net_return': port_return,
            'transaction_cost': cost,
            'date': self.dates[self.t - 1] if self.t - 1 < len(self.dates) else None
        }

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.t}, Portfolio Value: {self.portfolio_value:.4f}, Weights: {self.weights}")

    def close(self):
        pass

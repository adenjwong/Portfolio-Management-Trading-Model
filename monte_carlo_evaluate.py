import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv

def monte_carlo_simulation(model, price_data, window_size=50, num_sims=100, sim_horizon=250, seed=None):
    """
    Run Monte Carlo simulations of the RL agent on price_data.
    Each simulation samples returns with replacement from historical returns,
    generates a synthetic price series, and evaluates the agent.
    """
    rng = np.random.default_rng(seed)
    # compute historical returns
    returns = price_data.pct_change().dropna().values
    n_assets = returns.shape[1]

    final_values = []

    for sim in range(num_sims):
        # sample returns with replacement
        sampled_returns = returns[rng.integers(0, len(returns), size=sim_horizon)]
        # build synthetic price series starting at last real price
        last_prices = price_data.iloc[-1].values
        sim_prices = np.vstack([last_prices * np.cumprod(1 + sampled_returns, axis=0)])
        sim_index = pd.date_range(start=price_data.index[-1] + pd.Timedelta(days=1), periods=sim_horizon, freq='B')
        sim_df = pd.DataFrame(sim_prices, index=sim_index, columns=price_data.columns)

        # evaluate agent on synthetic series
        env = PortfolioEnv(sim_df, window_size=window_size)
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        final_values.append(env.portfolio_value)

    return np.array(final_values)

if __name__ == "__main__":
    data = pd.read_csv("data.csv", index_col=0, parse_dates=True)
    test_slice = data.iloc[-250:]
    model = SAC.load("models/best/best_model")

    # run Monte Carlo
    final_vals = monte_carlo_simulation(
        model,
        test_slice,
        window_size=50,
        num_sims=200,
        sim_horizon=250,
        seed=42
    )

    # summary
    mean = final_vals.mean()
    std = final_vals.std()
    pct_5 = np.percentile(final_vals, 5)
    pct_95 = np.percentile(final_vals, 95)
    print(f"Monte Carlo Results over {len(final_vals)} sims:")
    print(f"Mean final portfolio value: {mean:.2f}")
    print(f"Std dev: {std:.2f}")
    print(f"5th percentile: {pct_5:.2f}")
    print(f"95th percentile: {pct_95:.2f}")

    # plot distribution
    plt.figure(figsize=(8,5))
    plt.hist(final_vals, bins=30, density=True, alpha=0.7)
    plt.axvline(mean, color='blue', linestyle='--', label='Mean')
    plt.axvline(pct_5, color='red', linestyle=':', label='5th percentile')
    plt.axvline(pct_95, color='green', linestyle=':', label='95th percentile')
    plt.title('Monte Carlo Distribution of Final Portfolio Values')
    plt.xlabel('Final Portfolio Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

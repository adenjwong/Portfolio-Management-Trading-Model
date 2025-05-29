import yfinance as yf
import pandas as pd

# 1. Pick your assets and date range
# Expand to 15 well-known tickers in the S&P 500
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOG",  # Alphabet (Class C)
    "AMZN",  # Amazon
    "META",  # Meta Platforms (formerly FB)
    "TSLA",  # Tesla
    "BRK-B", # Berkshire Hathaway (Class B)
    "JNJ",   # Johnson & Johnson
    "V",     # Visa
    "WMT",   # Walmart
    "JPM",   # JPMorgan Chase & Co.
    "PG",    # Procter & Gamble
    "XOM",   # Exxon Mobil
    "NVDA",  # NVIDIA
    "NFLX"   # Netflix
]
start_date = "2018-01-01"
end_date   = "2023-12-31"

# 2. Download daily Adjusted Close prices
# Explicitly disable auto_adjust so 'Adj Close' column is included
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False  # ensure 'Adj Close' is returned
)["Adj Close"]

# 3. Fill any missing values and drop remaining NaNs
data = data.ffill().dropna()

# 4. Save to CSV for environment ingestion
data.to_csv("data.csv")
print("Saved data.csv with shape:", data.shape)

# 5. Inspect the head of the DataFrame
print(data.head())

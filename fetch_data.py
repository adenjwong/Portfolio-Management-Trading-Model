import yfinance as yf
import pandas as pd
from datetime import date

# 1. Pick your assets and date range
# Update tickers to include your desired assets
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOG",  # Alphabet (Class C)
    "AMZN",  # Amazon
    "META",  # Meta Platforms
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
# 2. Automatically set end_date to today's date
end_date = date.today().isoformat()

# 3. Download daily Adjusted Close prices
# Explicitly disable auto_adjust so 'Adj Close' column is included
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False  # ensure 'Adj Close' is returned
)["Adj Close"]

# 4. Fill any missing values and drop remaining NaNs
data = data.ffill().dropna()

# 5. Save to CSV for environment ingestion
data.to_csv("data.csv")
print(f"Saved data.csv from {start_date} to {end_date} with shape: {data.shape}")

# 6. Inspect the head of the DataFrame
print(data.head())

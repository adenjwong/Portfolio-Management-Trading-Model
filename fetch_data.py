import yfinance as yf
import pandas as pd

# 1. Pick your assets and date range
tickers = ["AAPL", "MSFT", "GOOG"]
start_date = "2018-01-01"
end_date   = "2023-12-31"

# 2. Download daily Adjusted Close prices
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# 3. Fill any missing values and drop remaining NaNs
data = data.ffill().dropna()

# 4. Inspect the head of the DataFrame
print(data.head())

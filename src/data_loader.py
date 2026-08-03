import yfinance as yf
import pandas as pd

def download_adjusted_close(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    prices = data["Adj Close"].dropna(how="all")
    prices = prices.dropna(axis=1, how="any")
    return prices

def save_prices_to_csv(prices, path="data/processed/adj_close.csv"):
    prices.to_csv(path)
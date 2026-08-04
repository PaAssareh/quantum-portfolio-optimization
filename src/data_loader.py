from pathlib import Path

import pandas as pd
import yfinance as yf

from config import PRICES_FILE


def download_adjusted_close(tickers, start_date, end_date):
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No price data downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers[:1]

    close.index.name = "Date"
    close = close.dropna(how="all")
    return close


def save_prices_to_csv(prices, path=PRICES_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
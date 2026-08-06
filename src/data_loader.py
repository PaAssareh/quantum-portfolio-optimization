import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, tickers, start_date, end_date, source="yahoo"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.source = source

    def load_market_data(self):
        if self.source != "yahoo":
            raise NotImplementedError(f"Data source {self.source} not implemented")

        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"]
            elif "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"]
            else:
                prices = data.xs("Close", axis=1, level=0, drop_level=True)
        else:
            if "Close" in data.columns:
                prices = data["Close"].to_frame()
            elif "Adj Close" in data.columns:
                prices = data["Adj Close"].to_frame()
            else:
                prices = data.copy()

        prices = prices.dropna(how="all")
        prices = prices.ffill().dropna()
        return prices

    def save_processed_data(self, df, path="data/processed/adj_close.csv"):
        df.to_csv(path)
        return path
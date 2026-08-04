import pandas as pd

TRADING_DAYS = 252


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    return returns


def annualized_mean_returns(returns: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.Series:
    return returns.mean() * trading_days


def annualized_covariance(returns: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    return returns.cov() * trading_days
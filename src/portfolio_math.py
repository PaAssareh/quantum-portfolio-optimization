import numpy as np
import pandas as pd

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def annualized_mean_returns(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    return returns.mean() * trading_days

def annualized_covariance(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    return returns.cov() * trading_days

def portfolio_objective(selection, mean_returns, cov_matrix, risk_aversion):
    x = np.array(selection)
    ret = np.dot(mean_returns, x)
    risk = x @ cov_matrix @ x
    return risk_aversion * risk - ret
import numpy as np


class CVaRAnalysis:
    def __init__(self, portfolio_math, alpha=0.95):
        self.portfolio_math = portfolio_math
        self.alpha = alpha

    def portfolio_returns_series(self, weights, returns_df):
        weights = np.array(weights, dtype=float)
        return returns_df.values @ weights

    def compute_var(self, returns_series):
        losses = -np.array(returns_series)
        return np.quantile(losses, self.alpha)

    def compute_cvar(self, returns_series):
        losses = -np.array(returns_series)
        var = np.quantile(losses, self.alpha)
        tail_losses = losses[losses >= var]
        if len(tail_losses) == 0:
            return float(var)
        return float(tail_losses.mean())

    def evaluate_strategy(self, weights, returns_df):
        series = self.portfolio_returns_series(weights, returns_df)
        return {
            "VaR": self.compute_var(series),
            "CVaR": self.compute_cvar(series)
        }
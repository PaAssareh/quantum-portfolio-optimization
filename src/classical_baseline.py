import numpy as np


class ClassicalBaseline:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.portfolio_weights = None

    def optimize_portfolio(self, method="equal_weight"):
        n = len(self.portfolio.assets)

        if method == "equal_weight":
            weights = np.ones(n) / n

        elif method == "minimum_variance":
            cov = self.portfolio.covariance_matrix
            inv_cov = np.linalg.pinv(cov)
            ones = np.ones(n)
            weights = inv_cov @ ones
            if weights.sum() != 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n) / n

        elif method == "mean_variance":
            mu = self.portfolio.expected_returns
            cov = self.portfolio.covariance_matrix + 1e-6 * np.eye(n)
            inv_cov = np.linalg.pinv(cov)
            weights = inv_cov @ mu
            if weights.sum() != 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n) / n

        else:
            weights = np.ones(n) / n

        weights = np.maximum(weights, 0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n) / n

        self.portfolio_weights = weights
        return weights

    def get_portfolio_weights(self):
        return self.portfolio_weights
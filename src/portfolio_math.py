import numpy as np


class PortfolioMath:
    def __init__(self, market_data):
        self.assets = list(market_data.columns)
        returns = market_data.pct_change().dropna()
        self.expected_returns = returns.mean().values
        self.covariance_matrix = returns.cov().values

    def portfolio_return(self, weights):
        weights = np.array(weights, dtype=float)
        return float(np.dot(weights, self.expected_returns))

    def portfolio_volatility(self, weights):
        weights = np.array(weights, dtype=float)
        return float(np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights))))

    def sharpe_ratio(self, weights):
        vol = self.portfolio_volatility(weights)
        if vol == 0:
            return 0.0
        return self.portfolio_return(weights) / vol

    def compare_results(self, quantum_weights, classical_weights, hybrid_weights=None):
        results = {}
        results["quantum_return"] = self.portfolio_return(quantum_weights)
        results["quantum_volatility"] = self.portfolio_volatility(quantum_weights)
        results["quantum_sharpe"] = self.sharpe_ratio(quantum_weights)

        results["classical_return"] = self.portfolio_return(classical_weights)
        results["classical_volatility"] = self.portfolio_volatility(classical_weights)
        results["classical_sharpe"] = self.sharpe_ratio(classical_weights)

        if hybrid_weights is not None:
            results["hybrid_return"] = self.portfolio_return(hybrid_weights)
            results["hybrid_volatility"] = self.portfolio_volatility(hybrid_weights)
            results["hybrid_sharpe"] = self.sharpe_ratio(hybrid_weights)

        return results
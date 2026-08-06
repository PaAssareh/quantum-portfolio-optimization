import pandas as pd


class FeatureEngineering:
    def __init__(self, prices):
        self.prices = prices
        self.returns = self.compute_returns(prices)

    def compute_returns(self, prices):
        return prices.pct_change().dropna()

    def compute_rolling_volatility(self, window=21):
        return self.returns.rolling(window=window).std().dropna()

    def compute_momentum(self, window=21):
        return self.prices.pct_change(periods=window).dropna()

    def compute_drawdown(self):
        rolling_max = self.prices.cummax()
        drawdown = self.prices / rolling_max - 1.0
        return drawdown.dropna()

    def build_feature_matrix(self):
        ret = self.compute_returns(self.prices)
        vol = ret.rolling(21).std()
        mom = self.compute_momentum(21)
        dd = self.compute_drawdown()

        common_index = ret.index.intersection(vol.index).intersection(mom.index).intersection(dd.index)

        features = pd.DataFrame(index=common_index)
        features["market_return"] = ret.mean(axis=1).loc[common_index]
        features["market_volatility"] = vol.mean(axis=1).loc[common_index]
        features["market_momentum"] = mom.mean(axis=1).loc[common_index]
        features["market_drawdown"] = dd.mean(axis=1).loc[common_index]

        return features.dropna()
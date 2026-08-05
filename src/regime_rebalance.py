import numpy as np


class RegimeRebalance:
    def __init__(self, portfolio, base_weights, regimes, regime_probabilities=None, market_signals=None):
        self.portfolio = portfolio
        self.base_weights = np.array(base_weights, dtype=float)
        self.regimes = np.array(regimes)
        self.regime_probabilities = regime_probabilities
        self.market_signals = market_signals

    def _current_regime_confidence(self):
        if self.regime_probabilities is None:
            return 1.0
        probs = np.asarray(self.regime_probabilities)
        if probs.ndim == 1:
            return float(np.max(probs)) if len(probs) else 1.0
        return float(np.max(probs[-1]))

    def _regime_factor(self, regime, confidence):
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if regime == 0:
            return 1.0 + 0.10 * confidence
        if regime == 1:
            return 1.0
        if regime == 2:
            return 1.0 - 0.15 * confidence
        return 1.0

    def _apply_market_signal_adjustment(self, weights):
        weights = np.array(weights, dtype=float)
        if self.market_signals is None:
            return weights

        signals = self.market_signals
        vol = float(signals.get("volatility", 0.0))
        momentum = float(signals.get("momentum", 0.0))

        if vol > 0:
            weights = weights * (1.0 / (1.0 + 0.5 * vol))
        if momentum != 0:
            weights = weights * (1.0 + 0.05 * np.tanh(momentum))
        return weights

    def normalize_weights(self, weights):
        weights = np.array(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            return weights / s
        n = len(weights)
        return np.ones(n) / n

    def rebalance_portfolio(self):
        current_regime = int(self.regimes[-1])
        confidence = self._current_regime_confidence()

        adjusted = self.base_weights.copy()
        factor = self._regime_factor(current_regime, confidence)
        adjusted = adjusted * factor
        adjusted = self._apply_market_signal_adjustment(adjusted)
        adjusted = self.normalize_weights(adjusted)

        return adjusted

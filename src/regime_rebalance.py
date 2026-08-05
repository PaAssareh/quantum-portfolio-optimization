import numpy as np


def regime_alpha(regime: str) -> float:
    regime = str(regime).lower()
    if regime == "bull":
        return 0.70
    return 0.40


def partial_rebalance(previous_weights: np.ndarray, target_weights: np.ndarray, regime: str) -> np.ndarray:
    previous_weights = np.asarray(previous_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    if previous_weights.shape != target_weights.shape:
        raise ValueError("previous_weights and target_weights must have the same shape")

    alpha = regime_alpha(regime)
    blended = (1.0 - alpha) * previous_weights + alpha * target_weights

    total = blended.sum()
    if total > 0:
        blended = blended / total

    return blended


def selection_from_weights(weights: np.ndarray, budget: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    selection = np.zeros(n, dtype=int)

    if budget <= 0:
        return selection

    top_idx = np.argsort(weights)[-budget:]
    selection[top_idx] = 1
    return selection
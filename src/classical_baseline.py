from itertools import combinations
import numpy as np

def brute_force_cardinality(mean_returns, cov_matrix, budget, risk_aversion):
    n = len(mean_returns)
    best_score = float("inf")
    best_selection = None

    for combo in combinations(range(n), budget):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        score = risk_aversion * (x @ cov_matrix @ x) - (mean_returns @ x)

        if score < best_score:
            best_score = score
            best_selection = x.copy()

    return best_selection, best_score
"""QUBO optimizer for portfolio selection (Option A: binary asset selection).

This module builds a QUBO for mean-variance portfolio optimization with:
- Expected return term
- Risk (variance) term
- Cardinality penalty (exactly K assets selected)
- Turnover penalty (discourages large changes from previous weights)

The QUBO is solved using simulated annealing.
"""

from typing import Dict, Tuple, Any
import numpy as np


def build_qubo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    K: int,
    lambda_risk: float = 1.0,
    P_card: float = 10.0,
    P_turn: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """
    Build QUBO for portfolio optimization (Option A: binary asset selection).

    H(z) = -sum_i mu_i * z_i
           + lambda_risk * sum_{i,j} Sigma_{ij} * z_i * z_j
           + P_card * (sum_i z_i - K)^2
           + P_turn * sum_i [z_i * (1/K - w_old_i) + (1 - z_i) * w_old_i]

    Returns:
        QUBO as a dictionary {(i, j): coeff} for i <= j.
    """
    N = len(mu)
    Q: Dict[Tuple[int, int], float] = {}

    def add_term(i: int, j: int, coeff: float):
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + coeff

    # Return term: -sum_i mu_i z_i
    for i in range(N):
        add_term(i, i, -float(mu[i]))

    # Risk term: lambda * sum_{i,j} Sigma_ij z_i z_j
    for i in range(N):
        for j in range(i, N):
            add_term(i, j, float(lambda_risk * Sigma[i, j]))

    # Cardinality penalty: P_card * (sum_i z_i - K)^2
    # = P_card * [sum_i (1 - 2K) z_i + 2 sum_{i<j} z_i z_j + K^2]
    for i in range(N):
        add_term(i, i, P_card * (1 - 2 * K))

    for i in range(N):
        for j in range(i + 1, N):
            add_term(i, j, 2 * P_card)

    # Turnover penalty:
    # P_turn * sum_i [z_i * (1/K - w_old_i) + (1 - z_i) * w_old_i]
    # = P_turn * sum_i [z_i * (1/K - 2*w_old_i) + w_old_i]
    # constant part can be ignored
    for i in range(N):
        add_term(i, i, P_turn * (1.0 / K - 2.0 * float(w_old[i])))

    return Q


def qubo_energy(sample: np.ndarray, Q: Dict[Tuple[int, int], float]) -> float:
    """Compute QUBO energy for a binary sample."""
    e = 0.0
    for (i, j), coeff in Q.items():
        if i == j:
            e += coeff * sample[i]
        else:
            e += coeff * sample[i] * sample[j]
    return float(e)


def solve_qubo_sa(
    Q: Dict[Tuple[int, int], float],
    N: int,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    beta_start: float = 0.01,
    beta_end: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Simple simulated annealing solver for QUBO.
    No extra dependency needed.
    """
    rng = np.random.default_rng(seed)
    best_sample = None
    best_energy = float("inf")

    for _ in range(num_reads):
        sample = rng.integers(0, 2, size=N, dtype=int)
        current_energy = qubo_energy(sample, Q)

        for sweep in range(num_sweeps):
            beta = beta_start + (beta_end - beta_start) * sweep / max(1, num_sweeps - 1)
            idx = rng.integers(0, N)

            candidate = sample.copy()
            candidate[idx] = 1 - candidate[idx]

            candidate_energy = qubo_energy(candidate, Q)
            delta = candidate_energy - current_energy

            if delta < 0 or rng.random() < np.exp(-beta * delta):
                sample = candidate
                current_energy = candidate_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_sample = sample.copy()

    return best_sample, float(best_energy)


def evaluate_portfolio(
    z: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    c_trans: float = 0.02,
) -> Dict[str, float]:
    """
    Evaluate selected portfolio assuming equal weight among selected assets.
    """
    K = int(z.sum())
    if K == 0:
        return {
            "K": 0,
            "expected_return": 0.0,
            "variance": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
        }

    w_new = z / K
    expected_return = float(w_new @ mu)
    variance = float(w_new @ Sigma @ w_new)
    volatility = float(np.sqrt(variance))
    sharpe_ratio = expected_return / volatility if volatility > 1e-12 else 0.0
    turnover = float(np.sum(np.abs(w_new - w_old)))
    transaction_cost = float(c_trans * turnover)

    return {
        "K": K,
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
    }


def run_qubo_optimizer(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    K: int,
    lambda_risk: float = 1.0,
    P_card: float = 10.0,
    P_turn: float = 1.0,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    c_trans: float = 0.02,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full QUBO pipeline:
    1) build QUBO
    2) solve by simulated annealing
    3) evaluate portfolio metrics
    """
    N = len(mu)
    Q = build_qubo(
        mu=mu,
        Sigma=Sigma,
        w_old=w_old,
        K=K,
        lambda_risk=lambda_risk,
        P_card=P_card,
        P_turn=P_turn,
    )

    z, energy = solve_qubo_sa(
        Q=Q,
        N=N,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )

    metrics = evaluate_portfolio(
        z=z,
        mu=mu,
        Sigma=Sigma,
        w_old=w_old,
        c_trans=c_trans,
    )

    weights = (z / z.sum()).tolist() if z.sum() > 0 else []

    return {
        "selected_asset_indices": np.where(z == 1)[0].tolist(),
        "weights": weights,
        "qubo_energy": float(energy),
        "metrics": metrics,
    }
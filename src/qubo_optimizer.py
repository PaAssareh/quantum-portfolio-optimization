from typing import Dict, Tuple, Any, List
import numpy as np


def build_qubo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    K: int,
    tickers: List[str],
    sector_map: Dict[str, str],
    sector_max_weights: Dict[str, float],
    lambda_risk: float = 1.0,
    P_card: float = 10.0,
    P_turn: float = 1.0,
    P_turn_quad: float = 2.0,
    P_sector: float = 20.0,
) -> Dict[Tuple[int, int], float]:
    N = len(mu)
    Q: Dict[Tuple[int, int], float] = {}

    def add_term(i: int, j: int, coeff: float):
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + coeff

    for i in range(N):
        add_term(i, i, -float(mu[i]))

    for i in range(N):
        for j in range(i, N):
            add_term(i, j, float(lambda_risk * Sigma[i, j]))

    for i in range(N):
        add_term(i, i, P_card * (1 - 2 * K))
    for i in range(N):
        for j in range(i + 1, N):
            add_term(i, j, 2 * P_card)

    for i in range(N):
        add_term(i, i, P_turn * (1.0 / K - 2.0 * float(w_old[i])))

    for i in range(N):
        diff = (1.0 / K) - float(w_old[i])
        add_term(i, i, P_turn_quad * (diff ** 2))

    sector_to_indices: Dict[str, List[int]] = {}
    for idx, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "Unknown")
        sector_to_indices.setdefault(sector, []).append(idx)

    for sector, indices in sector_to_indices.items():
        if sector not in sector_max_weights:
            continue

        max_weight = sector_max_weights[sector]
        max_assets_in_sector = int(np.floor(max_weight * K + 1e-9))

        if max_assets_in_sector >= len(indices):
            continue

        for i in indices:
            add_term(i, i, P_sector * (1 - 2 * max_assets_in_sector))

        for idx_i, i in enumerate(indices):
            for j in indices[idx_i + 1:]:
                add_term(i, j, 2 * P_sector)

    return Q


def qubo_energy(sample: np.ndarray, Q: Dict[Tuple[int, int], float]) -> float:
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
    num_reads: int = 250,
    num_sweeps: int = 2500,
    beta_start: float = 0.01,
    beta_end: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
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
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
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
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (
        (expected_return - risk_free_rate) / volatility if volatility > 1e-12 else 0.0
    )
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
    tickers: List[str],
    sector_map: Dict[str, str],
    sector_max_weights: Dict[str, float],
    lambda_risk: float = 1.0,
    P_card: float = 10.0,
    P_turn: float = 1.0,
    P_turn_quad: float = 2.0,
    P_sector: float = 20.0,
    num_reads: int = 250,
    num_sweeps: int = 2500,
    c_trans: float = 0.02,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    N = len(mu)

    Q = build_qubo(
        mu=mu,
        Sigma=Sigma,
        w_old=w_old,
        K=K,
        tickers=tickers,
        sector_map=sector_map,
        sector_max_weights=sector_max_weights,
        lambda_risk=lambda_risk,
        P_card=P_card,
        P_turn=P_turn,
        P_turn_quad=P_turn_quad,
        P_sector=P_sector,
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
        risk_free_rate=risk_free_rate,
    )

    full_weights = np.zeros(N, dtype=float)
    if z.sum() > 0:
        full_weights = z / z.sum()

    return {
        "selected_asset_indices": np.where(z == 1)[0].tolist(),
        "weights": full_weights.tolist(),
        "qubo_energy": float(energy),
        "metrics": metrics,
    }
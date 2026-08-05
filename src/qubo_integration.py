"""QUBO portfolio optimizer integration for the wiser project."""

from typing import Dict, Any, List
import numpy as np

from qubo_optimizer import run_qubo_optimizer as _run_qubo_optimizer
from classical_baseline import sector_constraints_ok


def run_qubo_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    previous_weights: np.ndarray,
    budget: int,
    tickers: List[str],
    sector_map: Dict[str, str],
    sector_max_weights: Dict[str, float],
    lambda_risk: float = 1.0,
    P_card: float = 10.0,
    P_turn: float = 1.0,
    P_turn_quad: float = 2.0,
    P_sector: float = 20.0,
    transaction_cost_penalty: float = 0.02,
    risk_free_rate: float = 0.0,
    num_reads: int = 250,
    num_sweeps: int = 2500,
    seed: int = 42,
) -> Dict[str, Any]:
    n_assets = len(mean_returns)

    qubo_result = _run_qubo_optimizer(
        mu=np.asarray(mean_returns, dtype=float),
        Sigma=np.asarray(cov_matrix, dtype=float),
        w_old=np.asarray(previous_weights, dtype=float),
        K=budget,
        tickers=tickers,
        sector_map=sector_map,
        sector_max_weights=sector_max_weights,
        lambda_risk=lambda_risk,
        P_card=P_card,
        P_turn=P_turn,
        P_turn_quad=P_turn_quad,
        P_sector=P_sector,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        c_trans=transaction_cost_penalty,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )

    selected_indices = qubo_result["selected_asset_indices"]

    weights_array = np.array(qubo_result["weights"], dtype=float)
    if len(weights_array) != n_assets:
        raise ValueError(
            f"Expected full weight vector of length {n_assets}, got {len(weights_array)}"
        )

    selection = np.zeros(n_assets, dtype=int)
    selection[selected_indices] = 1

    metrics = qubo_result["metrics"]
    objective_score = (
        metrics["expected_return"]
        - lambda_risk * metrics["variance"]
        - metrics["transaction_cost"]
    )

    sector_ok, sector_exposures = sector_constraints_ok(
        weights_array, tickers, sector_map, sector_max_weights
    )

    return {
        "selection": selection,
        "weights": weights_array,
        "objective_score": float(objective_score),
        "turnover": float(metrics["turnover"]),
        "transaction_cost": float(metrics["transaction_cost"]),
        "expected_return": float(metrics["expected_return"]),
        "variance": float(metrics["variance"]),
        "volatility": float(metrics["volatility"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "selected_count": int(selection.sum()),
        "constraint_budget_ok": int(selection.sum()) == budget,
        "constraint_long_only_ok": bool(np.all(weights_array >= -1e-8)),
        "constraint_fully_invested_ok": bool(abs(weights_array.sum() - 1.0) <= 1e-8),
        "constraint_sector_ok": bool(sector_ok),
        "sector_exposures": sector_exposures,
        "selected_assets": [tickers[i] for i in selected_indices],
    }
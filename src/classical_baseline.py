import itertools
import numpy as np


def compute_turnover(weights, previous_weights):
    weights = np.asarray(weights, dtype=float)
    previous_weights = np.asarray(previous_weights, dtype=float)
    return float(np.sum(np.abs(weights - previous_weights)))


def score_portfolio(
    weights,
    mean_returns,
    cov_matrix,
    risk_aversion,
    previous_weights=None,
    transaction_cost_penalty=0.0,
):
    weights = np.asarray(weights, dtype=float)
    expected_return = float(np.dot(mean_returns, weights))
    variance = float(weights @ cov_matrix @ weights)

    turnover = 0.0
    transaction_cost = 0.0

    if previous_weights is not None:
        turnover = compute_turnover(weights, previous_weights)
        transaction_cost = transaction_cost_penalty * turnover

    utility = expected_return - risk_aversion * variance - transaction_cost
    return utility, expected_return, variance, turnover, transaction_cost


def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    weights = np.asarray(weights, dtype=float)

    expected_return = float(np.dot(mean_returns, weights))
    variance = float(weights @ cov_matrix @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

    return {
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }


def compute_sector_exposures(weights, tickers, sector_map):
    exposures = {sector: 0.0 for sector in set(sector_map.values())}
    for ticker, weight in zip(tickers, weights):
        exposures[sector_map[ticker]] += float(weight)
    return exposures


def sector_constraints_ok(weights, tickers, sector_map, sector_max_weights, tol=1e-8):
    exposures = compute_sector_exposures(weights, tickers, sector_map)
    for sector, exposure in exposures.items():
        max_allowed = sector_max_weights.get(sector, 1.0)
        if exposure - max_allowed > tol:
            return False, exposures
    return True, exposures


def check_constraints(selection, weights, budget, tickers, sector_map, sector_max_weights, tol=1e-8):
    selection = np.asarray(selection, dtype=int)
    weights = np.asarray(weights, dtype=float)

    selected_count = int(selection.sum())
    sector_ok, sector_exposures = sector_constraints_ok(
        weights, tickers, sector_map, sector_max_weights, tol=tol
    )

    return {
        "selected_count": selected_count,
        "constraint_budget_ok": selected_count == budget,
        "constraint_long_only_ok": bool(np.all(weights >= -tol)),
        "constraint_fully_invested_ok": bool(abs(weights.sum() - 1.0) <= tol),
        "constraint_sector_ok": sector_ok,
        "sector_exposures": sector_exposures,
    }


def brute_force_cardinality(
    mean_returns,
    cov_matrix,
    budget,
    risk_aversion,
    tickers,
    sector_map,
    sector_max_weights,
    previous_weights=None,
    transaction_cost_penalty=0.0,
    risk_free_rate=0.0,
):
    n = len(mean_returns)
    if budget <= 0 or budget > n:
        raise ValueError("budget must be between 1 and number of assets")

    if previous_weights is not None:
        previous_weights = np.asarray(previous_weights, dtype=float)
        if len(previous_weights) != n:
            raise ValueError("previous_weights length must match number of assets")

    best_score = -np.inf
    best_combo = None
    best_weights = None
    best_turnover = None
    best_transaction_cost = None

    for combo in itertools.combinations(range(n), budget):
        weights = np.zeros(n, dtype=float)
        weights[list(combo)] = 1.0 / budget

        sector_ok, _ = sector_constraints_ok(
            weights, tickers, sector_map, sector_max_weights
        )
        if not sector_ok:
            continue

        score, _, _, turnover, transaction_cost = score_portfolio(
            weights,
            mean_returns,
            cov_matrix,
            risk_aversion,
            previous_weights=previous_weights,
            transaction_cost_penalty=transaction_cost_penalty,
        )

        if score > best_score:
            best_score = score
            best_combo = combo
            best_weights = weights.copy()
            best_turnover = turnover
            best_transaction_cost = transaction_cost

    if best_combo is None:
        raise ValueError("No feasible portfolio found under constraints")

    selection = np.zeros(n, dtype=int)
    selection[list(best_combo)] = 1

    metrics = portfolio_metrics(
        best_weights,
        mean_returns,
        cov_matrix,
        risk_free_rate=risk_free_rate,
    )
    constraints = check_constraints(
        selection,
        best_weights,
        budget,
        tickers,
        sector_map,
        sector_max_weights,
    )

    return {
        "selection": selection,
        "weights": best_weights,
        "objective_score": float(best_score),
        "turnover": float(best_turnover),
        "transaction_cost": float(best_transaction_cost),
        **metrics,
        **constraints,
    }
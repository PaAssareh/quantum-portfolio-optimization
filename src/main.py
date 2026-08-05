import sys
from pathlib import Path

sys.path.append("src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    BUDGET,
    RISK_AVERSION,
    TRANSACTION_COST_PENALTY,
    DATA_DIR,
    PRICES_FILE,
    RESULTS_FILE,
    COMPARISON_RESULTS_FILE,
    SUMMARY_RESULTS_FILE,
    CHART_FILE,
    SECTOR_MAP,
    SECTOR_MAX_WEIGHTS,
    PREVIOUS_WEIGHTS,
)
from data_loader import download_adjusted_close, save_prices_to_csv
from portfolio_math import compute_daily_returns, annualized_mean_returns, annualized_covariance
from classical_baseline import brute_force_cardinality, check_constraints
from qubo_integration import run_qubo_portfolio
from regime_detector import detect_regime, regime_parameters
from regime_rebalance import partial_rebalance, selection_from_weights


def ensure_previous_weights(tickers):
    return np.array([PREVIOUS_WEIGHTS.get(t, 0.0) for t in tickers], dtype=float)


def equal_weight_benchmark(mean_returns, cov_matrix, previous_weights, tickers):
    n = len(tickers)
    weights = np.ones(n, dtype=float) / n
    expected_return = float(np.dot(mean_returns, weights))
    variance = float(weights @ cov_matrix @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    turnover = float(np.sum(np.abs(weights - previous_weights)))
    transaction_cost = float(TRANSACTION_COST_PENALTY * turnover)
    objective_score = expected_return - RISK_AVERSION * variance - transaction_cost
    constraints = check_constraints(
        np.ones(n, dtype=int),
        weights,
        budget=n,
        tickers=tickers,
        sector_map=SECTOR_MAP,
        sector_max_weights={k: 1.0 for k in SECTOR_MAX_WEIGHTS},
    )
    return {
        "name": "Equal-Weight Benchmark",
        "selection": np.ones(n, dtype=int),
        "weights": weights,
        "objective_score": float(objective_score),
        "turnover": float(turnover),
        "transaction_cost": float(transaction_cost),
        "expected_return": float(expected_return),
        "variance": float(variance),
        "volatility": float(volatility),
        "sharpe_ratio": float(expected_return / max(volatility, 1e-12)),
        **constraints,
    }


def normalize_result(
    name,
    selection,
    weights,
    objective_score,
    turnover,
    transaction_cost,
    expected_return,
    variance,
    volatility,
    sharpe_ratio,
    constraints,
):
    return {
        "name": name,
        "selection": np.asarray(selection, dtype=int),
        "weights": np.asarray(weights, dtype=float),
        "objective_score": float(objective_score),
        "turnover": float(turnover),
        "transaction_cost": float(transaction_cost),
        "expected_return": float(expected_return),
        "variance": float(variance),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        **constraints,
    }


def print_report(title, result, tickers):
    selected_assets = [tickers[i] for i, v in enumerate(result["selection"]) if int(v) == 1]
    selected_weights = [round(float(w), 4) for w in result["weights"][result["weights"] > 0]]

    print(f"\n=== {title} ===")
    print(f"Selected assets: {selected_assets}")
    print(f"Selected weights: {selected_weights}")
    print(f"Objective score: {result['objective_score']}")
    print(f"Expected return: {result['expected_return']}")
    print(f"Variance: {result['variance']}")
    print(f"Volatility: {result['volatility']}")
    print(f"Sharpe ratio: {result['sharpe_ratio']}")
    print(f"Turnover: {result['turnover']}")
    print(f"Transaction cost: {result['transaction_cost']}")
    print("Constraint checks:")
    print(f"  budget: {result['constraint_budget_ok']}")
    print(f"  long_only: {result['constraint_long_only_ok']}")
    print(f"  fully_invested: {result['constraint_fully_invested_ok']}")
    print(f"  sector: {result['constraint_sector_ok']}")
    print(f"Sector exposures: {result['sector_exposures']}")


def save_outputs(results_df, summary_df):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(COMPARISON_RESULTS_FILE, index=False)
    summary_df.to_csv(SUMMARY_RESULTS_FILE, index=False)

    plt.figure(figsize=(10, 5))
    names = results_df["model"].tolist()
    sharpe = results_df["sharpe_ratio"].tolist()
    objective = results_df["objective_score"].tolist()

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width / 2, sharpe, width, label="Sharpe ratio")
    plt.bar(x + width / 2, objective, width, label="Objective score")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Value")
    plt.title("Portfolio Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=200)
    plt.close()


def run_static_and_regime_aware(prices, tickers):
    returns = compute_daily_returns(prices)
    mean_returns = annualized_mean_returns(returns).reindex(tickers).to_numpy(dtype=float)
    cov_matrix = annualized_covariance(returns).reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    previous_weights = ensure_previous_weights(tickers)

    classical = brute_force_cardinality(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        budget=BUDGET,
        risk_aversion=RISK_AVERSION,
        tickers=tickers,
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
        previous_weights=previous_weights,
        transaction_cost_penalty=TRANSACTION_COST_PENALTY,
    )

    static_qubo = run_qubo_portfolio(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        previous_weights=previous_weights,
        budget=BUDGET,
        tickers=tickers,
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
        lambda_risk=RISK_AVERSION,
        P_card=15.0,
        P_turn=2.0,
        P_turn_quad=2.0,
        P_sector=25.0,
        transaction_cost_penalty=TRANSACTION_COST_PENALTY,
        num_reads=250,
        num_sweeps=2500,
        seed=42,
    )

    regimes, _ = detect_regime(prices)
    latest_regime = str(regimes.iloc[-1]) if not regimes.empty else "bear"
    rp = regime_parameters(latest_regime)

    regime_qubo = run_qubo_portfolio(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        previous_weights=previous_weights,
        budget=BUDGET,
        tickers=tickers,
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
        lambda_risk=rp["lambda_risk"],
        P_card=rp["P_card"],
        P_turn=rp["P_turn"],
        P_turn_quad=rp["P_turn_quad"],
        P_sector=rp["P_sector"],
        transaction_cost_penalty=TRANSACTION_COST_PENALTY,
        num_reads=250,
        num_sweeps=2500,
        seed=42,
    )

    blended_weights = partial_rebalance(
        previous_weights,
        np.array(regime_qubo["weights"], dtype=float),
        latest_regime,
    )
    blended_selection = selection_from_weights(blended_weights, BUDGET)

    blended_expected_return = float(np.dot(mean_returns, blended_weights))
    blended_variance = float(blended_weights @ cov_matrix @ blended_weights)
    blended_volatility = float(np.sqrt(max(blended_variance, 0.0)))
    blended_turnover = float(np.sum(np.abs(blended_weights - previous_weights)))
    blended_transaction_cost = float(TRANSACTION_COST_PENALTY * blended_turnover)
    blended_objective_score = (
        blended_expected_return
        - rp["lambda_risk"] * blended_variance
        - blended_transaction_cost
    )
    blended_constraints = check_constraints(
        blended_selection,
        blended_weights,
        budget=BUDGET,
        tickers=tickers,
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
    )

    equal_weight = equal_weight_benchmark(mean_returns, cov_matrix, previous_weights, tickers)

    classical_result = normalize_result(
        "Classical Baseline",
        classical["selection"],
        classical["weights"],
        classical["objective_score"],
        classical["turnover"],
        classical["transaction_cost"],
        classical["expected_return"],
        classical["variance"],
        classical["volatility"],
        classical["sharpe_ratio"],
        {
            "constraint_budget_ok": classical["constraint_budget_ok"],
            "constraint_long_only_ok": classical["constraint_long_only_ok"],
            "constraint_fully_invested_ok": classical["constraint_fully_invested_ok"],
            "constraint_sector_ok": classical["constraint_sector_ok"],
            "sector_exposures": classical["sector_exposures"],
        },
    )

    static_result = normalize_result(
        "Static QUBO+",
        static_qubo["selection"],
        static_qubo["weights"],
        static_qubo["objective_score"],
        static_qubo["turnover"],
        static_qubo["transaction_cost"],
        static_qubo["expected_return"],
        static_qubo["variance"],
        static_qubo["volatility"],
        static_qubo["sharpe_ratio"],
        {
            "constraint_budget_ok": static_qubo["constraint_budget_ok"],
            "constraint_long_only_ok": static_qubo["constraint_long_only_ok"],
            "constraint_fully_invested_ok": static_qubo["constraint_fully_invested_ok"],
            "constraint_sector_ok": static_qubo["constraint_sector_ok"],
            "sector_exposures": static_qubo["sector_exposures"],
        },
    )

    regime_result = normalize_result(
        f"Regime-Aware QUBO+ ({latest_regime})",
        blended_selection,
        blended_weights,
        blended_objective_score,
        blended_turnover,
        blended_transaction_cost,
        blended_expected_return,
        blended_variance,
        blended_volatility,
        blended_expected_return / max(blended_volatility, 1e-12),
        {
            "constraint_budget_ok": blended_constraints["constraint_budget_ok"],
            "constraint_long_only_ok": blended_constraints["constraint_long_only_ok"],
            "constraint_fully_invested_ok": blended_constraints["constraint_fully_invested_ok"],
            "constraint_sector_ok": blended_constraints["constraint_sector_ok"],
            "sector_exposures": blended_constraints["sector_exposures"],
        },
    )

    benchmark_result = normalize_result(
        "Equal-Weight Benchmark",
        equal_weight["selection"],
        equal_weight["weights"],
        equal_weight["objective_score"],
        equal_weight["turnover"],
        equal_weight["transaction_cost"],
        equal_weight["expected_return"],
        equal_weight["variance"],
        equal_weight["volatility"],
        equal_weight["sharpe_ratio"],
        {
            "constraint_budget_ok": equal_weight["constraint_budget_ok"],
            "constraint_long_only_ok": equal_weight["constraint_long_only_ok"],
            "constraint_fully_invested_ok": equal_weight["constraint_fully_invested_ok"],
            "constraint_sector_ok": equal_weight["constraint_sector_ok"],
            "sector_exposures": equal_weight["sector_exposures"],
        },
    )

    print(f"\nDetected latest market regime: {latest_regime}")
    print_report("Optimized Baseline (Classical)", classical_result, tickers)
    print_report("Static QUBO+", static_result, tickers)
    print_report("Regime-Aware QUBO+ (blended)", regime_result, tickers)
    print_report("Equal-Weight Benchmark", benchmark_result, tickers)

    comparison = pd.DataFrame([
        {
            "model": classical_result["name"],
            "objective_score": classical_result["objective_score"],
            "expected_return": classical_result["expected_return"],
            "variance": classical_result["variance"],
            "volatility": classical_result["volatility"],
            "sharpe_ratio": classical_result["sharpe_ratio"],
            "turnover": classical_result["turnover"],
            "transaction_cost": classical_result["transaction_cost"],
            "constraint_budget_ok": classical_result["constraint_budget_ok"],
            "constraint_long_only_ok": classical_result["constraint_long_only_ok"],
            "constraint_fully_invested_ok": classical_result["constraint_fully_invested_ok"],
            "constraint_sector_ok": classical_result["constraint_sector_ok"],
        },
        {
            "model": static_result["name"],
            "objective_score": static_result["objective_score"],
            "expected_return": static_result["expected_return"],
            "variance": static_result["variance"],
            "volatility": static_result["volatility"],
            "sharpe_ratio": static_result["sharpe_ratio"],
            "turnover": static_result["turnover"],
            "transaction_cost": static_result["transaction_cost"],
            "constraint_budget_ok": static_result["constraint_budget_ok"],
            "constraint_long_only_ok": static_result["constraint_long_only_ok"],
            "constraint_fully_invested_ok": static_result["constraint_fully_invested_ok"],
            "constraint_sector_ok": static_result["constraint_sector_ok"],
        },
        {
            "model": regime_result["name"],
            "objective_score": regime_result["objective_score"],
            "expected_return": regime_result["expected_return"],
            "variance": regime_result["variance"],
            "volatility": regime_result["volatility"],
            "sharpe_ratio": regime_result["sharpe_ratio"],
            "turnover": regime_result["turnover"],
            "transaction_cost": regime_result["transaction_cost"],
            "constraint_budget_ok": regime_result["constraint_budget_ok"],
            "constraint_long_only_ok": regime_result["constraint_long_only_ok"],
            "constraint_fully_invested_ok": regime_result["constraint_fully_invested_ok"],
            "constraint_sector_ok": regime_result["constraint_sector_ok"],
        },
        {
            "model": benchmark_result["name"],
            "objective_score": benchmark_result["objective_score"],
            "expected_return": benchmark_result["expected_return"],
            "variance": benchmark_result["variance"],
            "volatility": benchmark_result["volatility"],
            "sharpe_ratio": benchmark_result["sharpe_ratio"],
            "turnover": benchmark_result["turnover"],
            "transaction_cost": benchmark_result["transaction_cost"],
            "constraint_budget_ok": benchmark_result["constraint_budget_ok"],
            "constraint_long_only_ok": benchmark_result["constraint_long_only_ok"],
            "constraint_fully_invested_ok": benchmark_result["constraint_fully_invested_ok"],
            "constraint_sector_ok": benchmark_result["constraint_sector_ok"],
        },
    ])

    summary = pd.DataFrame([
        {"metric": "Static QUBO+ vs Classical objective delta", "value": static_result["objective_score"] - classical_result["objective_score"]},
        {"metric": "Static QUBO+ vs Classical Sharpe delta", "value": static_result["sharpe_ratio"] - classical_result["sharpe_ratio"]},
        {"metric": "Regime-Aware QUBO+ vs Classical objective delta", "value": regime_result["objective_score"] - classical_result["objective_score"]},
        {"metric": "Regime-Aware QUBO+ vs Classical Sharpe delta", "value": regime_result["sharpe_ratio"] - classical_result["sharpe_ratio"]},
        {"metric": "Regime-Aware QUBO+ vs Static QUBO+ objective delta", "value": regime_result["objective_score"] - static_result["objective_score"]},
        {"metric": "Regime-Aware QUBO+ vs Static QUBO+ Sharpe delta", "value": regime_result["sharpe_ratio"] - static_result["sharpe_ratio"]},
        {"metric": "Static QUBO+ vs Benchmark Sharpe delta", "value": static_result["sharpe_ratio"] - benchmark_result["sharpe_ratio"]},
        {"metric": "Regime-Aware QUBO+ vs Benchmark Sharpe delta", "value": regime_result["sharpe_ratio"] - benchmark_result["sharpe_ratio"]},
    ])

    save_outputs(comparison, summary)

    print("\n=== Comparison Summary ===")
    print("\n-- Classical vs Benchmark --")
    print(f"Delta objective score: {classical_result['objective_score'] - benchmark_result['objective_score']}")
    print(f"Delta Sharpe ratio: {classical_result['sharpe_ratio'] - benchmark_result['sharpe_ratio']}")

    print("\n-- Static QUBO+ vs Benchmark --")
    print(f"Delta objective score: {static_result['objective_score'] - benchmark_result['objective_score']}")
    print(f"Delta Sharpe ratio: {static_result['sharpe_ratio'] - benchmark_result['sharpe_ratio']}")

    print("\n-- Regime-Aware QUBO+ vs Benchmark --")
    print(f"Delta objective score: {regime_result['objective_score'] - benchmark_result['objective_score']}")
    print(f"Delta Sharpe ratio: {regime_result['sharpe_ratio'] - benchmark_result['sharpe_ratio']}")

    print("\n-- Regime-Aware QUBO+ vs Static QUBO+ --")
    print(f"Delta objective score: {regime_result['objective_score'] - static_result['objective_score']}")
    print(f"Delta Sharpe ratio: {regime_result['sharpe_ratio'] - static_result['sharpe_ratio']}")

    print(f"\nPrices saved to {PRICES_FILE}")
    print(f"Baseline results saved to {RESULTS_FILE}")
    print(f"Comparison results saved to {COMPARISON_RESULTS_FILE}")
    print(f"Summary results saved to {SUMMARY_RESULTS_FILE}")
    print(f"Comparison chart saved to {CHART_FILE}")


def main():
    prices = download_adjusted_close(TICKERS, START_DATE, END_DATE)
    save_prices_to_csv(prices, PRICES_FILE)
    run_static_and_regime_aware(prices, TICKERS)


if __name__ == "__main__":
    main()
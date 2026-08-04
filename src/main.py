import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
from portfolio_math import (
    compute_daily_returns,
    annualized_mean_returns,
    annualized_covariance,
)
from classical_baseline import brute_force_cardinality, portfolio_metrics, compute_turnover
from qubo_integration import run_qubo_portfolio


RISK_FREE_RATE = 0.0


def build_comparison_chart(comparison_df, output_path):
    metric_names = [
        ("expected_return", "Exp Return"),
        ("volatility", "Volatility"),
        ("sharpe_ratio", "Sharpe"),
        ("objective_score", "Objective"),
    ]

    fig = go.Figure()

    for _, row in comparison_df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["model"],
                x=[label for _, label in metric_names],
                y=[row[col] for col, _ in metric_names],
                text=[f"{row[col]:.3f}" for col, _ in metric_names],
                textposition="outside",
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title={
            "text": "Portfolio metrics vs benchmark (2020-2025)<br><span style='font-size: 18px; font-weight: normal;'>Source: model outputs | Optimized portfolio compared with equal-weight benchmark</span>"
        },
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Metric")
    fig.update_yaxes(title_text="Value")
    fig.write_image(output_path)

    with open(output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "caption": "Portfolio vs benchmark metrics",
                "description": "Grouped bar chart comparing expected return, volatility, Sharpe ratio, and objective score for the optimized portfolio and the equal-weight benchmark.",
            },
            f,
        )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    prices = download_adjusted_close(TICKERS, START_DATE, END_DATE)
    save_prices_to_csv(prices, PRICES_FILE)

    returns = compute_daily_returns(prices)
    mean_returns = annualized_mean_returns(returns).values
    cov_matrix = annualized_covariance(returns).values

    previous_weights_array = np.array(
        [PREVIOUS_WEIGHTS[ticker] for ticker in prices.columns],
        dtype=float,
    )

    # === Classical Optimized Baseline ===
    optimized = brute_force_cardinality(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        budget=BUDGET,
        risk_aversion=RISK_AVERSION,
        tickers=prices.columns.tolist(),
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
        previous_weights=previous_weights_array,
        transaction_cost_penalty=TRANSACTION_COST_PENALTY,
        risk_free_rate=RISK_FREE_RATE,
    )

    selection = optimized["selection"]
    weights = optimized["weights"]

    selected_assets = [
        ticker for ticker, flag in zip(prices.columns, selection) if flag == 1
    ]
    selected_weights = [float(w) for w in weights if w > 0]

    baseline_row = {
        "model": "optimized_baseline",
        "selected_assets": ", ".join(selected_assets),
        "selected_weights": ", ".join([f"{w:.4f}" for w in selected_weights]),
        "objective_score": optimized["objective_score"],
        "expected_return": optimized["expected_return"],
        "variance": optimized["variance"],
        "volatility": optimized["volatility"],
        "sharpe_ratio": optimized["sharpe_ratio"],
        "turnover": optimized["turnover"],
        "transaction_cost": optimized["transaction_cost"],
        "budget": BUDGET,
        "selected_count": optimized["selected_count"],
        "constraint_budget_ok": optimized["constraint_budget_ok"],
        "constraint_long_only_ok": optimized["constraint_long_only_ok"],
        "constraint_fully_invested_ok": optimized["constraint_fully_invested_ok"],
        "constraint_sector_ok": optimized["constraint_sector_ok"],
        "sector_exposures": str(optimized["sector_exposures"]),
    }

    # === QUBO Optimizer ===
    qubo = run_qubo_portfolio(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        previous_weights=previous_weights_array,
        budget=BUDGET,
        tickers=prices.columns.tolist(),
        sector_map=SECTOR_MAP,
        sector_max_weights=SECTOR_MAX_WEIGHTS,
        lambda_risk=RISK_AVERSION,
        P_card=10.0,
        P_turn=1.0,
        transaction_cost_penalty=TRANSACTION_COST_PENALTY,
        risk_free_rate=RISK_FREE_RATE,
        num_reads=100,
        num_sweeps=1000,
        seed=42,
    )

    qubo_selection = qubo["selection"]
    qubo_weights = qubo["weights"]

    qubo_selected_assets = [
        ticker for ticker, flag in zip(prices.columns, qubo_selection) if flag == 1
    ]
    qubo_selected_weights = [float(w) for w in qubo_weights if w > 0]

    qubo_row = {
        "model": "qubo_optimizer",
        "selected_assets": ", ".join(qubo_selected_assets),
        "selected_weights": ", ".join([f"{w:.4f}" for w in qubo_selected_weights]),
        "objective_score": qubo["objective_score"],
        "expected_return": qubo["expected_return"],
        "variance": qubo["variance"],
        "volatility": qubo["volatility"],
        "sharpe_ratio": qubo["sharpe_ratio"],
        "turnover": qubo["turnover"],
        "transaction_cost": qubo["transaction_cost"],
        "budget": BUDGET,
        "selected_count": qubo["selected_count"],
        "constraint_budget_ok": qubo["constraint_budget_ok"],
        "constraint_long_only_ok": qubo["constraint_long_only_ok"],
        "constraint_fully_invested_ok": qubo["constraint_fully_invested_ok"],
        "constraint_sector_ok": qubo["constraint_sector_ok"],
        "sector_exposures": str(qubo["sector_exposures"]),
    }

    # === Equal-Weight Benchmark ===
    n_assets = len(prices.columns)
    ew_weights = np.ones(n_assets) / n_assets

    ew_metrics = portfolio_metrics(
        ew_weights,
        mean_returns,
        cov_matrix,
        risk_free_rate=RISK_FREE_RATE,
    )
    ew_turnover = compute_turnover(ew_weights, previous_weights_array)
    ew_transaction_cost = TRANSACTION_COST_PENALTY * ew_turnover

    ew_row = {
        "model": "equal_weight_benchmark",
        "selected_assets": ", ".join(prices.columns.tolist()),
        "selected_weights": ", ".join([f"{w:.4f}" for w in ew_weights]),
        "objective_score": ew_metrics["expected_return"] - RISK_AVERSION * ew_metrics["variance"] - ew_transaction_cost,
        "expected_return": ew_metrics["expected_return"],
        "variance": ew_metrics["variance"],
        "volatility": ew_metrics["volatility"],
        "sharpe_ratio": ew_metrics["sharpe_ratio"],
        "turnover": ew_turnover,
        "transaction_cost": ew_transaction_cost,
        "budget": n_assets,
        "selected_count": n_assets,
        "constraint_budget_ok": True,
        "constraint_long_only_ok": True,
        "constraint_fully_invested_ok": True,
        "constraint_sector_ok": False,
        "sector_exposures": "benchmark_not_sector_constrained",
    }

    # === Save Results ===
    # Classical baseline only (for backward compatibility)
    pd.DataFrame([baseline_row]).to_csv(RESULTS_FILE, index=False)

    # Comparison: classical + QUBO + benchmark
    comparison_df = pd.DataFrame([baseline_row, qubo_row, ew_row])
    comparison_df.to_csv(COMPARISON_RESULTS_FILE, index=False)

    # Summary: classical vs benchmark
    summary_row_classical = {
        "optimized_model": "optimized_baseline",
        "benchmark_model": "equal_weight_benchmark",
        "delta_expected_return": baseline_row["expected_return"] - ew_row["expected_return"],
        "delta_variance": baseline_row["variance"] - ew_row["variance"],
        "delta_volatility": baseline_row["volatility"] - ew_row["volatility"],
        "delta_sharpe_ratio": baseline_row["sharpe_ratio"] - ew_row["sharpe_ratio"],
        "delta_objective_score": baseline_row["objective_score"] - ew_row["objective_score"],
        "delta_turnover": baseline_row["turnover"] - ew_row["turnover"],
        "delta_transaction_cost": baseline_row["transaction_cost"] - ew_row["transaction_cost"],
        "optimized_beats_benchmark_on_objective": baseline_row["objective_score"] > ew_row["objective_score"],
        "optimized_beats_benchmark_on_sharpe": baseline_row["sharpe_ratio"] > ew_row["sharpe_ratio"],
    }

    # Summary: QUBO vs benchmark
    summary_row_qubo = {
        "optimized_model": "qubo_optimizer",
        "benchmark_model": "equal_weight_benchmark",
        "delta_expected_return": qubo_row["expected_return"] - ew_row["expected_return"],
        "delta_variance": qubo_row["variance"] - ew_row["variance"],
        "delta_volatility": qubo_row["volatility"] - ew_row["volatility"],
        "delta_sharpe_ratio": qubo_row["sharpe_ratio"] - ew_row["sharpe_ratio"],
        "delta_objective_score": qubo_row["objective_score"] - ew_row["objective_score"],
        "delta_turnover": qubo_row["turnover"] - ew_row["turnover"],
        "delta_transaction_cost": qubo_row["transaction_cost"] - ew_row["transaction_cost"],
        "optimized_beats_benchmark_on_objective": qubo_row["objective_score"] > ew_row["objective_score"],
        "optimized_beats_benchmark_on_sharpe": qubo_row["sharpe_ratio"] > ew_row["sharpe_ratio"],
    }

    # Summary: QUBO vs classical
    summary_row_qubo_vs_classical = {
        "qubo_model": "qubo_optimizer",
        "classical_model": "optimized_baseline",
        "delta_expected_return": qubo_row["expected_return"] - baseline_row["expected_return"],
        "delta_variance": qubo_row["variance"] - baseline_row["variance"],
        "delta_volatility": qubo_row["volatility"] - baseline_row["volatility"],
        "delta_sharpe_ratio": qubo_row["sharpe_ratio"] - baseline_row["sharpe_ratio"],
        "delta_objective_score": qubo_row["objective_score"] - baseline_row["objective_score"],
        "delta_turnover": qubo_row["turnover"] - baseline_row["turnover"],
        "delta_transaction_cost": qubo_row["transaction_cost"] - baseline_row["transaction_cost"],
        "qubo_beats_classical_on_objective": qubo_row["objective_score"] > baseline_row["objective_score"],
        "qubo_beats_classical_on_sharpe": qubo_row["sharpe_ratio"] > baseline_row["sharpe_ratio"],
    }

    pd.DataFrame([summary_row_classical, summary_row_qubo, summary_row_qubo_vs_classical]).to_csv(SUMMARY_RESULTS_FILE, index=False)
    build_comparison_chart(comparison_df, CHART_FILE)

    # === Print Results ===
    print("\n=== Optimized Baseline (Classical) ===")
    print("Selected assets:", selected_assets)
    print("Selected weights:", selected_weights)
    print("Objective score:", optimized["objective_score"])
    print("Expected return:", optimized["expected_return"])
    print("Variance:", optimized["variance"])
    print("Volatility:", optimized["volatility"])
    print("Sharpe ratio:", optimized["sharpe_ratio"])
    print("Turnover:", optimized["turnover"])
    print("Transaction cost:", optimized["transaction_cost"])
    print("Constraint checks:")
    print("  budget:", optimized["constraint_budget_ok"])
    print("  long_only:", optimized["constraint_long_only_ok"])
    print("  fully_invested:", optimized["constraint_fully_invested_ok"])
    print("  sector:", optimized["constraint_sector_ok"])
    print("Sector exposures:", optimized["sector_exposures"])

    print("\n=== QUBO Optimizer ===")
    print("Selected assets:", qubo_selected_assets)
    print("Selected weights:", qubo_selected_weights)
    print("Objective score:", qubo["objective_score"])
    print("Expected return:", qubo["expected_return"])
    print("Variance:", qubo["variance"])
    print("Volatility:", qubo["volatility"])
    print("Sharpe ratio:", qubo["sharpe_ratio"])
    print("Turnover:", qubo["turnover"])
    print("Transaction cost:", qubo["transaction_cost"])
    print("Constraint checks:")
    print("  budget:", qubo["constraint_budget_ok"])
    print("  long_only:", qubo["constraint_long_only_ok"])
    print("  fully_invested:", qubo["constraint_fully_invested_ok"])
    print("  sector:", qubo["constraint_sector_ok"])
    print("Sector exposures:", qubo["sector_exposures"])

    print("\n=== Equal-Weight Benchmark ===")
    print("Assets:", prices.columns.tolist())
    print("Weights:", ew_weights.tolist())
    print("Objective score:", ew_row["objective_score"])
    print("Expected return:", ew_row["expected_return"])
    print("Variance:", ew_row["variance"])
    print("Volatility:", ew_row["volatility"])
    print("Sharpe ratio:", ew_row["sharpe_ratio"])
    print("Turnover:", ew_row["turnover"])
    print("Transaction cost:", ew_row["transaction_cost"])

    print("\n=== Comparison Summary ===")
    print("\n-- Classical vs Benchmark --")
    print("Delta expected return:", summary_row_classical["delta_expected_return"])
    print("Delta variance:", summary_row_classical["delta_variance"])
    print("Delta volatility:", summary_row_classical["delta_volatility"])
    print("Delta Sharpe ratio:", summary_row_classical["delta_sharpe_ratio"])
    print("Delta objective score:", summary_row_classical["delta_objective_score"])
    print("Classical beats benchmark on objective:", summary_row_classical["optimized_beats_benchmark_on_objective"])
    print("Classical beats benchmark on Sharpe:", summary_row_classical["optimized_beats_benchmark_on_sharpe"])

    print("\n-- QUBO vs Benchmark --")
    print("Delta expected return:", summary_row_qubo["delta_expected_return"])
    print("Delta variance:", summary_row_qubo["delta_variance"])
    print("Delta volatility:", summary_row_qubo["delta_volatility"])
    print("Delta Sharpe ratio:", summary_row_qubo["delta_sharpe_ratio"])
    print("Delta objective score:", summary_row_qubo["delta_objective_score"])
    print("QUBO beats benchmark on objective:", summary_row_qubo["optimized_beats_benchmark_on_objective"])
    print("QUBO beats benchmark on Sharpe:", summary_row_qubo["optimized_beats_benchmark_on_sharpe"])

    print("\n-- QUBO vs Classical --")
    print("Delta expected return:", summary_row_qubo_vs_classical["delta_expected_return"])
    print("Delta variance:", summary_row_qubo_vs_classical["delta_variance"])
    print("Delta volatility:", summary_row_qubo_vs_classical["delta_volatility"])
    print("Delta Sharpe ratio:", summary_row_qubo_vs_classical["delta_sharpe_ratio"])
    print("Delta objective score:", summary_row_qubo_vs_classical["delta_objective_score"])
    print("QUBO beats classical on objective:", summary_row_qubo_vs_classical["qubo_beats_classical_on_objective"])
    print("QUBO beats classical on Sharpe:", summary_row_qubo_vs_classical["qubo_beats_classical_on_sharpe"])

    print(f"\nPrices saved to {PRICES_FILE}")
    print(f"Baseline results saved to {RESULTS_FILE}")
    print(f"Comparison results saved to {COMPARISON_RESULTS_FILE}")
    print(f"Summary results saved to {SUMMARY_RESULTS_FILE}")
    print(f"Comparison chart saved to {CHART_FILE}")


if __name__ == "__main__":
    main()
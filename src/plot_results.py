import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

METRICS_FILE = OUTPUTS_DIR / "phase3_metrics.json"
WEIGHTS_FILE = OUTPUTS_DIR / "phase3_weights.csv"


def load_results():
    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    with open(WEIGHTS_FILE, "r", encoding="utf-8") as file:
        weights = list(csv.DictReader(file))

    return results, weights


def plot_metric_comparison(results):
    metrics = results["metrics"]
    benchmarks = results["benchmarks"]

    strategies = [
        "Quantum",
        "Classical",
        "Hybrid",
        "Equal weight",
        "Minimum variance",
    ]

    returns = [
        metrics["quantum_return"],
        metrics["classical_return"],
        metrics["hybrid_return"],
        benchmarks["equal_weight_return"],
        benchmarks["minimum_variance_return"],
    ]

    sharpe = [
        metrics["quantum_sharpe"],
        metrics["classical_sharpe"],
        metrics["hybrid_sharpe"],
        benchmarks["equal_weight_sharpe"],
        benchmarks["minimum_variance_sharpe"],
    ]

    x_positions = range(len(strategies))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x_positions, returns, color="#2563eb")
    axes[0].set_title("Portfolio Return Comparison")
    axes[0].set_ylabel("Return")
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels(strategies, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x_positions, sharpe, color="#16a34a")
    axes[1].set_title("Sharpe Ratio Comparison")
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].set_xticks(list(x_positions))
    axes[1].set_xticklabels(strategies, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "performance_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cvar_comparison(results):
    metrics = results["metrics"]

    strategies = [
        "Quantum",
        "Classical",
        "Hybrid",
        "Equal weight",
        "Minimum variance",
    ]

    cvar_values = [
        float(metrics["quantum_cvar"]["CVaR"]),
        float(metrics["classical_cvar"]["CVaR"]),
        float(metrics["hybrid_cvar"]["CVaR"]),
        float(metrics["equal_weight_cvar"]["CVaR"]),
        float(metrics["minimum_variance_cvar"]["CVaR"]),
    ]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(strategies, cvar_values, color="#dc2626")
    axis.set_title("CVaR Comparison")
    axis.set_ylabel("CVaR")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cvar_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_portfolio_weights(weights):
    if not weights:
        raise ValueError("No portfolio weights were found.")

    asset_names = [f"Asset {index + 1}" for index in range(len(weights))]

    strategies = [
        "quantum",
        "classical",
        "hybrid",
        "equal_weight",
        "minimum_variance",
    ]

    fig, axis = plt.subplots(figsize=(12, 6))

    x_positions = range(len(asset_names))
    width = 0.16

    for offset, strategy in enumerate(strategies):
        values = [float(row[strategy]) for row in weights]
        positions = [
            position + (offset - 2) * width
            for position in x_positions
        ]
        axis.bar(
            positions,
            values,
            width=width,
            label=strategy.replace("_", " ").title(),
        )

    axis.set_title("Portfolio Weight Comparison")
    axis.set_ylabel("Weight")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(asset_names)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "portfolio_weights.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results, weights = load_results()

    plot_metric_comparison(results)
    plot_cvar_comparison(results)
    plot_portfolio_weights(weights)

    print("Figures generated successfully.")


if __name__ == "__main__":
    main()
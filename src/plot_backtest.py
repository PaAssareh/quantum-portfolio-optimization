from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_FILE = RESULTS_DIR / "phase2_backtest_metrics.csv"
WEIGHTS_FILE = RESULTS_DIR / "phase2_backtest_weights.csv"
REPORT_FILE = RESULTS_DIR / "phase2_backtest_report.md"


def load_data():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}"
        )

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Weights file not found: {WEIGHTS_FILE}"
        )

    metrics = pd.read_csv(METRICS_FILE)
    weights = pd.read_csv(WEIGHTS_FILE)

    required_metrics = {
        "model",
        "net_annualized_return",
        "annualized_volatility",
        "sharpe",
        "cvar",
        "turnover",
        "transaction_cost",
    }

    missing_metrics = required_metrics.difference(
        metrics.columns
    )

    if missing_metrics:
        raise ValueError(
            "Missing metrics columns: "
            f"{sorted(missing_metrics)}"
        )

    required_weights = {
        "model",
        "ticker",
        "weight",
    }

    missing_weights = required_weights.difference(
        weights.columns
    )

    if missing_weights:
        raise ValueError(
            "Missing weights columns: "
            f"{sorted(missing_weights)}"
        )

    return metrics, weights


def save_bar_chart(
    values,
    title,
    ylabel,
    filename,
    color,
):
    fig, axis = plt.subplots(figsize=(10, 5))

    axis.bar(
        values.index,
        values.values,
        color=color,
    )

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_performance(metrics):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].bar(
        metrics["model"],
        metrics["net_annualized_return"],
        color="#2563eb",
    )
    axes[0].set_title(
        "Net Annualized Return"
    )
    axes[0].set_ylabel("Return")
    axes[0].tick_params(
        axis="x",
        rotation=25,
    )
    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    axes[1].bar(
        metrics["model"],
        metrics["sharpe"],
        color="#16a34a",
    )
    axes[1].set_title(
        "Sharpe Ratio"
    )
    axes[1].set_ylabel("Sharpe")
    axes[1].tick_params(
        axis="x",
        rotation=25,
    )
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "backtest_performance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_risk(metrics):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].bar(
        metrics["model"],
        metrics["annualized_volatility"],
        color="#f59e0b",
    )
    axes[0].set_title(
        "Annualized Volatility"
    )
    axes[0].set_ylabel("Volatility")
    axes[0].tick_params(
        axis="x",
        rotation=25,
    )
    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    cvar_loss = -metrics["cvar"]

    axes[1].bar(
        metrics["model"],
        cvar_loss,
        color="#dc2626",
    )
    axes[1].set_title(
        "CVaR Loss Magnitude"
    )
    axes[1].set_ylabel("-CVaR")
    axes[1].tick_params(
        axis="x",
        rotation=25,
    )
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "backtest_risk.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_costs(metrics):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].bar(
        metrics["model"],
        metrics["turnover"],
        color="#7c3aed",
    )
    axes[0].set_title(
        "Portfolio Turnover"
    )
    axes[0].set_ylabel("Turnover")
    axes[0].tick_params(
        axis="x",
        rotation=25,
    )
    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    axes[1].bar(
        metrics["model"],
        metrics["transaction_cost"],
        color="#0891b2",
    )
    axes[1].set_title(
        "Transaction Cost"
    )
    axes[1].set_ylabel("Cost")
    axes[1].tick_params(
        axis="x",
        rotation=25,
    )
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "backtest_costs.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_weights(weights):
    weight_table = weights.pivot(
        index="ticker",
        columns="model",
        values="weight",
    )

    weight_table.plot(
        kind="bar",
        figsize=(13, 6),
    )

    plt.title(
        "Backtest Portfolio Weights"
    )
    plt.xlabel("Ticker")
    plt.ylabel("Weight")
    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.legend(
        title="Model"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "backtest_weights.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def format_percent(value):
    return f"{value * 100:.2f}%"


def build_report(metrics):
    best_sharpe = metrics.loc[
        metrics["sharpe"].idxmax()
    ]

    best_return = metrics.loc[
        metrics["net_annualized_return"].idxmax()
    ]

    lowest_volatility = metrics.loc[
        metrics["annualized_volatility"].idxmin()
    ]

    lowest_cvar_loss = metrics.loc[
        metrics["cvar"].idxmax()
    ]

    lines = [
        "# Phase 2 Backtest Report",
        "",
        "## Configuration",
        "",
        f"- Training ratio: "
        f"{metrics['train_ratio'].iloc[0]:.2f}",
        f"- QAOA depth: "
        f"{int(metrics['qaoa_p'].iloc[0])}",
        f"- QAOA maxiter: "
        f"{int(metrics['qaoa_maxiter'].iloc[0])}",
        "",
        "## Model comparison",
        "",
        "| Model | Net return | Volatility | "
        "Sharpe | CVaR | Turnover | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {format_percent(row['net_annualized_return'])} "
            f"| {format_percent(row['annualized_volatility'])} "
            f"| {row['sharpe']:.4f} "
            f"| {format_percent(row['cvar'])} "
            f"| {row['turnover']:.4f} "
            f"| {row['transaction_cost']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best results",
            "",
            f"- Best Sharpe: "
            f"{best_sharpe['model']} "
            f"({best_sharpe['sharpe']:.4f})",
            f"- Highest net annualized return: "
            f"{best_return['model']} "
            f"({format_percent(best_return['net_annualized_return'])})",
            f"- Lowest volatility: "
            f"{lowest_volatility['model']} "
            f"({format_percent(lowest_volatility['annualized_volatility'])})",
            f"- Lowest CVaR loss magnitude: "
            f"{lowest_cvar_loss['model']} "
            f"({format_percent(lowest_cvar_loss['cvar'])})",
            "",
            "## Figures",
            "",
            "![Performance](figures/backtest_performance.png)",
            "",
            "![Risk](figures/backtest_risk.png)",
            "",
            "![Costs](figures/backtest_costs.png)",
            "",
            "![Weights](figures/backtest_weights.png)",
            "",
            "## Interpretation",
            "",
            "The results are based on a single 70/30 "
            "train-test split and a fast QAOA configuration. "
            "They should be treated as an out-of-sample "
            "pipeline check, not as the final performance claim.",
            "",
            "The Hybrid strategy differs from the Quantum "
            "portfolio and provides a separate risk-adjusted "
            "allocation for comparison.",
        ]
    )

    return "\n".join(lines)


def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics, weights = load_data()

    plot_performance(metrics)
    plot_risk(metrics)
    plot_costs(metrics)
    plot_weights(weights)

    report = build_report(metrics)
    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print("Backtest figures generated successfully.")
    print(f"Saved report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
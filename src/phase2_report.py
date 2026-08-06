from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
METRICS_FILE = RESULTS_DIR / "phase2_backtest_metrics.csv"
REPORT_FILE = RESULTS_DIR / "phase2_report.txt"


def format_percent(value):
    return f"{value * 100:.2f}%"


def build_report(metrics):
    best_sharpe_row = metrics.loc[metrics["test_sharpe"].idxmax()]
    best_return_row = metrics.loc[
        metrics["test_annualized_return"].idxmax()
    ]
    lowest_risk_row = metrics.loc[
        metrics["test_annualized_volatility"].idxmin()
    ]

    lines = [
        "Phase 2 Out-of-Sample Backtest Report",
        "=" * 44,
        "",
        "Dataset split:",
        f"- Training observations: {int(metrics['train_observations'].iloc[0])}",
        f"- Test observations: {int(metrics['test_observations'].iloc[0])}",
        "",
        "Model comparison:",
        "",
        "Model                  Return       Volatility   Sharpe       Max Drawdown   CVaR",
        "-" * 82,
    ]

    for _, row in metrics.iterrows():
        lines.append(
            f"{row['model']:<22}"
            f"{format_percent(row['test_annualized_return']):>10}"
            f"{format_percent(row['test_annualized_volatility']):>14}"
            f"{row['test_sharpe']:>10.4f}"
            f"{format_percent(row['test_max_drawdown']):>15}"
            f"{format_percent(row['test_cvar']):>10}"
        )

    lines.extend(
        [
            "",
            "Best results:",
            f"- Best Sharpe: {best_sharpe_row['model']} "
            f"({best_sharpe_row['test_sharpe']:.4f})",
            f"- Highest annualized return: {best_return_row['model']} "
            f"({format_percent(best_return_row['test_annualized_return'])})",
            f"- Lowest volatility: {lowest_risk_row['model']} "
            f"({format_percent(lowest_risk_row['test_annualized_volatility'])})",
            "",
            "Interpretation:",
            (
                "The out-of-sample results show that the quantum portfolio "
                "should be evaluated against classical and equal-weight "
                "baselines rather than assumed to be superior."
            ),
            (
                "The best model is selected using test Sharpe ratio, while "
                "return, volatility, drawdown and CVaR are reported as "
                "additional risk-performance measures."
            ),
            "",
            "Reproducibility:",
            "- QAOA was optimized on the training period only.",
            "- Portfolio performance was evaluated on the unseen test period.",
            "- The processed market data is pinned in data/processed/adj_close.csv.",
        ]
    )

    return "\n".join(lines)


def main():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}"
        )

    metrics = pd.read_csv(METRICS_FILE)

    required_columns = {
        "model",
        "train_observations",
        "test_observations",
        "test_annualized_return",
        "test_annualized_volatility",
        "test_sharpe",
        "test_max_drawdown",
        "test_cvar",
    }

    missing_columns = required_columns.difference(metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in metrics file: {sorted(missing_columns)}"
        )

    report = build_report(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
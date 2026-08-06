import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from portfolio_math import PortfolioMath
from qubo_optimizer import QUBOOptimizer
from classical_baseline import ClassicalBaseline


RESULTS_DIR = Path("results")
METRICS_FILE = RESULTS_DIR / "phase2_backtest_metrics.csv"


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)

    total = weights.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)

    return weights / total


def calculate_metrics(test_prices, weights, alpha=0.95):
    weights = normalize_weights(weights)

    test_returns = test_prices.pct_change().dropna()
    portfolio_returns = test_returns.dot(weights).dropna()

    if portfolio_returns.empty:
        raise ValueError("Test period does not contain enough return observations.")

    cumulative = (1.0 + portfolio_returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)

    periods = len(portfolio_returns)
    annualized_return = float((1.0 + total_return) ** (252 / periods) - 1.0)
    annualized_volatility = float(portfolio_returns.std() * np.sqrt(252))

    sharpe = 0.0
    if annualized_volatility > 0:
        sharpe = annualized_return / annualized_volatility

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min())

    var = float(np.quantile(portfolio_returns, 1.0 - alpha))
    tail_losses = portfolio_returns[portfolio_returns <= var]

    if len(tail_losses) > 0:
        cvar = float(tail_losses.mean())
    else:
        cvar = var

    return {
        "test_total_return": total_return,
        "test_annualized_return": annualized_return,
        "test_annualized_volatility": annualized_volatility,
        "test_sharpe": float(sharpe),
        "test_max_drawdown": max_drawdown,
        "test_var": var,
        "test_cvar": cvar,
    }


def optimize_on_training(train_prices):
    feature_engineering = FeatureEngineering(train_prices)
    feature_engineering.build_feature_matrix()

    portfolio = PortfolioMath(train_prices)

    qubo = QUBOOptimizer(
        portfolio,
        risk_factor=Config.RISK_FACTOR,
        budget=Config.BUDGET,
        penalty=Config.PENALTY,
    )

    print("Running QAOA on training data...")
    qubo.optimize_portfolio(
        p=Config.QAOA_P,
        maxiter=Config.QAOA_MAXITER,
    )
    quantum_weights = qubo.get_portfolio_weights()

    classical = ClassicalBaseline(portfolio)

    print("Computing classical training portfolios...")
    classical_weights = classical.optimize_portfolio(method="mean_variance")
    equal_weights = classical.optimize_portfolio(method="equal_weight")
    minimum_variance_weights = classical.optimize_portfolio(
        method="minimum_variance"
    )

    return {
        "quantum": normalize_weights(quantum_weights),
        "classical": normalize_weights(classical_weights),
        "equal_weight": normalize_weights(equal_weights),
        "minimum_variance": normalize_weights(minimum_variance_weights),
    }


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading market data...")

    loader = DataLoader(
        tickers=Config.TICKERS,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        source=Config.DATA_SOURCE,
    )

    prices = loader.load_market_data()
    prices = prices.dropna(how="all").dropna(axis=1, how="all")

    split_index = int(len(prices) * 0.70)

    train_prices = prices.iloc[:split_index].copy()
    test_prices = prices.iloc[split_index:].copy()

    print(f"Full data shape: {prices.shape}")
    print(f"Training data: {train_prices.shape}")
    print(f"Test data: {test_prices.shape}")

    weights_by_model = optimize_on_training(train_prices)

    rows = []

    for model_name, weights in weights_by_model.items():
        metrics = calculate_metrics(
            test_prices,
            weights,
            alpha=Config.CVaR_ALPHA,
        )

        row = {
            "model": model_name,
            "train_observations": len(train_prices),
            "test_observations": len(test_prices),
            **metrics,
        }

        rows.append(row)

        print(
            f"{model_name}: "
            f"return={metrics['test_annualized_return']:.6f}, "
            f"volatility={metrics['test_annualized_volatility']:.6f}, "
            f"sharpe={metrics['test_sharpe']:.6f}, "
            f"cvar={metrics['test_cvar']:.6f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(METRICS_FILE, index=False)

    print(f"Saved backtest metrics to {METRICS_FILE}")
    print(f"Finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
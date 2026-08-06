import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from classical_baseline import ClassicalBaseline
from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from portfolio_math import PortfolioMath
from qubo_optimizer import QUBOOptimizer


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

METRICS_FILE = RESULTS_DIR / "phase2_backtest_metrics.csv"
WEIGHTS_FILE = RESULTS_DIR / "phase2_backtest_weights.csv"


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)

    weights = np.nan_to_num(
        weights,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    weights = np.maximum(weights, 0.0)
    total = weights.sum()

    if total <= 0:
        return np.ones(
            len(weights),
            dtype=float,
        ) / len(weights)

    return weights / total


def validate_current_weights(weights, expected_length):
    weights = normalize_weights(weights)

    if len(weights) != expected_length:
        raise ValueError(
            "Portfolio weights and price columns have different lengths."
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Portfolio weights contain invalid values."
        )

    if np.any(weights < 0):
        raise ValueError(
            "Portfolio weights cannot be negative."
        )

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "Portfolio weights must sum to one."
        )

    return weights


def validate_previous_weights(
    previous_weights,
    expected_length,
):
    previous_weights = np.asarray(
        previous_weights,
        dtype=float,
    )

    if len(previous_weights) != expected_length:
        raise ValueError(
            "Previous weights and price columns have different lengths."
        )

    if not np.isfinite(previous_weights).all():
        raise ValueError(
            "Previous portfolio weights contain invalid values."
        )

    if np.any(previous_weights < 0):
        raise ValueError(
            "Previous portfolio weights cannot be negative."
        )

    total = previous_weights.sum()

    if total > 0:
        previous_weights = previous_weights / total

    return previous_weights


def calculate_turnover(
    weights,
    previous_weights,
):
    return float(
        np.abs(weights - previous_weights).sum()
    )


def calculate_metrics(
    test_prices,
    weights,
    previous_weights,
    transaction_cost,
    alpha=0.95,
):
    expected_length = len(test_prices.columns)

    weights = validate_current_weights(
        weights,
        expected_length,
    )

    previous_weights = validate_previous_weights(
        previous_weights,
        expected_length,
    )

    test_returns = test_prices.pct_change().dropna()

    gross_portfolio_returns = test_returns.dot(
        weights
    ).dropna()

    if gross_portfolio_returns.empty:
        raise ValueError(
            "Test period does not contain enough return observations."
        )

    turnover = calculate_turnover(
        weights,
        previous_weights,
    )

    trading_cost = turnover * transaction_cost

    net_portfolio_returns = gross_portfolio_returns.copy()

    net_portfolio_returns.iloc[0] -= trading_cost

    gross_cumulative = (
        1.0 + gross_portfolio_returns
    ).cumprod()

    net_cumulative = (
        1.0 + net_portfolio_returns
    ).cumprod()

    gross_total_return = float(
        gross_cumulative.iloc[-1] - 1.0
    )

    net_total_return = float(
        net_cumulative.iloc[-1] - 1.0
    )

    periods = len(gross_portfolio_returns)

    gross_annualized_return = float(
        (1.0 + gross_total_return)
        ** (252 / periods)
        - 1.0
    )

    net_annualized_return = float(
        (1.0 + net_total_return)
        ** (252 / periods)
        - 1.0
    )

    annualized_volatility = float(
        gross_portfolio_returns.std()
        * np.sqrt(252)
    )

    if annualized_volatility > 0:
        sharpe = (
            net_annualized_return
            / annualized_volatility
        )
    else:
        sharpe = 0.0

    running_max = net_cumulative.cummax()

    drawdown = (
        net_cumulative / running_max
    ) - 1.0

    max_drawdown = float(
        drawdown.min()
    )

    var = float(
        np.quantile(
            net_portfolio_returns,
            1.0 - alpha,
        )
    )

    tail_losses = net_portfolio_returns[
        net_portfolio_returns <= var
    ]

    if len(tail_losses) > 0:
        cvar = float(
            tail_losses.mean()
        )
    else:
        cvar = var

    return {
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "gross_annualized_return": (
            gross_annualized_return
        ),
        "net_annualized_return": (
            net_annualized_return
        ),
        "annualized_volatility": (
            annualized_volatility
        ),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "var": var,
        "cvar": cvar,
        "turnover": turnover,
        "transaction_cost": trading_cost,
    }


def build_hybrid_weights(
    quantum_weights,
    minimum_variance_weights,
    train_prices,
):
    train_returns = train_prices.pct_change().dropna()

    asset_volatility = train_returns.std()
    market_volatility = float(
        train_returns.mean(axis=1).std()
    )
    typical_asset_volatility = float(
        asset_volatility.median()
    )

    if typical_asset_volatility <= 0:
        defensive_factor = 0.0
    else:
        defensive_factor = (
            market_volatility
            / typical_asset_volatility
        )
        defensive_factor = float(
            np.clip(
                defensive_factor - 1.0,
                0.0,
                1.0,
            )
        )

    minimum_variance_blend = (
        0.25
        + 0.25 * defensive_factor
    )

    hybrid_weights = (
        (1.0 - minimum_variance_blend)
        * quantum_weights
        + minimum_variance_blend
        * minimum_variance_weights
    )

    return normalize_weights(
        hybrid_weights
    )

def optimize_on_training(
    train_prices,
    qaoa_maxiter,
):
    feature_engineering = FeatureEngineering(
        train_prices
    )

    feature_engineering.build_feature_matrix()

    portfolio = PortfolioMath(
        train_prices
    )

    qubo = QUBOOptimizer(
        portfolio,
        risk_factor=Config.RISK_FACTOR,
        budget=Config.BUDGET,
        penalty=Config.PENALTY,
    )

    print("Running QAOA on training data...")

    qubo.optimize_portfolio(
        p=Config.QAOA_P,
        maxiter=qaoa_maxiter,
    )

    quantum_weights = normalize_weights(
        qubo.get_portfolio_weights()
    )

    classical = ClassicalBaseline(
        portfolio
    )

    print(
        "Computing classical training portfolios..."
    )

    classical_weights = normalize_weights(
        classical.optimize_portfolio(
            method="mean_variance"
        )
    )

    equal_weights = normalize_weights(
        classical.optimize_portfolio(
            method="equal_weight"
        )
    )

    minimum_variance_weights = normalize_weights(
        classical.optimize_portfolio(
            method="minimum_variance"
        )
    )

    hybrid_weights = build_hybrid_weights(
        quantum_weights,
        minimum_variance_weights,
        train_prices,
    )

    return {
        "quantum": quantum_weights,
        "classical": classical_weights,
        "hybrid": hybrid_weights,
        "equal_weight": equal_weights,
        "minimum_variance": (
            minimum_variance_weights
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the out-of-sample "
            "portfolio backtest."
        )
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced QAOA iterations.",
    )

    parser.add_argument(
        "--fast-iter",
        type=int,
        default=5,
        help=(
            "Number of QAOA iterations "
            "in fast mode."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help=(
            "Fraction of data used "
            "for training."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(
            "--train-ratio must be between 0 and 1."
        )

    if args.fast_iter < 1:
        raise ValueError(
            "--fast-iter must be at least 1."
        )

    start_time = time.time()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fast:
        qaoa_maxiter = args.fast_iter
    else:
        qaoa_maxiter = Config.QAOA_MAXITER

    print(
        f"QAOA maxiter: {qaoa_maxiter}"
    )

    print(
        f"Training ratio: {args.train_ratio:.2f}"
    )

    print("Loading market data...")

    loader = DataLoader(
        tickers=Config.TICKERS,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        source=Config.DATA_SOURCE,
    )

    prices = loader.load_market_data()

    prices = prices.dropna(
        how="all"
    ).dropna(
        axis=1,
        how="all",
    )

    if prices.empty or len(prices.columns) == 0:
        raise ValueError(
            "No usable market-price data was loaded."
        )

    split_index = int(
        len(prices)
        * args.train_ratio
    )

    if split_index < 2:
        raise ValueError(
            "Training period must contain "
            "at least two rows."
        )

    if len(prices) - split_index < 2:
        raise ValueError(
            "Test period must contain "
            "at least two rows."
        )

    train_prices = prices.iloc[
        :split_index
    ].copy()

    test_prices = prices.iloc[
        split_index:
    ].copy()

    print(
        f"Full data shape: {prices.shape}"
    )

    print(
        f"Training data: {train_prices.shape}"
    )

    print(
        f"Test data: {test_prices.shape}"
    )

    weights_by_model = optimize_on_training(
        train_prices,
        qaoa_maxiter=qaoa_maxiter,
    )

    previous_weights = np.zeros(
        len(test_prices.columns),
        dtype=float,
    )

    metrics_rows = []
    weight_rows = []

    for model_name, weights in (
        weights_by_model.items()
    ):
        metrics = calculate_metrics(
            test_prices=test_prices,
            weights=weights,
            previous_weights=previous_weights,
            transaction_cost=(
                Config.TRANSACTION_COST
            ),
            alpha=Config.CVaR_ALPHA,
        )

        metrics_row = {
            "model": model_name,
            "train_observations": (
                len(train_prices)
            ),
            "test_observations": (
                len(test_prices)
            ),
            "train_ratio": args.train_ratio,
            "qaoa_p": Config.QAOA_P,
            "qaoa_maxiter": qaoa_maxiter,
            **metrics,
        }

        metrics_rows.append(
            metrics_row
        )

        for ticker, weight in zip(
            test_prices.columns,
            weights,
        ):
            weight_rows.append(
                {
                    "model": model_name,
                    "ticker": ticker,
                    "weight": float(weight),
                }
            )

        print(
            f"{model_name}: "
            f"net_return="
            f"{metrics['net_annualized_return']:.6f}, "
            f"volatility="
            f"{metrics['annualized_volatility']:.6f}, "
            f"sharpe="
            f"{metrics['sharpe']:.6f}, "
            f"cvar="
            f"{metrics['cvar']:.6f}, "
            f"turnover="
            f"{metrics['turnover']:.6f}, "
            f"cost="
            f"{metrics['transaction_cost']:.6f}"
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    weights_df = pd.DataFrame(
        weight_rows
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    weights_df.to_csv(
        WEIGHTS_FILE,
        index=False,
    )

    elapsed = time.time() - start_time

    print(
        f"Saved backtest metrics to "
        f"{METRICS_FILE}"
    )

    print(
        f"Saved backtest weights to "
        f"{WEIGHTS_FILE}"
    )

    print(
        f"Finished in {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
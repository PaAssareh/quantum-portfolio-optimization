import warnings
import time
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from portfolio_math import PortfolioMath
from qubo_optimizer import QUBOOptimizer
from classical_baseline import ClassicalBaseline
from regime_detector import RegimeDetector
from regime_rebalance import RegimeRebalance
from cvar_analysis import CVaRAnalysis
from portfolio_copilot import PortfolioCoPilot
from datetime import datetime

from scipy.sparse import SparseEfficiencyWarning

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from portfolio_math import PortfolioMath
from qubo_optimizer import QUBOOptimizer
from classical_baseline import ClassicalBaseline
from regime_detector import RegimeDetector
from regime_rebalance import RegimeRebalance
from cvar_analysis import CVaRAnalysis
from portfolio_copilot import PortfolioCoPilot

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def main():
    start_time = time.time()

    def log_step(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    log_step("Starting portfolio pipeline")

    log_step("Loading market data")
    loader = DataLoader(
        tickers=Config.TICKERS,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        source=Config.DATA_SOURCE
    )
    prices = loader.load_market_data()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Market data loaded: {getattr(prices, 'shape', 'n/a')}")
    loader.save_processed_data(prices)

    log_step("Building feature matrix")
    features = FeatureEngineering(prices)
    feature_df = features.build_feature_matrix()
    returns_df = features.returns
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature matrix built")

    log_step("Initializing portfolio math")
    portfolio = PortfolioMath(prices)

    log_step("Running QUBO optimization")
    qubo = QUBOOptimizer(
        portfolio,
        risk_factor=Config.RISK_FACTOR,
        budget=Config.BUDGET,
        penalty=Config.PENALTY
    )
    qubo.optimize_portfolio(p=Config.QAOA_P, maxiter=Config.QAOA_MAXITER)
    quantum_weights = qubo.get_portfolio_weights()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] QUBO optimization complete")

    log_step("Computing classical baselines")
    classical = ClassicalBaseline(portfolio)
    classical_weights = classical.optimize_portfolio(method="mean_variance")
    equal_weights = classical.optimize_portfolio(method="equal_weight")
    minimum_variance_weights = classical.optimize_portfolio(method="minimum_variance")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Classical baselines complete")

    log_step("Detecting market regimes")
    detector = RegimeDetector(
        feature_df,
        model=Config.REGIME_MODEL,
        states=Config.REGIME_STATES,
        threshold=Config.REGIME_THRESHOLD
    )
    regimes = detector.detect_regimes()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Regime detection complete")

    log_step("Rebalancing portfolio")
    rebalance = RegimeRebalance(portfolio, quantum_weights, regimes)
    hybrid_weights = rebalance.rebalance_portfolio()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rebalancing complete")

    log_step("Comparing portfolio results")
    comparison = portfolio.compare_results(quantum_weights, classical_weights, hybrid_weights)
    benchmark_equal = portfolio.compare_results(equal_weights, equal_weights)
    benchmark_minvar = portfolio.compare_results(minimum_variance_weights, minimum_variance_weights)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Comparison complete")

    log_step("Evaluating CVaR metrics")
    cvar = CVaRAnalysis(portfolio, alpha=Config.CVaR_ALPHA)
    quantum_cvar = cvar.evaluate_strategy(quantum_weights, returns_df)
    classical_cvar = cvar.evaluate_strategy(classical_weights, returns_df)
    hybrid_cvar = cvar.evaluate_strategy(hybrid_weights, returns_df)
    equal_cvar = cvar.evaluate_strategy(equal_weights, returns_df)
    minvar_cvar = cvar.evaluate_strategy(minimum_variance_weights, returns_df)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CVaR evaluation complete")

    metrics = {
        **comparison,
        "quantum_cvar": quantum_cvar,
        "classical_cvar": classical_cvar,
        "hybrid_cvar": hybrid_cvar,
        "equal_weight_cvar": equal_cvar,
        "minimum_variance_cvar": minvar_cvar,
        "current_regime": int(regimes[-1]) if len(regimes) else None
    }

    benchmarks = {
        "equal_weight_return": benchmark_equal["quantum_return"],
        "equal_weight_sharpe": benchmark_equal["quantum_sharpe"],
        "minimum_variance_return": benchmark_minvar["quantum_return"],
        "minimum_variance_sharpe": benchmark_minvar["quantum_sharpe"]
    }
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    def make_json_serializable(value):
        if isinstance(value, dict):
            return {
                key: make_json_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                make_json_serializable(item)
                for item in value
            ]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        return value

    results_payload = {
        "configuration": {
            "qaoa_p": Config.QAOA_P,
            "qaoa_optimizer": Config.QAOA_OPTIMIZER,
            "qaoa_maxiter": Config.QAOA_MAXITER,
            "risk_factor": Config.RISK_FACTOR,
            "budget": Config.BUDGET,
            "cvar_alpha": Config.CVaR_ALPHA,
        },
        "metrics": make_json_serializable(metrics),
        "benchmarks": make_json_serializable(benchmarks),
        "weights": {
            "quantum": np.asarray(
                quantum_weights,
                dtype=float,
            ).tolist(),
            "classical": np.asarray(
                classical_weights,
                dtype=float,
            ).tolist(),
            "hybrid": np.asarray(
                hybrid_weights,
                dtype=float,
            ).tolist(),
            "equal_weight": np.asarray(
                equal_weights,
                dtype=float,
            ).tolist(),
            "minimum_variance": np.asarray(
                minimum_variance_weights,
                dtype=float,
            ).tolist(),
        },
    }

    with open(
        outputs_dir / "phase3_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    np.savetxt(
        outputs_dir / "phase3_weights.csv",
        np.column_stack(
            [
                quantum_weights,
                classical_weights,
                hybrid_weights,
                equal_weights,
                minimum_variance_weights,
            ]
        ),
        delimiter=",",
        header=(
            "quantum,classical,hybrid,"
            "equal_weight,minimum_variance"
        ),
        comments="",
    )
    log_step("Generating portfolio report")
    copilot = PortfolioCoPilot(
        portfolio,
        quantum_weights=quantum_weights,
        classical_weights=classical_weights,
        hybrid_weights=hybrid_weights,
        regime_info=metrics["current_regime"],
        metrics=metrics,
        benchmarks=benchmarks
    )
    print(copilot.generate_report())

    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
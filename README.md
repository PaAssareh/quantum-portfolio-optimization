# quantum-portfolio-optimization
# WISER-Q: Hybrid Quantum-Classical Portfolio Optimization

WISER-Q is a portfolio optimization project that compares a classical benchmark against a quantum-inspired QUBO approach on real market data. The current repository focuses on **Phase 1** of the project: a small universe of real assets, realistic constraints, and clear performance comparison. **Phase 2** — simulated-data sensitivity and scalability analysis — is planned next.

## Project Goal

The goal is to build an optimization pipeline that balances:
- expected return,
- risk,
- turnover,
- transaction costs,
- and sector constraints.

We compare three portfolio construction approaches:
1. **Classical Baseline**: brute-force cardinality-constrained optimization.
2. **Static QUBO+**: a quantum-inspired formulation with penalty terms for return, risk, cardinality, transaction cost, and sector constraints.
3. **Regime-Aware QUBO+**: a dynamic version that detects market regime and applies partial rebalancing to reduce friction.

## Why This Problem Matters

Portfolio optimization is a practical finance problem where small changes in the objective function or constraints can change the final allocation. In real settings, a model is not useful if it ignores implementation frictions such as transaction costs or sector limits. This project is designed to show that quantum-inspired optimization can be made more realistic, interpretable, and competition-ready.

## Phase 1 Setup

Phase 1 uses a small, real-world asset universe:
- **10 U.S. large-cap stocks**
- **Daily adjusted close prices**
- **2020-01-01 to 2025-01-01**
- **Budget / cardinality constraint**: 4 assets selected
- **Sector constraints**: enforced through hard checks and QUBO penalties

The current experiments use:
- `AAPL`, `AMZN`, `GOOGL`, `JNJ`, `JPM`, `KO`, `MSFT`, `NVDA`, `PG`, `XOM`

This is a good Phase 1 dataset because it is small enough for exact classical benchmarking, but still realistic enough to demonstrate meaningful portfolio behavior.

## Methods

### Classical Baseline
The classical model performs brute-force search over all valid 4-asset portfolios and keeps only allocations that satisfy sector constraints. The selected portfolio is evaluated using:
- expected return,
- variance,
- volatility,
- Sharpe ratio,
- turnover,
- transaction cost.

### Static QUBO+
The QUBO model encodes portfolio selection as a binary optimization problem. The objective includes:
- return reward,
- risk penalty,
- cardinality penalty,
- transaction-cost-aware turnover penalty,
- sector penalty.

This version is the strongest **performance-focused** method in the current project.

### Regime-Aware QUBO+
A market regime detector uses rolling return and rolling volatility as a simple market-state proxy. The regime-aware branch then:
- changes QUBO penalty parameters based on the detected regime,
- applies partial rebalancing instead of fully replacing the old portfolio,
- reduces trading friction in more defensive conditions.

This version is the most **adaptive** and realistic method in the project.

## Results So Far

On the current dataset and parameter settings:
- **Static QUBO+** achieves the best Sharpe ratio among the tested methods.
- **Classical Baseline** is strong on the combined objective score.
- **Regime-Aware QUBO+** lowers turnover and transaction cost, but is currently more conservative and needs further tuning.

The important result is that the project already demonstrates a meaningful trade-off between:
- raw performance,
- risk-adjusted performance,
- and implementation frictions.

## Phase 2 Roadmap

Phase 2 will extend the project with simulated data to test:
- sensitivity to parameter changes,
- scalability as the number of assets increases,
- robustness under different volatility regimes,
- and the effect of transaction-cost assumptions.

Planned Phase 2 additions:
- synthetic market generator,
- larger asset universes,
- sensitivity sweeps over `lambda_risk`, `P_turn`, `P_turn_quad`, and sector penalties,
- runtime comparison between classical and QUBO variants,
- more detailed robustness analysis.

## Repository Structure

```text
wiser/
├── src/
│   ├── main.py
│   ├── classical_baseline.py
│   ├── config.py
│   ├── data_loader.py
│   ├── portfolio_math.py
│   ├── qubo_integration.py
│   ├── qubo_optimizer.py
│   ├── regime_detector.py
│   └── regime_rebalance.py
├── data/
│   └── processed/
├── README.md
└── requirements.txt
```

## How to Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python src/main.py
```

The script will:
- download or load price data,
- compute returns and covariance,
- run the classical baseline,
- run the static QUBO+
- run the regime-aware QUBO+
- save results in `data/processed/`,
- and generate a comparison chart.

## Main Outputs

The pipeline saves:
- `data/processed/adj_close.csv`
- `data/processed/baseline_results.csv`
- `data/processed/comparison_results.csv`
- `data/processed/summary_results.csv`
- `data/processed/comparison_chart.png`

## Limitations

- Phase 1 uses a small asset universe, so scalability is not yet demonstrated.
- The regime detector is deliberately simple and rule-based.
- The regime-aware version currently improves turnover and friction, but still needs parameter tuning to outperform the static QUBO on every metric.
- The QUBO solver is simulation-based, not hardware-executed.

## Future Work

- Add Phase 2 simulation experiments.
- Increase asset universe size.
- Add sensitivity analysis and runtime scaling plots.
- Improve regime detection with a more advanced model.
- Explore hardware or hybrid quantum backends.

## AI Tools Used

Generative AI tools were used for:
- brainstorming project structure,
- editing and improving README wording,
- code assistance for QUBO and regime-aware pipeline integration,
- and helping prepare a clearer competition-oriented narrative.

## References / Credits

Primary references include:
- WISER challenge materials
- QUBO and portfolio optimization literature
- transaction-cost-aware portfolio optimization research
- regime-switching portfolio allocation research
- the `yfinance` data source used for market data access

All code, data, and external sources should be credited appropriately in the final submission package.
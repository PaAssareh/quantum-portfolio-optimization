\# Phase 3 Results — Quantum Portfolio Optimization Engine



\## Overview



This report presents the results of the Phase 3 portfolio optimization pipeline after correcting the QAOA portfolio-weight representation.



The main correction converts the binary asset-selection vector returned by QAOA into a fully invested portfolio allocation. For a budget of three selected assets, each selected asset receives a weight of approximately 0.3333 instead of a binary weight of 1.0.



\## Configuration



\- QAOA repetitions: 2

\- Classical optimizer: COBYLA

\- Maximum optimizer iterations: 50

\- Risk factor: configured project value

\- Portfolio budget: 3 selected assets

\- Number of assets: 10

\- Market-data shape: 1,509 observations × 10 assets

\- Current detected market regime: 1

\- Total runtime: 2,488.54 seconds



\## Portfolio Allocations



\### Quantum allocation



```text

\\\[0.33333333, 0.00000000, 0.00000000,

\&#x20;0.33333333, 0.00000000, 0.33333333,

\&#x20;0.00000000, 0.00000000, 0.00000000,

\&#x20;0.00000000]

```



\### Classical allocation



```text

\\\[0.37291344, 0.00000000, 0.00000000,

\&#x20;0.02679578, 0.00000000, 0.18252150,

\&#x20;0.05250893, 0.00000000, 0.11690615,

\&#x20;0.24835421]

```



\### Hybrid allocation



```text

\\\[0.33333333, 0.00000000, 0.00000000,

\&#x20;0.33333333, 0.00000000, 0.33333333,

\&#x20;0.00000000, 0.00000000, 0.00000000,

\&#x20;0.00000000]

```



The quantum and hybrid allocations were identical in this run.



\## Performance Results



| Strategy | Return | Volatility | Sharpe Ratio | CVaR |

|---|---:|---:|---:|---:|

| Quantum | 0.001220 | 0.017117 | 0.071248 | 0.039279 |

| Classical | 0.001095 | 0.014641 | 0.074762 | 0.034841 |

| Hybrid | 0.001220 | 0.017117 | 0.071248 | 0.039279 |

| Equal Weight | 0.000802 | — | 0.062649 | 0.030709 |

| Minimum Variance | 0.000458 | — | 0.042632 | 0.025238 |



\## Risk Results



| Strategy | VaR | CVaR |

|---|---:|---:|

| Quantum | 0.025948 | 0.039279 |

| Classical | 0.020763 | 0.034841 |

| Hybrid | 0.025948 | 0.039279 |

| Equal Weight | 0.017274 | 0.030709 |

| Minimum Variance | 0.014764 | 0.025238 |



\## Interpretation



The quantum portfolio achieved a slightly higher return than the classical baseline:



```text

Quantum return:   0.001220

Classical return: 0.001095

```



However, the classical portfolio achieved a slightly higher Sharpe ratio:



```text

Quantum Sharpe:   0.071248

Classical Sharpe: 0.074762

```



The quantum portfolio produced a higher CVaR than the classical, equal-weight, and minimum-variance strategies. This indicates a higher level of estimated tail risk.



The hybrid allocation preserved the quantum allocation in this run and therefore produced identical return, Sharpe, volatility, VaR, and CVaR metrics.



These results do not demonstrate definitive quantum advantage. Instead, they show the trade-off between return, diversification, and downside risk and provide a transparent comparison with classical portfolio methods.



\## Main Findings



\- QAOA selected a sparse portfolio containing three assets.

\- The corrected allocation is fully invested and sums to 1.0.

\- The quantum strategy achieved higher return than the classical baseline.

\- The classical strategy achieved a slightly higher Sharpe ratio.

\- Minimum-variance optimization produced the lowest CVaR.

\- Hybrid post-processing did not change the quantum allocation in this run.

\- Statevector-based QAOA simulation remains computationally expensive.



\## Limitations



The results depend on the selected historical data period, risk-factor configuration, QAOA parameters, optimizer settings, and asset universe.



The QAOA calculation was performed using classical statevector simulation rather than fault-tolerant quantum hardware. The computational cost may increase significantly as the number of assets or optimization iterations grows.



The results are for research and demonstration purposes only and do not constitute financial advice.



\## Reproducibility



Activate the project virtual environment and run:



```powershell

python src/main.py

```



The main pipeline performs:



1\. Market-data loading.

2\. Feature engineering.

3\. Portfolio construction.

4\. QAOA-based QUBO optimization.

5\. Classical baseline optimization.

6\. Market-regime detection.

7\. Regime-aware rebalancing.

8\. Portfolio comparison.

9\. CVaR analysis.

10\. Portfolio Co-Pilot report generation.



\## Conclusion



The Phase 3 implementation successfully corrected the QAOA weight representation and produced a fully invested portfolio allocation. The current results show that the quantum strategy can achieve a slightly higher return than the classical baseline, but the classical strategy provides a slightly better Sharpe ratio and lower tail risk.



The next optimization steps should focus on reducing QAOA runtime, improving the objective formulation, and testing robustness across different market periods and regimes.

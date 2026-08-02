import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def fetch_market_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    mu = returns.mean().values
    sigma = returns.cov().values
    return mu, sigma, tickers

class QuantumPortfolioOptimizer:
    def __init__(self, mu, sigma, risk_factor=0.5, budget=1, penalty=1):
        self.mu = mu
        self.sigma = sigma
        self.risk_factor = risk_factor
        self.budget = budget
        self.penalty = penalty
        self.qp = self._build_quadratic_program()

    def _build_quadratic_program(self):
        num_assets = len(self.mu)
        qp = QuadraticProgram()
        qp.binary_var_list(list(range(num_assets)))
        
        # Objective function
        qp.minimize(linear=self.mu, quadratic=self.risk_factor * self.sigma)
        
        # Budget constraint
        qp.linear_constraint(linear=[1] * num_assets, sense='<=', rhs=self.budget, name='budget')
        
        return qp
    
    def solve(self, quantum_instance=None, classical_optimizer=COBYLA(maxiter=100)):
        qaoa_mes = MinimumEigenOptimizer(min_eigen_solver=QAOA(quantum_instance=quantum_instance, optimizer=classical_optimizer))
        qaoa_result = qaoa_mes.solve(self.qp)
        
        return qaoa_result.x, qaoa_result.fval
    
def classical_portfolio_optimization(mu, sigma, risk_factor, budget):
    num_assets = len(mu)
    qp = QuadraticProgram()
    qp.binary_var_list(list(range(num_assets)))
    qp.minimize(linear=mu, quadratic=risk_factor * sigma)
    qp.linear_constraint(linear=[1] * num_assets, sense='<=', rhs=budget, name='budget')
    
    exact_mes = MinimumEigenOptimizer(min_eigen_solver=NumPyMinimumEigensolver())
    exact_result = exact_mes.solve(qp)
    
    return exact_result.x, exact_result.fval

if __name__ == "__main__":
    # Fetch market data
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "NFLX", "XOM", "CVX", "PFE", "UNH"]
    start_date = "2015-01-01"
    end_date = "2023-06-21"
    mu, sigma, asset_names = fetch_market_data(tickers, start_date, end_date)

    # Set up the optimizer
    risk_factor = 0.5
    budget = 3
    penalty = 1
    optimizer = QuantumPortfolioOptimizer(mu, sigma, risk_factor, budget, penalty)

    # Solve with QAOA
    backend = Sampler()
    quantum_solution, quantum_value = optimizer.solve(quantum_instance=backend)
    print(f"Quantum Solution: {quantum_solution}, Value: {quantum_value:.4f}")

    # Solve classically for comparison
    classical_solution, classical_value = classical_portfolio_optimization(mu, sigma, risk_factor, budget)
    print(f"Classical Solution: {classical_solution}, Value: {classical_value:.4f}")

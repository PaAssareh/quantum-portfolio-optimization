import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QUBOOptimizer:
    def __init__(self, portfolio, risk_factor=0.5, budget=1, penalty=1):
        self.portfolio = portfolio
        self.risk_factor = risk_factor
        self.budget = budget
        self.penalty = penalty
        self.portfolio_weights = None
        self.result = None

    def _build_quadratic_program(self):
        n = len(self.portfolio.assets)
        qp = QuadraticProgram()
        qp.binary_var_list([f"x{i}" for i in range(n)])

        linear = -np.array(self.portfolio.expected_returns, dtype=float)
        quadratic = self.risk_factor * np.array(self.portfolio.covariance_matrix, dtype=float)

        qp.minimize(linear=linear, quadratic=quadratic)
        qp.linear_constraint(
            linear=[1] * n,
            sense="==",
            rhs=self.budget,
            name="budget"
        )
        return qp

    def optimize_portfolio(self, p=2, optimizer=None, maxiter=300):
        if optimizer is None:
            optimizer = COBYLA(maxiter=maxiter)

        qp = self._build_quadratic_program()
        sampler = StatevectorSampler(seed=42)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
        meo = MinimumEigenOptimizer(qaoa)
        result = meo.solve(qp)

        self.result = result
        self.portfolio_weights = np.array(result.x, dtype=float)
        return result

    def get_portfolio_weights(self):
        return self.portfolio_weights
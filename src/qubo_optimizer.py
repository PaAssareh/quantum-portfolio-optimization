import numpy as np

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QUBOOptimizer:
    def __init__(self, portfolio, risk_factor=0.5, budget=1, penalty=1):
        self.portfolio = portfolio
        self.risk_factor = float(risk_factor)
        self.budget = int(budget)
        self.penalty = float(penalty)

        self.portfolio_weights = None
        self.result = None
        self.quadratic_program = None

    def _build_quadratic_program(self):
        number_of_assets = len(self.portfolio.assets)

        if self.budget < 1 or self.budget > number_of_assets:
            raise ValueError(
                f"Budget must be between 1 and {number_of_assets}, "
                f"got {self.budget}."
            )

        qp = QuadraticProgram()

        variable_names = [
            f"x{i}" for i in range(number_of_assets)
        ]
        qp.binary_var_list(variable_names)

        expected_returns = np.asarray(
            self.portfolio.expected_returns,
            dtype=float,
        )

        covariance_matrix = np.asarray(
            self.portfolio.covariance_matrix,
            dtype=float,
        )

        if expected_returns.ndim != 1:
            raise ValueError(
                "Expected returns must be a one-dimensional array."
            )

        if expected_returns.shape[0] != number_of_assets:
            raise ValueError(
                "Expected returns size does not match "
                "the number of assets."
            )

        if covariance_matrix.shape != (
            number_of_assets,
            number_of_assets,
        ):
            raise ValueError(
                "Covariance matrix shape does not match "
                "the number of assets."
            )

        if not np.all(np.isfinite(expected_returns)):
            raise ValueError(
                "Expected returns contain non-finite values."
            )

        covariance_matrix = np.nan_to_num(
            covariance_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        covariance_matrix = (
            covariance_matrix + covariance_matrix.T
        ) / 2.0

        linear = -expected_returns
        quadratic = self.risk_factor * covariance_matrix

        qp.minimize(
            linear=linear,
            quadratic=quadratic,
        )

        qp.linear_constraint(
            linear=[1] * number_of_assets,
            sense="==",
            rhs=self.budget,
            name="budget",
        )

        self.quadratic_program = qp

        return qp

    def _validate_solution(self, solution):
        solution = np.asarray(solution, dtype=float)
        number_of_assets = len(self.portfolio.assets)

        if solution.ndim != 1:
            raise ValueError(
                "QAOA solution must be a one-dimensional vector."
            )

        if len(solution) != number_of_assets:
            raise ValueError(
                "QAOA solution size does not match "
                "the number of assets."
            )

        if not np.all(np.isfinite(solution)):
            raise ValueError(
                "QAOA solution contains non-finite values."
            )

        binary_solution = np.rint(solution).astype(int)

        if not np.allclose(
            solution,
            binary_solution,
            atol=1e-5,
        ):
            raise ValueError(
                f"QAOA solution is not binary: {solution}"
            )

        if not np.all(np.isin(binary_solution, [0, 1])):
            raise ValueError(
                "QAOA solution contains values other than 0 or 1."
            )

        selected_count = int(binary_solution.sum())

        if selected_count != self.budget:
            raise ValueError(
                "Infeasible QAOA solution: "
                f"selected {selected_count} assets, "
                f"expected {self.budget}."
            )

        # Convert selected assets into fully invested equal weights.
        return binary_solution.astype(float) / float(self.budget)

    def optimize_portfolio(
        self,
        p=2,
        optimizer=None,
        maxiter=300,
    ):
        p = int(p)
        maxiter = int(maxiter)

        if p < 1:
            raise ValueError(
                "QAOA reps p must be at least 1."
            )

        if maxiter < 1:
            raise ValueError(
                "QAOA maxiter must be at least 1."
            )

        if optimizer is None:
            optimizer = COBYLA(maxiter=maxiter)

        qp = self._build_quadratic_program()

        sampler = StatevectorSampler(seed=42)

        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=p,
        )

        minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = minimum_eigen_optimizer.solve(qp)

        validated_weights = self._validate_solution(result.x)

        self.result = result
        self.portfolio_weights = validated_weights

        return result

    def get_portfolio_weights(self):
        if self.portfolio_weights is None:
            raise RuntimeError(
                "Portfolio has not been optimized yet."
            )

        return self.portfolio_weights.copy()
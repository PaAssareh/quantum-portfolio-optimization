import numpy as np


class PortfolioCoPilot:
    def __init__(
        self,
        portfolio,
        quantum_weights,
        classical_weights=None,
        hybrid_weights=None,
        regime_info=None,
        metrics=None,
        benchmarks=None,
    ):
        self.portfolio = portfolio

        self.quantum_weights = (
            np.asarray(quantum_weights, dtype=float)
            if quantum_weights is not None
            else None
        )

        self.classical_weights = (
            np.asarray(classical_weights, dtype=float)
            if classical_weights is not None
            else None
        )

        self.hybrid_weights = (
            np.asarray(hybrid_weights, dtype=float)
            if hybrid_weights is not None
            else None
        )

        self.regime_info = regime_info
        self.metrics = metrics or {}
        self.benchmarks = benchmarks or {}

    def _fmt(self, value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"

        if isinstance(value, dict):
            formatted = {
                key: self._fmt(item)
                for key, item in value.items()
            }
            return str(formatted)

        return str(value)

    def generate_report(self):
        lines = []

        lines.append("Portfolio Co-Pilot Report")
        lines.append("=" * 30)

        if self.regime_info is not None:
            lines.append(
                f"Current regime: {self.regime_info}"
            )

        if self.metrics:
            lines.append("")
            lines.append("Key metrics:")

            for key, value in self.metrics.items():
                lines.append(
                    f"- {key}: {self._fmt(value)}"
                )

        if self.benchmarks:
            lines.append("")
            lines.append("Benchmark comparison:")

            for key, value in self.benchmarks.items():
                lines.append(
                    f"- {key}: {self._fmt(value)}"
                )

        if self.quantum_weights is not None:
            lines.append("")
            lines.append(
                f"Quantum weights: {self.quantum_weights}"
            )

        if self.classical_weights is not None:
            lines.append(
                f"Classical weights: {self.classical_weights}"
            )

        if self.hybrid_weights is not None:
            lines.append(
                f"Hybrid weights: {self.hybrid_weights}"
            )

        lines.append("")
        lines.append("Interpretation:")

        quantum_sharpe = float(
            self.metrics.get("quantum_sharpe", 0.0)
        )

        hybrid_sharpe = float(
            self.metrics.get("hybrid_sharpe", 0.0)
        )

        classical_sharpe = float(
            self.metrics.get("classical_sharpe", 0.0)
        )

        quantum_cvar_data = self.metrics.get(
            "quantum_cvar"
        )

        hybrid_cvar_data = self.metrics.get(
            "hybrid_cvar"
        )

        quantum_cvar = None
        hybrid_cvar = None

        if isinstance(quantum_cvar_data, dict):
            if "CVaR" in quantum_cvar_data:
                quantum_cvar = float(
                    quantum_cvar_data["CVaR"]
                )

        if isinstance(hybrid_cvar_data, dict):
            if "CVaR" in hybrid_cvar_data:
                hybrid_cvar = float(
                    hybrid_cvar_data["CVaR"]
                )

        same_sharpe = abs(
            hybrid_sharpe - quantum_sharpe
        ) <= 1e-10

        same_cvar = (
            quantum_cvar is not None
            and hybrid_cvar is not None
            and abs(hybrid_cvar - quantum_cvar) <= 1e-10
        )

        if same_sharpe and same_cvar:
            lines.append(
                "- Hybrid allocation preserved the quantum "
                "portfolio and produced identical Sharpe "
                "and CVaR metrics."
            )
        else:
            if hybrid_sharpe > quantum_sharpe:
                lines.append(
                    "- Hybrid post-processing improved Sharpe "
                    "relative to the quantum portfolio."
                )
            elif hybrid_sharpe < quantum_sharpe:
                lines.append(
                    "- Hybrid post-processing reduced Sharpe "
                    "relative to the quantum portfolio."
                )
            else:
                lines.append(
                    "- Hybrid post-processing preserved the "
                    "quantum Sharpe ratio."
                )

            if (
                quantum_cvar is not None
                and hybrid_cvar is not None
            ):
                if hybrid_cvar < quantum_cvar:
                    lines.append(
                        "- Hybrid allocation reduced tail risk "
                        "versus the quantum portfolio."
                    )
                elif hybrid_cvar > quantum_cvar:
                    lines.append(
                        "- Hybrid allocation increased tail risk "
                        "versus the quantum portfolio."
                    )
                else:
                    lines.append(
                        "- Hybrid allocation preserved the "
                        "quantum CVaR level."
                    )

        quantum_return = float(
            self.metrics.get("quantum_return", 0.0)
        )

        classical_return = float(
            self.metrics.get("classical_return", 0.0)
        )

        if quantum_return > classical_return:
            lines.append(
                "- The quantum portfolio achieved higher "
                "return than the classical baseline."
            )
        elif quantum_return < classical_return:
            lines.append(
                "- The classical baseline achieved higher "
                "return than the quantum portfolio."
            )
        else:
            lines.append(
                "- The quantum and classical portfolios achieved "
                "the same return."
            )

        if quantum_sharpe > classical_sharpe:
            lines.append(
                "- The quantum portfolio achieved a higher "
                "Sharpe ratio than the classical baseline."
            )
        elif quantum_sharpe < classical_sharpe:
            lines.append(
                "- The classical baseline achieved a higher "
                "Sharpe ratio than the quantum portfolio."
            )
        else:
            lines.append(
                "- The quantum and classical portfolios achieved "
                "the same Sharpe ratio."
            )

        return "\n".join(lines)
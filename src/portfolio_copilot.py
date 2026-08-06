import numpy as np


class PortfolioCoPilot:
    def __init__(self, portfolio, quantum_weights, classical_weights=None, hybrid_weights=None, regime_info=None, metrics=None, benchmarks=None):
        self.portfolio = portfolio
        self.quantum_weights = np.array(quantum_weights, dtype=float) if quantum_weights is not None else None
        self.classical_weights = np.array(classical_weights, dtype=float) if classical_weights is not None else None
        self.hybrid_weights = np.array(hybrid_weights, dtype=float) if hybrid_weights is not None else None
        self.regime_info = regime_info
        self.metrics = metrics or {}
        self.benchmarks = benchmarks or {}

    def _fmt(self, value):
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def generate_report(self):
        lines = []
        lines.append("Portfolio Co-Pilot Report")
        lines.append("=" * 30)

        if self.regime_info is not None:
            lines.append(f"Current regime: {self.regime_info}")

        if self.metrics:
            lines.append("")
            lines.append("Key metrics:")
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {self._fmt(value)}")

        if self.benchmarks:
            lines.append("")
            lines.append("Benchmark comparison:")
            for key, value in self.benchmarks.items():
                lines.append(f"- {key}: {self._fmt(value)}")

        if self.quantum_weights is not None:
            lines.append("")
            lines.append(f"Quantum weights: {self.quantum_weights}")

        if self.classical_weights is not None:
            lines.append(f"Classical weights: {self.classical_weights}")

        if self.hybrid_weights is not None:
            lines.append(f"Hybrid weights: {self.hybrid_weights}")

        lines.append("")
        lines.append("Interpretation:")
        if self.metrics.get("hybrid_sharpe", 0) >= self.metrics.get("quantum_sharpe", 0):
            lines.append("- Hybrid post-processing improved or preserved Sharpe relative to raw quantum output.")
        else:
            lines.append("- Hybrid post-processing reduced Sharpe relative to raw quantum output but may have lowered risk.")

        if "hybrid_cvar" in self.metrics and "quantum_cvar" in self.metrics:
            q_cvar = self.metrics["quantum_cvar"]["CVaR"]
            h_cvar = self.metrics["hybrid_cvar"]["CVaR"]
            if h_cvar <= q_cvar:
                lines.append("- Hybrid allocation reduced tail risk versus raw quantum output.")
            else:
                lines.append("- Hybrid allocation increased tail risk versus raw quantum output.")

        return "\n".join(lines)
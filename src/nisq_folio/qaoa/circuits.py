"""QAOA circuits with the standard transverse-field (X) mixer.

For an Ising cost Hamiltonian H_C = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i (+ constant) and
mixer H_M = sum_i X_i, the depth-p QAOA state is

    |gamma, beta> = prod_{l=1..p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^N

Gate decomposition (constants only change the global phase and are dropped):
    exp(-i gamma J_ij Z_i Z_j) = RZZ(2 gamma J_ij) on (i, j)
    exp(-i gamma h_i Z_i)      = RZ(2 gamma h_i) on i
    exp(-i beta X_i)           = RX(2 beta) on i

Constraint-preserving and adaptive mixers are added in ``nisq_folio.mixers`` (Week 6).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from nisq_folio.qubo.encoding import IsingModel


def qaoa_ansatz(
    ising: IsingModel, layers: int, measure: bool = False
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build the parameterized depth-p QAOA circuit with an X mixer.

    Args:
        ising: The cost Hamiltonian.
        layers: Depth p >= 1.
        measure: If True, measure all qubits at the end.

    Returns:
        ``(circuit, gammas, betas)``: the circuit and its p cost and p mixer parameters.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    n = ising.n
    gammas = ParameterVector("gamma", layers)
    betas = ParameterVector("beta", layers)
    circuit = QuantumCircuit(n, name=f"qaoa_p{layers}")
    circuit.h(range(n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if ising.j[i, j] != 0.0]
    for layer in range(layers):
        for i, j in pairs:
            circuit.rzz(2.0 * gammas[layer] * float(ising.j[i, j]), i, j)
        for i in range(n):
            if ising.h[i] != 0.0:
                circuit.rz(2.0 * gammas[layer] * float(ising.h[i]), i)
        circuit.rx(2.0 * betas[layer], range(n))
    if measure:
        circuit.measure_all()
    return circuit, gammas, betas

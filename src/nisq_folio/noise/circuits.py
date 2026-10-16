"""Gate-level circuits for noisy QAOA runs (DECISIONS.md ADR-023).

Dicke state |D^N_K> (uniform superposition of all weight-K bitstrings): the deterministic
"split and cyclic shift" construction of Baertschi & Eidenbenz (FCT 2019; verify the reference
before citing). Starting from K ones on the last K qubits, it applies, for m = N down to 2,
the block SCS_{m, min(K, m-1)}, built from

    two-qubit part on (m-2, m-1):
        CX(m-2 -> m-1) . CRY(2 acos sqrt(1/m)) on m-2, control m-1 . CX(m-2 -> m-1)
    three-qubit parts j = 2..k on (m-1-j, m-j, m-1):
        CX(m-1-j -> m-1) . CCRY(2 acos sqrt(j/m)) on m-1-j, controls (m-1, m-j) . CX(m-1-j -> m-1)

(0-based qubit indices). The Dicke state is permutation symmetric, so qubit ordering does not
matter for the result; correctness is verified against the exact state vector in the tests.

XY mixer gate: exp(-i beta (XX + YY) / 2) is Qiskit's ``XXPlusYYGate(2 * beta, 0)``.
Cost layer: exp(-i gamma H / s) with H = sum J ZZ + sum h Z is RZZ(2 gamma J / s) and
RZ(2 gamma h / s), dropping the constant; s is the same scale as the NumPy simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import CRYGate, RYGate, XXPlusYYGate

from nisq_folio.mixers.xy import ring_edges
from nisq_folio.problem import FloatArray
from nisq_folio.qubo.encoding import IsingModel

ParameterValue = float | ParameterExpression
"""An angle: a number, or a symbolic parameter bound later."""


def _scs(circuit: QuantumCircuit, size: int, k: int) -> None:
    """Append SCS_{size,k}, acting on qubits size-1-k .. size-1 (module docstring)."""
    top = size - 1
    circuit.cx(top - 1, top)
    circuit.append(CRYGate(2 * np.arccos(np.sqrt(1 / size))), [top, top - 1])
    circuit.cx(top - 1, top)
    for j in range(2, k + 1):
        target = top - j
        circuit.cx(target, top)
        doubly = RYGate(2 * np.arccos(np.sqrt(j / size))).control(2, annotated=False)
        circuit.append(doubly, [top, target + 1, target])
        circuit.cx(target, top)


def dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """Return a circuit preparing |D^N_K> from |0...0> (module docstring)."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    circuit = QuantumCircuit(n, name=f"dicke_{n}_{k}")
    for q in range(n - k, n):
        circuit.x(q)
    if k in (0, n):
        return circuit
    for size in range(n, 1, -1):
        _scs(circuit, size, min(k, size - 1))
    return circuit


def objective_ising(sigma: FloatArray, mu: FloatArray, a: float, b: float) -> IsingModel:
    """Ising form of f(x) = a x^T Sigma x - b mu^T x alone (no penalty), via x = (1 - z) / 2.

    Same algebra as ``qubo_to_ising`` with Q = a Sigma + diag(-b mu) and no constant.
    """
    q = a * np.asarray(sigma, dtype=np.float64)
    q[np.diag_indices_from(q)] -= b * np.asarray(mu, dtype=np.float64)
    q = (q + q.T) / 2
    diag = np.diag(q)
    off = q - np.diag(diag)
    j = np.triu(off, k=1) / 2.0
    h = -(diag + off.sum(axis=1)) / 2.0
    offset = diag.sum() / 2.0 + np.triu(off, k=1).sum() / 2.0
    return IsingModel(j=j, h=h, offset=float(offset))


def append_mixer(circuit: QuantumCircuit, name: str, beta: ParameterValue) -> None:
    """Append mixer ``name`` ("x", "xy_ring", or "xy_i_j") with angle beta."""
    n = circuit.num_qubits
    if name == "x":
        circuit.rx(2 * beta, range(n))
    elif name == "xy_ring":
        for i, j in ring_edges(n):
            circuit.append(XXPlusYYGate(2 * beta, 0.0), [i, j])
    elif name.startswith("xy_"):
        i, j = (int(v) for v in name.split("_")[1:])
        circuit.append(XXPlusYYGate(2 * beta, 0.0), [i, j])
    else:
        raise ValueError(f"unknown mixer {name!r}")


def variant_circuit(
    ising: IsingModel,
    scale: float,
    initial: str,
    k: int,
    mixers: Sequence[str],
    gammas: Sequence[ParameterValue],
    betas: Sequence[ParameterValue],
) -> QuantumCircuit:
    """Build the bound QAOA circuit for a mixer sequence (no measurement).

    Args:
        ising: Cost Hamiltonian (QUBO with penalty for "x", objective only otherwise).
        scale: s; the cost layer implements exp(-i gamma H / s).
        initial: ``"plus"`` (|+>^N) or ``"dicke"`` (|D^N_K>).
        k: K, used for the Dicke state.
        mixers: Mixer name per layer.
        gammas: Cost angles, one per layer.
        betas: Mixer angles, one per layer.

    Returns:
        The circuit on N qubits.
    """
    n = ising.n
    if not (len(mixers) == len(gammas) == len(betas)):
        raise ValueError("mixers, gammas and betas must have the same length")
    if initial == "plus":
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
    elif initial == "dicke":
        circuit = dicke_circuit(n, k)
    else:
        raise ValueError(f"initial must be 'plus' or 'dicke', got {initial!r}")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if ising.j[i, j] != 0.0]
    for name, gamma, beta in zip(mixers, gammas, betas, strict=True):
        for i, j in pairs:
            circuit.rzz(2 * gamma * float(ising.j[i, j]) / scale, i, j)
        for i in range(n):
            if ising.h[i] != 0.0:
                circuit.rz(2 * gamma * float(ising.h[i]) / scale, i)
        append_mixer(circuit, name, beta)
    return circuit

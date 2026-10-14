"""QUBO and Ising encodings of the cardinality-constrained portfolio problem.

The cardinality constraint sum(x) = K is moved into the objective as a quadratic penalty:

    E(x) = f(x) + P * (1^T x - K)^2
         = x^T Q x + c,   with   Q = a*Sigma + P*1 1^T + diag(-b*mu - 2*P*K),   c = P*K^2

using f(x) = a x^T Sigma x - b mu^T x (a = lambda/K^2, b = 1/K, see ``problem.py``) and
x_i^2 = x_i for binary x. On feasible x the penalty is zero, so E(x) = f(x).

Substituting x_i = (1 - z_i) / 2 with spins z_i in {+1, -1} gives the Ising form

    E = sum_{i<j} J_ij z_i z_j + sum_i h_i z_i + offset,
    J_ij = Q_ij / 2,   h_i = -(Q_ii + sum_{j != i} Q_ij) / 2,
    offset = c + sum_i Q_ii / 2 + sum_{i<j} Q_ij / 2

Bit/spin convention (matches Qiskit): qubit i measured as 1 <=> z_i = -1 <=> x_i = 1 (asset i
selected). In a computational-basis index k, qubit i is bit i of k (little-endian).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from nisq_folio.problem import FloatArray, PortfolioProblem


@dataclass(frozen=True)
class QuboModel:
    """E(x) = x^T Q x + constant over x in {0,1}^N.

    Attributes:
        q: Symmetric matrix Q, shape (N, N); linear terms are on the diagonal.
        constant: Constant term c.
        penalty: The penalty weight P used to build it.
    """

    q: FloatArray
    constant: float
    penalty: float

    @property
    def n(self) -> int:
        """Number of binary variables N."""
        return int(self.q.shape[0])

    def energy(self, x: np.ndarray) -> float:
        """Return E(x) = x^T Q x + c for a 0/1 vector x."""
        xf = np.asarray(x, dtype=np.float64)
        return float(xf @ self.q @ xf + self.constant)

    def all_energies(self) -> FloatArray:
        """Return E for all 2^N bitstrings, indexed little-endian (bit i of k is x_i).

        Memory is O(N * 2^N); intended for N <= 20.
        """
        bits = all_bitstrings(self.n)
        return np.einsum("ki,ij,kj->k", bits, self.q, bits) + self.constant


@dataclass(frozen=True)
class IsingModel:
    """E(z) = sum_{i<j} J_ij z_i z_j + sum_i h_i z_i + offset over z in {+1,-1}^N.

    Attributes:
        j: Couplings, shape (N, N), strictly upper triangular (J_ij for i < j).
        h: Local fields, shape (N,).
        offset: Constant energy offset.
    """

    j: FloatArray
    h: FloatArray
    offset: float

    @property
    def n(self) -> int:
        """Number of spins N."""
        return int(self.h.size)

    def energy(self, z: np.ndarray) -> float:
        """Return E(z) for a +/-1 vector z."""
        zf = np.asarray(z, dtype=np.float64)
        return float(zf @ self.j @ zf + self.h @ zf + self.offset)

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Return the cost Hamiltonian H_C = sum J_ij Z_i Z_j + sum h_i Z_i + offset * I.

        Its diagonal in the computational basis is E evaluated on every bitstring, so
        <psi|H_C|psi> is the expected QUBO energy of the measured selection.
        """
        terms: list[tuple[str, list[int], float]] = [("", [], self.offset)]
        rows, cols = np.nonzero(self.j)
        terms += [
            ("ZZ", [int(r), int(c)], float(self.j[r, c])) for r, c in zip(rows, cols, strict=True)
        ]
        terms += [("Z", [i], float(self.h[i])) for i in np.flatnonzero(self.h)]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n).simplify(atol=0.0)


def all_bitstrings(n: int) -> FloatArray:
    """Return all 2^N bit vectors as rows, little-endian: row k has x_i = bit i of k."""
    k = np.arange(2**n, dtype=np.int64)[:, None]
    return ((k >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)


def build_qubo(problem: PortfolioProblem, penalty: float) -> QuboModel:
    """Encode a problem as a QUBO with penalty weight P on (1^T x - K)^2.

    Implements Q = a*Sigma + P*1 1^T + diag(-b*mu - 2*P*K) and c = P*K^2 (module docstring).

    Args:
        problem: The portfolio problem.
        penalty: P > 0.

    Returns:
        The QUBO model.
    """
    if not (np.isfinite(penalty) and penalty > 0):
        raise ValueError(f"penalty must be finite and > 0, got {penalty}")
    n, k = problem.n, problem.k
    a, b = problem.quadratic_coefficient, problem.linear_coefficient
    q = a * problem.sigma + penalty * np.ones((n, n))
    q[np.diag_indices(n)] += -b * problem.mu - 2.0 * penalty * k
    return QuboModel(q=(q + q.T) / 2.0, constant=penalty * k**2, penalty=float(penalty))


def qubo_to_ising(qubo: QuboModel) -> IsingModel:
    """Convert x^T Q x + c to Ising form via x_i = (1 - z_i) / 2 (module docstring).

    Args:
        qubo: A QUBO with symmetric Q.

    Returns:
        The equivalent Ising model: E_ising(z(x)) = E_qubo(x) for every bitstring.
    """
    q = qubo.q
    diag = np.diag(q)
    off = q - np.diag(diag)
    # Each pair i<j contributes 2 Q_ij x_i x_j = 2 Q_ij (1 - z_i - z_j + z_i z_j) / 4.
    j = np.triu(off, k=1) / 2.0
    h = -(diag + off.sum(axis=1)) / 2.0
    offset = qubo.constant + diag.sum() / 2.0 + np.triu(off, k=1).sum() / 2.0
    return IsingModel(j=j, h=h, offset=float(offset))

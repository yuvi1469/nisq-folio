"""Exact noiseless QAOA simulation in NumPy (DECISIONS.md ADR-020).

Because the cost Hamiltonian is diagonal in the computational basis, the depth-p QAOA state

    |gamma, beta> = prod_{l=1..p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^N

can be computed directly on the 2^N amplitudes:

- cost layer:  psi_k <- exp(-i gamma E_k / s) psi_k, with E_k the QUBO energy of bitstring k and
  s the Hamiltonian scale (below); constants in E only change the global phase;
- X mixer:     exp(-i beta X) = cos(beta) I - i sin(beta) X applied to every qubit.

Energy scale: the cost Hamiltonian is divided by s = sqrt(sum_{i<j} J_ij^2 + sum_i h_i^2), the
standard deviation of the Ising energy under uniformly random spins (the Pauli terms are
orthogonal). This makes gamma = O(1) meaningful for every instance, independent of the units of
mu, Sigma, and P. Indexing is little-endian, as in Qiskit and ``qubo.encoding``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nisq_folio.problem import FloatArray
from nisq_folio.qubo.encoding import IsingModel

ComplexArray = NDArray[np.complex128]


def hamiltonian_scale(ising: IsingModel) -> float:
    """Return s = sqrt(sum J_ij^2 + sum h_i^2), the energy std under uniform random spins."""
    scale = float(np.sqrt(np.sum(ising.j**2) + np.sum(ising.h**2)))
    if scale <= 0:
        raise ValueError("cost Hamiltonian has no non-constant terms")
    return scale


def apply_x_mixer(state: ComplexArray, beta: float, n: int) -> ComplexArray:
    """Apply exp(-i beta sum_i X_i) = prod_i (cos(beta) I - i sin(beta) X_i) to ``state``.

    Args:
        state: 2^N amplitudes, little-endian.
        beta: Mixer angle.
        n: Number of qubits N.

    Returns:
        The new state (a new array).
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    out = np.array(state, dtype=np.complex128)
    for qubit in range(n):
        # Index k = high * 2^(q+1) + bit * 2^q + low, so this view's middle axis is qubit q.
        view = out.reshape(-1, 2, 2**qubit)
        zero, one = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :] = c * zero + s * one
        view[:, 1, :] = s * zero + c * one
    return out


class NoiselessQaoa:
    """Exact X-mixer QAOA for a diagonal cost function given by its 2^N energies."""

    def __init__(self, energies: FloatArray, scale: float) -> None:
        """Prepare the simulator.

        Args:
            energies: E_k for every bitstring k (little-endian), length 2^N.
            scale: Hamiltonian scale s; the cost layer uses E / s.
        """
        energies = np.asarray(energies, dtype=np.float64)
        n = int(np.log2(energies.size))
        if energies.ndim != 1 or 2**n != energies.size:
            raise ValueError("energies must have length 2^N")
        if not scale > 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.n = n
        self._phases = energies / scale

    def state(self, gammas: FloatArray, betas: FloatArray) -> ComplexArray:
        """Return the QAOA state for angles (gamma_1..p, beta_1..p)."""
        gammas, betas = np.atleast_1d(gammas), np.atleast_1d(betas)
        if gammas.shape != betas.shape or gammas.ndim != 1 or gammas.size == 0:
            raise ValueError("gammas and betas must be equal-length, non-empty 1-D arrays")
        psi = np.full(2**self.n, 2 ** (-self.n / 2), dtype=np.complex128)
        for gamma, beta in zip(gammas, betas, strict=True):
            psi = psi * np.exp(-1j * gamma * self._phases)
            psi = apply_x_mixer(psi, float(beta), self.n)
        return psi

    def probabilities(self, gammas: FloatArray, betas: FloatArray) -> FloatArray:
        """Return measurement probabilities |psi_k|^2, renormalized to sum exactly to 1."""
        probs = np.abs(self.state(gammas, betas)) ** 2
        return probs / probs.sum()

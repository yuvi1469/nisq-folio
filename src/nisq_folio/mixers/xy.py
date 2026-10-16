"""Hamming-weight-preserving (XY) mixers and the Dicke initial state (DECISIONS.md ADR-021).

An XY interaction on qubits (i, j) has generator G_ij = (X_i X_j + Y_i Y_j) / 2, which maps
|01> <-> |10> and annihilates |00> and |11>. So

    exp(-i beta G_ij) = cos(beta) on span{|01>, |10>} mixed by -i sin(beta), identity elsewhere

swaps one selected asset for one unselected asset in superposition and never changes the
number of selected assets. Starting from the Dicke state

    |D^N_K> = C(N, K)^(-1/2) * sum_{x : 1^T x = K} |x>,

which is the uniform superposition of all feasible selections, an XY-mixer QAOA stays inside
the feasible subspace (noiselessly), so no penalty term is needed and every sample is valid.

The ring mixer is the ordered product over ring edges (i, i+1 mod N), even-i edges first, then
odd-i edges: one first-order Trotter step of sum_i G_{i,i+1}, as it would be run as a circuit.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

from nisq_folio.qaoa.statevector import ComplexArray


@lru_cache(maxsize=4096)
def _pair_indices(n: int, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices with (bit_i, bit_j) = (0, 1), and their partners with (1, 0)."""
    k = np.arange(2**n, dtype=np.int64)
    first = k[(((k >> i) & 1) == 0) & (((k >> j) & 1) == 1)]
    second = first ^ ((1 << i) | (1 << j))
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


def apply_xy_pair(state: ComplexArray, i: int, j: int, beta: float, n: int) -> ComplexArray:
    """Apply exp(-i beta (X_i X_j + Y_i Y_j) / 2) to ``state`` (little-endian, N qubits).

    Args:
        state: 2^N amplitudes.
        i: First qubit.
        j: Second qubit (different from ``i``).
        beta: Rotation angle.
        n: Number of qubits N.

    Returns:
        The new state (a new array).
    """
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"need two distinct qubits in [0, {n}), got {i}, {j}")
    first, second = _pair_indices(n, i, j)
    c, s = np.cos(beta), -1j * np.sin(beta)
    out = np.array(state, dtype=np.complex128)
    a, b = out[first].copy(), out[second].copy()
    out[first] = c * a + s * b
    out[second] = s * a + c * b
    return out


def ring_edges(n: int) -> list[tuple[int, int]]:
    """Return ring edges (i, i+1 mod N) in parity order: even i first, then odd i."""
    if n < 2:
        raise ValueError("a ring needs at least two qubits")
    edges = [(0, 1)] if n == 2 else [(i, (i + 1) % n) for i in range(n)]
    return [e for e in edges if e[0] % 2 == 0] + [e for e in edges if e[0] % 2 == 1]


def apply_xy_ring(state: ComplexArray, beta: float, n: int) -> ComplexArray:
    """Apply one parity-ordered Trotter step of the XY ring mixer (module docstring)."""
    for i, j in ring_edges(n):
        state = apply_xy_pair(state, i, j, beta, n)
    return state


def dicke_state(n: int, k: int) -> ComplexArray:
    """Return |D^N_K>: equal amplitude C(N, K)^(-1/2) on every bitstring with K ones."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    weights = np.array([bin(x).count("1") for x in range(2**n)])
    state = np.zeros(2**n, dtype=np.complex128)
    state[weights == k] = comb(n, k) ** -0.5
    return state

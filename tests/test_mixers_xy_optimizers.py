"""Tests for XY mixers, the Dicke state, and the SPSA / COBYLA optimizer wrappers."""

from math import comb

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from nisq_folio.mixers.xy import apply_xy_pair, apply_xy_ring, dicke_state, ring_edges
from nisq_folio.qaoa.optimizers import (
    SpsaSettings,
    minimum_budget,
    run_cobyla,
    run_spsa,
)


def xy_generator(n: int, i: int, j: int) -> np.ndarray:
    """Dense (X_i X_j + Y_i Y_j) / 2 from Qiskit Pauli operators (independent reference)."""
    return SparsePauliOp.from_sparse_list([("XX", [i, j], 0.5), ("YY", [i, j], 0.5)], n).to_matrix()


def random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return psi / np.linalg.norm(psi)


def weight_distribution(psi: np.ndarray, n: int) -> np.ndarray:
    weights = np.array([bin(x).count("1") for x in range(2**n)])
    return np.bincount(weights, weights=np.abs(psi) ** 2, minlength=n + 1)


# ---------------------------------------------------------------- XY mixers


@settings(max_examples=40)
@given(st.integers(2, 5), st.data(), st.floats(-3, 3), st.integers(0, 1000))
def test_pair_rotation_equals_matrix_exponential(
    n: int, data: st.DataObject, beta: float, seed: int
) -> None:
    i, j = data.draw(st.lists(st.integers(0, n - 1), min_size=2, max_size=2, unique=True))
    psi = random_state(n, seed)
    expected = expm(-1j * beta * xy_generator(n, i, j)) @ psi
    np.testing.assert_allclose(apply_xy_pair(psi, i, j, beta, n), expected, atol=1e-12)


@settings(max_examples=25)
@given(st.integers(2, 6), st.floats(-3, 3), st.integers(0, 1000))
def test_ring_is_the_ordered_product_of_pair_rotations(n: int, beta: float, seed: int) -> None:
    psi = random_state(n, seed)
    expected = psi
    for i, j in ring_edges(n):
        expected = expm(-1j * beta * xy_generator(n, i, j)) @ expected
    np.testing.assert_allclose(apply_xy_ring(psi, beta, n), expected, atol=1e-11)


@given(st.integers(2, 7), st.floats(-3, 3), st.integers(0, 1000))
def test_xy_ring_preserves_hamming_weight(n: int, beta: float, seed: int) -> None:
    """The probability of each number of selected assets is unchanged: feasibility is kept."""
    psi = random_state(n, seed)
    np.testing.assert_allclose(
        weight_distribution(apply_xy_ring(psi, beta, n), n), weight_distribution(psi, n), atol=1e-12
    )


def test_ring_edges_are_parity_ordered() -> None:
    assert ring_edges(2) == [(0, 1)]
    assert ring_edges(4) == [(0, 1), (2, 3), (1, 2), (3, 0)]
    assert sorted(ring_edges(6)) == sorted((i, (i + 1) % 6) for i in range(6))
    with pytest.raises(ValueError):
        ring_edges(1)


@given(st.integers(1, 8), st.data())
def test_dicke_state_is_uniform_over_weight_k(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(0, n))
    psi = dicke_state(n, k)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    dist = weight_distribution(psi, n)
    assert dist[k] == pytest.approx(1.0)
    nonzero = np.abs(psi[np.abs(psi) > 0])
    np.testing.assert_allclose(nonzero, comb(n, k) ** -0.5)


def test_invalid_mixer_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        apply_xy_pair(np.ones(4), 1, 1, 0.3, 2)
    with pytest.raises(ValueError, match="distinct"):
        apply_xy_pair(np.ones(4), 0, 2, 0.3, 2)
    with pytest.raises(ValueError):
        dicke_state(3, 4)


# ---------------------------------------------------------------- optimizers


def counted(fun):  # noqa: ANN001
    calls = []

    def wrapper(x: np.ndarray) -> float:
        calls.append(x.copy())
        return fun(x)

    return wrapper, calls


@given(st.integers(12, 200), st.integers(1, 6))
def test_spsa_uses_exactly_its_budgeted_evaluations(budget: int, dim: int) -> None:
    fun, calls = counted(lambda x: float(np.sum(x**2)))
    run_spsa(fun, np.ones(dim), budget, np.random.default_rng(0))
    m = SpsaSettings().calibration_samples
    assert len(calls) == 2 * m + 2 * ((budget - 2 * m) // 2) <= budget


def test_spsa_minimizes_a_noisy_quadratic_reproducibly() -> None:
    noise = np.random.default_rng(1)
    target = np.array([0.4, -0.3, 0.2])

    def f(x: np.ndarray) -> float:
        return float(np.sum((x - target) ** 2) + 0.001 * noise.normal())

    x = run_spsa(f, np.zeros(3), 400, np.random.default_rng(3))
    assert np.linalg.norm(x - target) < 0.1
    a = run_spsa(lambda x: float(np.sum(x**2)), np.ones(2), 50, np.random.default_rng(9))
    b = run_spsa(lambda x: float(np.sum(x**2)), np.ones(2), 50, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def test_spsa_handles_a_flat_objective() -> None:
    x = run_spsa(lambda x: 1.0, np.ones(2), 20, np.random.default_rng(0))
    assert np.isfinite(x).all()


def test_cobyla_wrapper_respects_budget() -> None:
    fun, calls = counted(lambda x: float(np.sum((x - 1) ** 2)))
    x = run_cobyla(fun, np.zeros(2), 30, 0.25)
    assert len(calls) <= 30
    assert np.linalg.norm(x - 1) < 0.1


def test_budgets_and_settings_are_validated() -> None:
    assert minimum_budget("cobyla", 4) == 6
    assert minimum_budget("spsa", 4) == 12
    with pytest.raises(ValueError, match="optimizer"):
        minimum_budget("adam", 2)
    with pytest.raises(ValueError, match="COBYLA"):
        run_cobyla(lambda x: 0.0, np.zeros(4), 5, 0.1)
    with pytest.raises(ValueError, match="SPSA"):
        run_spsa(lambda x: 0.0, np.zeros(2), 11, np.random.default_rng(0))
    for bad in ({"target_step": 0}, {"perturbation": -1}, {"calibration_samples": 0}):
        with pytest.raises(ValueError):
            SpsaSettings(**bad)

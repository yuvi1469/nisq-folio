"""Tests for nisq_folio.qubo.encoding and nisq_folio.qubo.penalty."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import problems, random_problem
from nisq_folio.baselines.brute_force import enumerate_objectives
from nisq_folio.problem import PortfolioProblem
from nisq_folio.qubo.encoding import all_bitstrings, build_qubo, qubo_to_ising
from nisq_folio.qubo.penalty import (
    cardinality_lower_bounds,
    minimal_exact_penalty,
    sufficient_penalty,
)


def objective_all(problem: PortfolioProblem) -> np.ndarray:
    """f(x) for all 2^N bitstrings, little-endian, via the problem's own scalar objective."""
    return np.array([problem.objective(b.astype(int)) for b in all_bitstrings(problem.n)])


def indefinite_problem(n: int, k: int, seed: int) -> PortfolioProblem:
    """A problem whose Sigma is symmetric but NOT positive semidefinite (stresses the bound)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, size=(n, n))
    return PortfolioProblem(
        rng.normal(0.1, 0.1, n), (a + a.T) / 2, k, 4.5, tuple(map(str, range(n)))
    )


# ---------------------------------------------------------------- encoding


def test_bitstrings_are_little_endian() -> None:
    bits = all_bitstrings(3)
    assert bits.shape == (8, 3)
    np.testing.assert_array_equal(bits[1], [1, 0, 0])  # k=1 -> x_0 = 1
    np.testing.assert_array_equal(bits[6], [0, 1, 1])  # k=6 = 0b110


@given(problems(max_n=7), st.floats(0.01, 50.0))
def test_qubo_energy_is_objective_plus_penalty(problem: PortfolioProblem, penalty: float) -> None:
    """E(x) = f(x) + P (1^T x - K)^2 for every bitstring."""
    qubo = build_qubo(problem, penalty)
    bits = all_bitstrings(problem.n)
    expected = objective_all(problem) + penalty * (bits.sum(axis=1) - problem.k) ** 2
    np.testing.assert_allclose(qubo.all_energies(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(qubo.q, qubo.q.T)
    assert qubo.energy(bits[3]) == pytest.approx(expected[3], rel=1e-10, abs=1e-12)


@given(problems(max_n=7), st.floats(0.01, 50.0))
def test_ising_energy_matches_qubo_on_every_bitstring(
    problem: PortfolioProblem, penalty: float
) -> None:
    """With z_i = 1 - 2 x_i (bit 1 <=> spin -1), E_ising(z) = E_qubo(x)."""
    qubo = build_qubo(problem, penalty)
    ising = qubo_to_ising(qubo)
    assert np.allclose(np.tril(ising.j), 0.0)  # strictly upper triangular
    for bits, energy in zip(all_bitstrings(problem.n), qubo.all_energies(), strict=True):
        assert ising.energy(1 - 2 * bits) == pytest.approx(energy, rel=1e-10, abs=1e-12)


@given(problems(max_n=6), st.floats(0.01, 50.0))
def test_pauli_operator_is_diagonal_with_qubo_energies(
    problem: PortfolioProblem, penalty: float
) -> None:
    """H_C's matrix is diagonal and its diagonal (Qiskit ordering) equals E on each bitstring."""
    qubo = build_qubo(problem, penalty)
    matrix = qubo_to_ising(qubo).to_sparse_pauli_op().to_matrix()
    np.testing.assert_allclose(np.diag(matrix).real, qubo.all_energies(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(matrix - np.diag(np.diag(matrix)), 0.0, atol=1e-12)


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_penalty_is_rejected(penalty: float) -> None:
    with pytest.raises(ValueError, match="penalty"):
        build_qubo(random_problem(4, 2, 1.0, seed=0), penalty)


# ---------------------------------------------------------------- penalty rule


@given(problems(max_n=7))
def test_sufficient_penalty_makes_qubo_minimum_the_constrained_optimum(
    problem: PortfolioProblem,
) -> None:
    """The guarantee of ADR-017: argmin_x E(x) is feasible and E_min = f*."""
    energies = build_qubo(problem, sufficient_penalty(problem)).all_energies()
    best = int(np.argmin(energies))
    assert all_bitstrings(problem.n)[best].sum() == problem.k
    assert energies[best] == pytest.approx(enumerate_objectives(problem)[1].min(), abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_guarantee_holds_for_indefinite_sigma(seed: int) -> None:
    """The bound uses lambda_min(Sigma), so it must hold even when Sigma is not PSD."""
    problem = indefinite_problem(7, 3, seed)
    assert np.linalg.eigvalsh(problem.sigma).min() < 0
    energies = build_qubo(problem, sufficient_penalty(problem)).all_energies()
    assert all_bitstrings(7)[np.argmin(energies)].sum() == 3


@given(problems(max_n=7))
def test_cardinality_lower_bounds_are_valid(problem: PortfolioProblem) -> None:
    """L_m <= f(x) for every x with exactly m selected assets."""
    bounds = cardinality_lower_bounds(problem)
    f_all = objective_all(problem)
    sizes = all_bitstrings(problem.n).sum(axis=1).astype(int)
    for m in range(problem.n + 1):
        assert bounds[m] <= f_all[sizes == m].min() + 1e-12


@given(problems(max_n=7))
def test_minimal_exact_penalty_is_the_threshold(problem: PortfolioProblem) -> None:
    """Just above P_exact the QUBO is exact; just below (if P_exact > 0) it is not."""
    p_exact = minimal_exact_penalty(problem)
    f_star = enumerate_objectives(problem)[1].min()
    feasible = all_bitstrings(problem.n).sum(axis=1) == problem.k

    above = build_qubo(problem, p_exact * 1.001 + 1e-9).all_energies()
    assert above[~feasible].min() >= f_star - 1e-12
    if p_exact > 1e-6:
        below = build_qubo(problem, p_exact * 0.99).all_energies()
        assert below[~feasible].min() < f_star
    assert sufficient_penalty(problem) >= p_exact


def test_flat_problem_gets_a_tiny_positive_penalty() -> None:
    """Mu = 0, Sigma = 0: every requirement is <= 0, so any P > 0 is exact."""
    problem = PortfolioProblem(np.zeros(4), np.zeros((4, 4)), 2, 1.0, tuple("abcd"))
    assert 0.0 < sufficient_penalty(problem) < 1e-300
    assert minimal_exact_penalty(problem) == 0.0


def test_margin_must_be_positive() -> None:
    with pytest.raises(ValueError, match="margin"):
        sufficient_penalty(random_problem(4, 2, 1.0, seed=0), margin=0.0)

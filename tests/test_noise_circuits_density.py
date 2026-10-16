"""Tests for gate-level circuits, the scaled noise model, and noisy QAOA runs (ADR-023).

The central checks: every circuit matches the exact NumPy simulator, and at noise level 0 the
whole noisy pipeline (transpile, route, density matrix, readout) reproduces the noiseless one.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import Statevector

from helpers import random_problem
from nisq_folio.mixers.variants import VariantSettings, adapt_pool, evolve, run_variant
from nisq_folio.mixers.xy import apply_xy_pair, dicke_state
from nisq_folio.noise.circuits import append_mixer, dicke_circuit, objective_ising, variant_circuit
from nisq_folio.noise.density import (
    NoisyEvaluator,
    ReferenceRates,
    apply_readout_error,
    depolarizing_parameter,
    scaled_noise_model,
)
from nisq_folio.noise.runner import run_noisy_variant
from nisq_folio.qaoa.metrics import SolutionLandscape
from nisq_folio.qaoa.statevector import NoiselessQaoa
from nisq_folio.qubo.encoding import all_bitstrings, build_qubo, qubo_to_ising
from nisq_folio.qubo.penalty import sufficient_penalty

RATES = ReferenceRates(single_qubit=2e-4, two_qubit=2e-3, readout=1e-2)
BACKEND = "FakeGuadalupeV2"  # small device: fast transpile and noise in tests


# ---------------------------------------------------------------- circuits


@pytest.mark.parametrize(("n", "k"), [(n, k) for n in range(1, 8) for k in range(n + 1)])
def test_dicke_circuit_prepares_the_dicke_state(n: int, k: int) -> None:
    fidelity = abs(np.vdot(Statevector(dicke_circuit(n, k)).data, dicke_state(n, k))) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-10)


@settings(max_examples=20)
@given(st.integers(2, 5), st.data(), st.floats(-3, 3), st.integers(0, 999))
def test_xy_gate_matches_numpy(n: int, data: st.DataObject, beta: float, seed: int) -> None:
    i, j = data.draw(st.lists(st.integers(0, n - 1), min_size=2, max_size=2, unique=True))
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n)
    append_mixer(circuit, f"xy_{min(i, j)}_{max(i, j)}", beta)
    out = Statevector(psi).evolve(circuit).data
    np.testing.assert_allclose(out, apply_xy_pair(psi, min(i, j), max(i, j), beta, n), atol=1e-12)


def test_objective_ising_matches_the_objective() -> None:
    problem = random_problem(5, 2, 4.5, seed=1)
    ising = objective_ising(
        problem.sigma, problem.mu, problem.quadratic_coefficient, problem.linear_coefficient
    )
    for bits in all_bitstrings(5):
        assert ising.energy(1 - 2 * bits) == pytest.approx(problem.objective(bits.astype(int)))


@pytest.mark.parametrize("seq", [["xy_ring", "xy_1_3"], ["xy_0_4"], ["xy_ring"] * 3])
def test_dicke_variant_circuit_matches_numpy(seq: list[str]) -> None:
    problem = random_problem(5, 2, 4.5, seed=6)
    land = SolutionLandscape.from_problem(problem)
    ising = objective_ising(
        problem.sigma, problem.mu, problem.quadratic_coefficient, problem.linear_coefficient
    )
    s = land.objective.std()
    g, b = np.linspace(0.3, 1.2, len(seq)), np.linspace(-0.5, 0.4, len(seq))
    pool = adapt_pool(5)
    exact = np.abs(evolve(dicke_state(5, 2), land.objective / s, g, b, [pool[m] for m in seq])) ** 2
    circuit = variant_circuit(ising, s, "dicke", 2, seq, list(g), list(b))
    np.testing.assert_allclose(np.abs(Statevector(circuit).data) ** 2, exact, atol=1e-12)


def test_plus_variant_circuit_matches_numpy() -> None:
    problem = random_problem(5, 2, 4.5, seed=6)
    qubo = build_qubo(problem, 0.5)
    energies = qubo.all_energies()
    g, b = np.array([0.4, 0.9]), np.array([0.2, -0.7])
    circuit = variant_circuit(qubo_to_ising(qubo), energies.std(), "plus", 2, ["x", "x"],
                              list(g), list(b))  # fmt: skip
    exact = NoiselessQaoa(energies, energies.std()).probabilities(g, b)
    np.testing.assert_allclose(np.abs(Statevector(circuit).data) ** 2, exact, atol=1e-12)


def test_circuit_arguments_are_validated() -> None:
    ising = qubo_to_ising(build_qubo(random_problem(3, 1, 1.0, 0), 1.0))
    with pytest.raises(ValueError, match="same length"):
        variant_circuit(ising, 1.0, "plus", 1, ["x"], [0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="initial"):
        variant_circuit(ising, 1.0, "zero", 1, ["x"], [0.1], [0.1])
    with pytest.raises(ValueError, match="unknown mixer"):
        variant_circuit(ising, 1.0, "plus", 1, ["y"], [0.1], [0.1])
    with pytest.raises(ValueError):
        dicke_circuit(2, 3)


# ---------------------------------------------------------------- noise model


def test_depolarizing_parameter_matches_average_infidelity() -> None:
    assert depolarizing_parameter(0.01, 1) == pytest.approx(0.02)
    assert depolarizing_parameter(0.03, 2) == pytest.approx(0.04)


def test_noise_model_is_ideal_at_level_zero_and_scales() -> None:
    assert scaled_noise_model(0.0, RATES).is_ideal()
    assert not scaled_noise_model(1.0, RATES).is_ideal()
    with pytest.raises(ValueError):
        scaled_noise_model(-1.0, RATES)


def test_reference_rates_from_kingston() -> None:
    rates = ReferenceRates.from_backend("FakeKingston")
    assert 0 < rates.single_qubit < rates.two_qubit < rates.readout < 0.05


@given(st.integers(1, 6), st.floats(0.0, 0.5), st.integers(0, 999))
def test_readout_error_is_a_stochastic_map(n: int, flip: float, seed: int) -> None:
    p = np.random.default_rng(seed).dirichlet(np.ones(2**n))
    out = apply_readout_error(p, flip)
    assert out.min() >= -1e-15
    assert out.sum() == pytest.approx(1.0)
    if flip == 0.0:
        np.testing.assert_array_equal(out, p)


def test_readout_error_known_values() -> None:
    # One qubit certainly 0, flip 0.1 -> 0.9 / 0.1. Two qubits |00> -> independent flips.
    np.testing.assert_allclose(apply_readout_error(np.array([1.0, 0.0]), 0.1), [0.9, 0.1])
    two = apply_readout_error(np.array([1.0, 0, 0, 0]), 0.1)
    np.testing.assert_allclose(two, [0.81, 0.09, 0.09, 0.01])
    with pytest.raises(ValueError):
        apply_readout_error(np.array([1.0, 0.0]), 0.6)


# ---------------------------------------------------------------- noisy evaluation and runs


def make(level: float, mixer: str = "xy_ring") -> tuple[NoisyEvaluator, SolutionLandscape]:
    problem = random_problem(5, 2, 4.5, seed=2)
    land = SolutionLandscape.from_problem(problem)
    ising = objective_ising(
        problem.sigma, problem.mu, problem.quadratic_coefficient, problem.linear_coefficient
    )
    ev = NoisyEvaluator(ising, land.objective.std(), "dicke", 2, level, RATES, backend=BACKEND)
    return ev, land


def test_noisy_pipeline_at_level_zero_reproduces_the_exact_simulator() -> None:
    """Transpile + routing + density matrix + layout + readout at m = 0 equals NumPy."""
    ev, land = make(0.0)
    seq, g, b = ["xy_ring", "xy_0_3"], np.array([0.5, 1.0]), np.array([0.3, -0.2])
    pool = adapt_pool(5)
    exact = np.abs(evolve(dicke_state(5, 2), land.objective / land.objective.std(), g, b,
                          [pool[m] for m in seq])) ** 2  # fmt: skip
    np.testing.assert_allclose(ev.distribution(seq, g, b), exact, atol=1e-10)
    assert ev.circuit_runs == 1


def test_noise_leaks_probability_out_of_the_feasible_subspace() -> None:
    feasible = []
    for level in (0.0, 1.0, 5.0):
        ev, land = make(level)
        probs = ev.distribution(["xy_ring"], np.array([0.6]), np.array([0.4]))
        feasible.append(probs[land.feasible].sum())
    assert feasible[0] == pytest.approx(1.0)
    assert feasible[0] > feasible[1] > feasible[2]


@pytest.mark.parametrize(
    ("mixer", "optimizer", "layers"),
    [("x", "cobyla", 2), ("xy_ring", "spsa", 1), ("adapt", "cobyla", 2)],
)
def test_noisy_run_at_level_zero_equals_the_noiseless_run(
    mixer: str, optimizer: str, layers: int
) -> None:
    problem = random_problem(5, 2, 4.5, seed=4)
    bound = sufficient_penalty(problem)
    s = VariantSettings(
        mixer=mixer, optimizer=optimizer, layers=layers, shots=64, max_evaluations=30
    )
    clean = run_variant(problem, bound, s, np.random.default_rng(3))
    noisy = run_noisy_variant(
        problem, bound, s, 0.0, RATES, np.random.default_rng(3), backend=BACKEND
    )
    assert noisy.n_evaluations == clean.n_evaluations
    assert noisy.mixer_sequence == clean.mixer_sequence
    assert noisy.metrics.amplification_feasible == pytest.approx(
        clean.metrics.amplification_feasible, rel=1e-6
    )
    assert noisy.circuit_runs >= noisy.n_evaluations
    assert noisy.as_dict()["total_shots"] == noisy.optimization_shots + noisy.selection_shots

"""Tests for the NumPy QAOA simulator: it must agree with Qiskit and with the physics."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import Statevector

from helpers import random_problem
from nisq_folio.qaoa.circuits import qaoa_ansatz
from nisq_folio.qaoa.statevector import NoiselessQaoa, apply_x_mixer, hamiltonian_scale
from nisq_folio.qubo.encoding import IsingModel, all_bitstrings, build_qubo, qubo_to_ising


def setup(n: int, seed: int) -> tuple[IsingModel, np.ndarray, float]:
    qubo = build_qubo(random_problem(n, max(1, n // 2), 4.5, seed), 0.7)
    ising = qubo_to_ising(qubo)
    return ising, qubo.all_energies(), hamiltonian_scale(ising)


@settings(max_examples=30)
@given(
    st.integers(2, 6),
    st.integers(1, 3),
    st.integers(0, 10_000),
    st.lists(st.floats(-3, 3), min_size=6, max_size=6),
)
def test_matches_qiskit_circuit(n: int, layers: int, seed: int, angles: list[float]) -> None:
    """Same probabilities as the Qiskit circuit; circuit angles act on the unscaled H_C."""
    ising, energies, scale = setup(n, seed)
    gammas, betas = np.array(angles[:layers]), np.array(angles[3 : 3 + layers])
    circuit, g, b = qaoa_ansatz(ising, layers)
    bound = circuit.assign_parameters(
        dict(zip(g, gammas / scale, strict=True)) | dict(zip(b, betas, strict=True))
    )
    expected = np.abs(Statevector(bound).data) ** 2
    got = NoiselessQaoa(energies, scale).probabilities(gammas, betas)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_zero_angles_leave_the_uniform_superposition() -> None:
    _, energies, scale = setup(5, 1)
    probs = NoiselessQaoa(energies, scale).probabilities(np.zeros(2), np.zeros(2))
    np.testing.assert_allclose(probs, np.full(32, 1 / 32), atol=1e-15)


@given(st.integers(1, 6), st.integers(0, 1000))
def test_mixer_at_half_pi_flips_every_bit(n: int, seed: int) -> None:
    """exp(-i pi/2 X) = -i X, so the mixer maps basis state k to its complement (up to phase)."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2**n))
    state = np.zeros(2**n, dtype=complex)
    state[k] = 1.0
    out = apply_x_mixer(state, np.pi / 2, n)
    complement = (2**n - 1) ^ k
    assert abs(out[complement]) == pytest.approx(1.0)
    assert np.abs(np.delete(out, complement)).max() < 1e-12


@given(st.integers(1, 6), st.floats(-4, 4), st.integers(0, 1000))
def test_mixer_is_unitary(n: int, beta: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    state /= np.linalg.norm(state)
    assert np.linalg.norm(apply_x_mixer(state, beta, n)) == pytest.approx(1.0, abs=1e-12)


@given(st.integers(2, 7), st.integers(0, 10_000))
def test_scale_is_energy_std_under_uniform_spins(n: int, seed: int) -> None:
    """s^2 = sum J^2 + sum h^2 equals Var_z[E(z)] for z uniform on {+1,-1}^N."""
    ising, _, scale = setup(n, seed)
    spins = 1 - 2 * all_bitstrings(n)
    energies = np.array([ising.energy(z) for z in spins])
    assert scale == pytest.approx(energies.std(), rel=1e-10)


def test_invalid_inputs_are_rejected() -> None:
    _, energies, scale = setup(3, 0)
    with pytest.raises(ValueError, match="2\\^N"):
        NoiselessQaoa(np.zeros(6), 1.0)
    with pytest.raises(ValueError, match="scale"):
        NoiselessQaoa(energies, 0.0)
    with pytest.raises(ValueError, match="equal-length"):
        NoiselessQaoa(energies, scale).state(np.zeros(2), np.zeros(1))
    flat = IsingModel(j=np.zeros((2, 2)), h=np.zeros(2), offset=3.0)
    with pytest.raises(ValueError, match="no non-constant terms"):
        hamiltonian_scale(flat)

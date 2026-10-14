"""Tests for nisq_folio.qaoa.circuits: the circuit must implement the QAOA formula exactly."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm

from helpers import random_problem
from nisq_folio.qaoa.circuits import qaoa_ansatz
from nisq_folio.qubo.encoding import IsingModel, build_qubo, qubo_to_ising


def reference_state(ising: IsingModel, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """prod_l exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^N computed with dense matrices."""
    n = ising.n
    h_c = ising.to_sparse_pauli_op().to_matrix()
    h_m = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], n).to_matrix()
    state = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state = expm(-1j * beta * h_m) @ (expm(-1j * gamma * h_c) @ state)
    return state


@settings(max_examples=25)
@given(
    st.integers(2, 5),
    st.integers(1, 3),
    st.integers(0, 10_000),
    st.lists(st.floats(-3.0, 3.0), min_size=6, max_size=6),
)
def test_circuit_state_equals_qaoa_formula(n: int, layers: int, seed: int, angles: list) -> None:
    """Same state up to a global phase (from the dropped constant offset): |<a|b>| = 1."""
    problem = random_problem(n, max(1, n // 2), 4.5, seed)
    ising = qubo_to_ising(build_qubo(problem, 1.0))
    circuit, gammas, betas = qaoa_ansatz(ising, layers)
    g, b = np.array(angles[:layers]), np.array(angles[3 : 3 + layers])
    bound = circuit.assign_parameters(
        dict(zip(gammas, g, strict=True)) | dict(zip(betas, b, strict=True))
    )
    overlap = np.vdot(Statevector(bound).data, reference_state(ising, g, b))
    assert abs(overlap) == pytest.approx(1.0, abs=1e-9)


def test_parameters_gates_and_measurement() -> None:
    ising = qubo_to_ising(build_qubo(random_problem(4, 2, 4.5, seed=1), 1.0))
    circuit, gammas, betas = qaoa_ansatz(ising, 2)
    assert len(gammas) == len(betas) == 2
    assert set(circuit.parameters) == set(gammas) | set(betas)
    ops = circuit.count_ops()
    assert ops["rzz"] == 2 * 6  # all C(4, 2) couplings are nonzero for a dense Sigma
    assert ops["h"] == 4
    assert "measure" not in ops
    measured, _, _ = qaoa_ansatz(ising, 1, measure=True)
    assert measured.count_ops()["measure"] == 4


def test_zero_couplings_and_fields_are_skipped() -> None:
    j = np.zeros((3, 3))
    j[0, 2] = 0.5
    ising = IsingModel(j=j, h=np.array([0.0, 1.0, 0.0]), offset=0.0)
    ops = qaoa_ansatz(ising, 1)[0].count_ops()
    assert ops["rzz"] == 1
    assert ops["rz"] == 1


def test_layers_must_be_positive() -> None:
    ising = IsingModel(j=np.zeros((2, 2)), h=np.zeros(2), offset=0.0)
    with pytest.raises(ValueError, match="layers"):
        qaoa_ansatz(ising, 0)

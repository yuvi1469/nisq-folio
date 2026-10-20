"""Tunable noise model and exact noisy evaluation (DECISIONS.md ADR-023).

Noise level m scales the error rates of a reference device (default FakeKingston, IBM Heron):

    r_1q = m * median single-qubit (sx) error,   r_2q = m * median CZ error,
    r_ro = m * median readout error

so m = 0 is noiseless and m = 1 is roughly today's hardware. Gate errors are depolarizing with
parameter lambda = r * d / (d - 1) (d = 2^n), which makes the average gate infidelity equal r:
lambda = 2 r for single-qubit and 4 r / 3 for two-qubit gates. RZ is virtual (error-free) on IBM
hardware, as in Qiskit's device models. Readout error is a symmetric bit flip with probability
r_ro applied to the exact output distribution. Thermal relaxation and crosstalk are not modelled
(a simplification stated in ADR-023).

Circuits are transpiled once per mixer sequence onto the device's coupling map and basis
(cz, rz, sx, x), then bound per evaluation and simulated exactly as a density matrix; the
logical-to-physical qubit layout after routing is used to read out logical bit order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from nisq_folio.noise.circuits import variant_circuit
from nisq_folio.noise.models import fake_backend
from nisq_folio.problem import FloatArray
from nisq_folio.qubo.encoding import IsingModel

BASIS_GATES = ("cz", "rz", "sx", "x")
ROUTING_MODES = ("full_device", "compact_region")
"""How circuits are placed on the device (ADR-027).

``full_device``: the transpiler may use any of the device's qubits, so routing can pull in
ancilla qubits (more qubits to simulate: a 10-qubit circuit once needed 15, i.e. 16 GB).
``compact_region``: the circuit is confined to a fixed connected N-qubit region of the coupling
map (:func:`compact_region`), so exactly N qubits are simulated. With a qubit-uniform noise model
the choice of region does not bias results.
"""
ZERO_THRESHOLD = 1e-15


@dataclass(frozen=True)
class ReferenceRates:
    """Median error rates of the reference device (per gate / per measured bit)."""

    single_qubit: float
    two_qubit: float
    readout: float

    @classmethod
    def from_backend(cls, name: str) -> ReferenceRates:
        """Read median sx, cz and measure errors from a fake backend's calibration snapshot."""
        target = fake_backend(name).target

        def median(op: str) -> float:
            errors = [p.error for p in target[op].values() if p is not None and p.error is not None]
            if not errors:
                raise ValueError(f"backend {name} reports no errors for {op!r}")
            return float(np.median(errors))

        return cls(single_qubit=median("sx"), two_qubit=median("cz"), readout=median("measure"))


def depolarizing_parameter(error_rate: float, num_qubits: int) -> float:
    """Return lambda = r * d / (d - 1) with d = 2^n (module docstring)."""
    d = 2**num_qubits
    return error_rate * d / (d - 1)


def scaled_noise_model(level: float, rates: ReferenceRates) -> NoiseModel:
    """Depolarizing gate-noise model at noise level m (readout handled separately)."""
    if level < 0:
        raise ValueError(f"noise level must be >= 0, got {level}")
    model = NoiseModel(basis_gates=list(BASIS_GATES))
    if level == 0:
        return model
    one = depolarizing_error(depolarizing_parameter(level * rates.single_qubit, 1), 1)
    two = depolarizing_error(depolarizing_parameter(level * rates.two_qubit, 2), 2)
    model.add_all_qubit_quantum_error(one, ["sx", "x"])
    model.add_all_qubit_quantum_error(two, ["cz"])
    return model


def apply_readout_error(probabilities: FloatArray, flip: float) -> FloatArray:
    """Apply an independent symmetric bit flip with probability ``flip`` to every bit.

    p_out(y) = sum_x p(x) * prod_i [flip if x_i != y_i else 1 - flip]
    """
    if not 0.0 <= flip <= 0.5:
        raise ValueError(f"flip must be in [0, 0.5], got {flip}")
    p = np.asarray(probabilities, dtype=np.float64)
    n = int(np.log2(p.size))
    if flip == 0.0:
        return p.copy()
    confusion = np.array([[1 - flip, flip], [flip, 1 - flip]])
    tensor = p.reshape((2,) * n)
    for axis in range(n):
        tensor = np.moveaxis(np.tensordot(confusion, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def compact_region(coupling_map: CouplingMap, n: int) -> list[int]:
    """Return N physical qubits forming a connected region: breadth-first from qubit 0.

    Neighbours are visited in increasing index order, so the region is deterministic.
    """
    graph = coupling_map.graph.to_undirected(multigraph=False)
    if n > graph.num_nodes():
        raise ValueError(f"device has only {graph.num_nodes()} qubits, need {n}")
    region, frontier, seen = [], [0], {0}
    while frontier and len(region) < n:
        node = frontier.pop(0)
        region.append(node)
        for neighbour in sorted(graph.neighbors(node)):
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    if len(region) < n:
        raise ValueError(f"could not find a connected region of {n} qubits")
    return region


class NoisyEvaluator:
    """Exact output distributions of QAOA circuits under the scaled noise model."""

    def __init__(
        self,
        ising: IsingModel,
        scale: float,
        initial: str,
        k: int,
        level: float,
        rates: ReferenceRates,
        backend: str = "FakeKingston",
        optimization_level: int = 3,
        seed_transpiler: int = 0,
        max_memory_mb: int = 6000,
        max_parallel_threads: int = 0,
        routing: str = "full_device",
    ) -> None:
        """Prepare an evaluator for one problem and noise level.

        Args:
            ising: Cost Hamiltonian.
            scale: Hamiltonian scale s.
            initial: ``"plus"`` or ``"dicke"``.
            k: K, for the Dicke state.
            level: Noise level m.
            rates: Reference error rates.
            backend: Fake backend supplying the coupling map.
            optimization_level: Transpiler optimization level.
            seed_transpiler: Transpiler seed (CLAUDE.md rule 1).
            max_memory_mb: Aer memory cap.
            max_parallel_threads: Aer thread cap (0 = automatic).
            routing: One of :data:`ROUTING_MODES`.
        """
        self.ising, self.scale, self.initial, self.k = ising, scale, initial, k
        self.level, self.rates = level, rates
        if routing not in ROUTING_MODES:
            raise ValueError(f"routing must be one of {ROUTING_MODES}, got {routing!r}")
        self._coupling = fake_backend(backend).coupling_map
        if routing == "compact_region":
            self._coupling = self._coupling.reduce(compact_region(self._coupling, ising.n))
        self._optimization_level = optimization_level
        self._seed = seed_transpiler
        self._simulator = AerSimulator(
            method="density_matrix", noise_model=scaled_noise_model(level, rates)
        )
        self._simulator.set_options(
            max_memory_mb=max_memory_mb, max_parallel_threads=max_parallel_threads
        )
        self._cache: dict[
            tuple[str, ...], tuple[QuantumCircuit, ParameterVector, ParameterVector]
        ] = {}
        self.circuit_runs = 0

    def _compiled(
        self, mixers: tuple[str, ...]
    ) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        if mixers not in self._cache:
            gammas = ParameterVector("g", len(mixers))
            betas = ParameterVector("b", len(mixers))
            logical = variant_circuit(self.ising, self.scale, self.initial, self.k, mixers,
                                      list(gammas), list(betas))  # fmt: skip
            compiled = transpile(
                logical,
                coupling_map=self._coupling,
                basis_gates=list(BASIS_GATES),
                optimization_level=self._optimization_level,
                seed_transpiler=self._seed,
            )
            layout = (
                compiled.layout.final_index_layout()
                if compiled.layout
                else list(range(logical.num_qubits))
            )
            compiled.save_probabilities(layout)
            self._cache[mixers] = (compiled, gammas, betas)
        return self._cache[mixers]

    def two_qubit_gates(self, mixers: Sequence[str]) -> int:
        """CZ count of the compiled circuit for a mixer sequence."""
        return int(self._compiled(tuple(mixers))[0].count_ops().get("cz", 0))

    def distribution(
        self, mixers: Sequence[str], gammas: FloatArray, betas: FloatArray
    ) -> FloatArray:
        """Return the exact noisy output distribution (gate noise + readout), little-endian."""
        compiled, g, b = self._compiled(tuple(mixers))
        values = dict(zip(g, np.asarray(gammas, float), strict=True))
        values.update(zip(b, np.asarray(betas, float), strict=True))
        result = self._simulator.run(compiled.assign_parameters(values), shots=1).result()
        if not result.success:
            raise RuntimeError(f"noisy simulation failed: {result.status}")
        self.circuit_runs += 1
        probs = np.asarray(result.data()["probabilities"], dtype=np.float64)
        # Floating-point residue (~1e-16) in bins that are exactly zero in theory (e.g. infeasible
        # states under an XY mixer) is set to 0. Physically negligible (expected count < 1e-12 at
        # 1,024 shots), and it keeps NumPy's multinomial sampler consuming the same random numbers
        # as the noiseless path, so at m = 0 noisy and noiseless runs are identical (ADR-023).
        probs[probs < ZERO_THRESHOLD] = 0.0
        probs = apply_readout_error(probs / probs.sum(), self.level * self.rates.readout)
        return probs / probs.sum()

"""Runtime benchmark for noisy QAOA simulation (settles DECISIONS.md ADR-007).

Measures the wall time of ONE QAOA energy evaluation, i.e. one circuit run with S shots, which
is the unit of work an optimizer repeats ~100+ times per optimization. For each problem size N,
depth p, and simulation method:

- ``ideal_statevector``: noiseless reference, untranspiled circuit (what noiseless cells use).
- ``statevector``: backend noise model, one sampled noisy trajectory per shot.
- ``density_matrix``: backend noise model, exact mixed-state evolution (4^N memory).

Noisy circuits are transpiled to the fake backend's coupling map and basis gates, because the
noise model is defined per physical qubit and gate; the resulting depth and two-qubit gate
count are recorded since they drive both runtime and accumulated noise.
"""

from __future__ import annotations

import ctypes
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from nisq_folio.noise.models import fake_backend, noisy_simulator
from nisq_folio.problem import PortfolioProblem
from nisq_folio.qaoa.circuits import qaoa_ansatz
from nisq_folio.qubo.encoding import build_qubo, qubo_to_ising
from nisq_folio.qubo.penalty import sufficient_penalty

BENCHMARK_METHODS = ("ideal_statevector", "statevector", "density_matrix")


@dataclass(frozen=True)
class BenchmarkSettings:
    """Settings for :func:`benchmark_case`.

    Attributes:
        backend: Fake backend class name, e.g. ``"FakeKingston"``.
        shots: Shots per energy evaluation.
        repeats: Timed repetitions per case (the median is reported).
        optimization_level: Qiskit transpiler optimization level (0-3).
        max_memory_mb: Aer memory cap; larger simulations are refused instead of swapping.
        penalty_margin: Margin for the QUBO penalty rule (ADR-017).
        max_seconds_per_run: Stop repeating a case as soon as one run takes longer than this
            (flagged ``exceeded_time_limit``); the default never stops early.
    """

    backend: str
    shots: int
    repeats: int
    optimization_level: int
    max_memory_mb: int
    penalty_margin: float
    max_seconds_per_run: float = float("inf")


def bound_qaoa_circuit(
    problem: PortfolioProblem, layers: int, rng: np.random.Generator, penalty_margin: float
) -> QuantumCircuit:
    """Build a measured depth-p QAOA circuit for ``problem`` with random angles bound.

    Angles are drawn as gamma_l ~ U(0, pi) and beta_l ~ U(0, pi/2); runtime does not depend on
    their values, but binding real numbers lets the transpiler treat the circuit realistically.

    Args:
        problem: The portfolio problem.
        layers: QAOA depth p.
        rng: Seeded generator for the angles.
        penalty_margin: Margin for :func:`sufficient_penalty`.

    Returns:
        The bound, measured circuit on N qubits.
    """
    ising = qubo_to_ising(build_qubo(problem, sufficient_penalty(problem, penalty_margin)))
    circuit, gammas, betas = qaoa_ansatz(ising, layers, measure=True)
    values = dict(zip(gammas, rng.uniform(0.0, np.pi, layers), strict=True))
    values.update(zip(betas, rng.uniform(0.0, np.pi / 2, layers), strict=True))
    return circuit.assign_parameters(values)


def benchmark_case(
    circuit: QuantumCircuit, method: str, settings: BenchmarkSettings, seed: int
) -> dict[str, Any]:
    """Time one energy evaluation of ``circuit`` with the given simulation method.

    Args:
        circuit: Bound, measured logical circuit.
        method: One of :data:`BENCHMARK_METHODS`.
        settings: Benchmark settings.
        seed: Used for ``seed_transpiler`` and ``seed_simulator`` (CLAUDE.md rule 1).

    Returns:
        A JSON-serializable record: status (``"ok"``, or ``"refused"`` with the error message),
        transpiled depth and two-qubit gate count, per-repeat and median seconds, and
        ``exceeded_time_limit`` (True if a run took longer than ``max_seconds_per_run``, in
        which case the remaining repeats were skipped).
    """
    if method not in BENCHMARK_METHODS:
        raise ValueError(f"method must be one of {BENCHMARK_METHODS}, got {method!r}")
    if method == "ideal_statevector":
        simulator = AerSimulator(method="statevector", seed_simulator=seed)
        compiled = transpile(circuit, simulator, seed_transpiler=seed, optimization_level=1)
    else:
        backend = fake_backend(settings.backend)
        simulator = noisy_simulator(backend, method, seed)
        compiled = transpile(
            circuit,
            backend,
            seed_transpiler=seed,
            optimization_level=settings.optimization_level,
        )
    simulator.set_options(max_memory_mb=settings.max_memory_mb)
    ops = compiled.count_ops()
    record: dict[str, Any] = {
        "method": method,
        "depth": compiled.depth(),
        "two_qubit_gates": int(sum(n for g, n in ops.items() if g in ("cz", "ecr", "cx", "rzz"))),
        "seconds": [],
    }
    for _ in range(settings.repeats):
        start = time.perf_counter()
        result = simulator.run(compiled, shots=settings.shots).result()
        elapsed = time.perf_counter() - start
        if not result.success:
            record.update(status="refused", error=str(result.status))
            return record
        record["seconds"].append(elapsed)
        if elapsed > settings.max_seconds_per_run:
            break  # one run already exceeds the budget; repeating it only costs more time
    exceeded = max(record["seconds"]) > settings.max_seconds_per_run
    record.update(
        status="ok",
        median_seconds=float(np.median(record["seconds"])),
        exceeded_time_limit=exceeded,
    )
    return record


def machine_info() -> dict[str, Any]:
    """Describe the machine, since wall times are only meaningful relative to the hardware."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    info["total_ram_gb"] = _total_ram_gb()
    return info


def _total_ram_gb() -> float | None:
    """Best-effort total physical memory in GB (None if it cannot be determined)."""
    if os.name == "nt":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 1024**3, 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, ValueError, OSError):
        return None

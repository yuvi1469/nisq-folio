"""Noisy simulators built from IBM fake backends (offline; no account needed, ADR-008)."""

from __future__ import annotations

from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

SIMULATION_METHODS = ("density_matrix", "statevector")
"""Supported Aer methods for noisy runs.

``density_matrix`` evolves the full mixed state rho (4^N entries), so noise is exact and one
run yields exact outcome probabilities. ``statevector`` with a noise model samples one noisy
trajectory per shot (2^N entries each), so cost grows with shots and estimates carry
trajectory sampling error.
"""


def fake_backend(name: str) -> BackendV2:
    """Return an IBM fake backend by class name, e.g. ``"FakeKingston"``.

    Args:
        name: Class name in ``qiskit_ibm_runtime.fake_provider``.

    Returns:
        A new backend instance carrying that device's calibration snapshot.

    Raises:
        ValueError: If no such fake backend exists.
    """
    cls = getattr(fake_provider, name, None)
    if cls is None or not name.startswith("Fake"):
        raise ValueError(f"unknown fake backend {name!r}")
    return cls()


def noisy_simulator(backend: BackendV2, method: str, seed: int) -> AerSimulator:
    """Return an Aer simulator with the backend's noise model, coupling map and basis gates.

    Args:
        backend: The fake backend to mimic.
        method: One of :data:`SIMULATION_METHODS`.
        seed: ``seed_simulator`` for Aer (CLAUDE.md rule 1).

    Returns:
        The configured simulator.
    """
    if method not in SIMULATION_METHODS:
        raise ValueError(f"method must be one of {SIMULATION_METHODS}, got {method!r}")
    return AerSimulator.from_backend(backend, method=method, seed_simulator=seed)

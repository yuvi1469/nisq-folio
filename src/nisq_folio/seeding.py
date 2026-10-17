"""Derive independent, reproducible random streams from the config's root seed (ADR-004).

Every stochastic component gets its own stream, identified by a tuple of integer keys:

    rng = child_rng(root_seed, STREAM_X, n, replicate, ...)

which is ``numpy.random.default_rng(SeedSequence(entropy=root_seed, spawn_key=keys))``. The same
keys always give the same stream, different keys give statistically independent streams, and no
global RNG state is touched.
"""

from __future__ import annotations

import numpy as np

# Stream identifiers: the first key of every stream. Never renumber an existing stream,
# or previously reported results stop being reproducible; only append new ones.
STREAM_INSTANCES = 0
STREAM_SIMULATED_ANNEALING = 1
STREAM_NOISE_BENCHMARK = 2
STREAM_TUNING_INSTANCES = 3  # instances used only for choosing QAOA settings (ADR-020)
STREAM_QAOA_SHOTS = 4
STREAM_MIXER_STUDY = 5
STREAM_EXPERIMENT = 6


def child_rng(root_seed: int, *keys: int) -> np.random.Generator:
    """Return the random generator for stream ``keys`` under ``root_seed``.

    Args:
        root_seed: The config's root seed (non-negative).
        *keys: Non-negative integers identifying the stream, e.g. (STREAM_INSTANCES, n).

    Returns:
        A new, independently seeded Generator.
    """
    if root_seed < 0 or any(k < 0 for k in keys):
        raise ValueError("root_seed and keys must be non-negative integers")
    return np.random.default_rng(np.random.SeedSequence(entropy=root_seed, spawn_key=keys))

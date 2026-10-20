"""Tests for compact-region routing (ADR-027).

Regression for the Week 12 failure: under full-device routing a 10-qubit ADAPT circuit was
routed through 15 physical qubits and needed 16 GB. Compact routing must use exactly N qubits,
stay exact at m = 0, and be selectable from the experiment config.
"""

from pathlib import Path

import numpy as np
import pytest

from helpers import random_problem
from nisq_folio.config import ROUTING_MODES as CONFIG_ROUTING_MODES
from nisq_folio.config import ConfigError, load_config
from nisq_folio.mixers.variants import VariantSettings, adapt_pool, evolve, run_variant
from nisq_folio.mixers.xy import dicke_state
from nisq_folio.noise.density import ROUTING_MODES, ReferenceRates, compact_region
from nisq_folio.noise.models import fake_backend
from nisq_folio.noise.runner import make_evaluator, run_noisy_variant
from nisq_folio.qubo.penalty import sufficient_penalty
from test_experiment_runner import small_config

RATES = ReferenceRates(single_qubit=2e-4, two_qubit=2e-3, readout=1e-2)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_routing_mode_lists_agree() -> None:
    assert CONFIG_ROUTING_MODES == ROUTING_MODES


@pytest.mark.parametrize("n", [2, 5, 8, 12, 14])
def test_compact_region_is_connected_deterministic_and_sized(n: int) -> None:
    coupling = fake_backend("FakeKingston").coupling_map
    region = compact_region(coupling, n)
    assert len(region) == len(set(region)) == n
    assert compact_region(coupling, n) == region
    reduced = coupling.reduce(region)  # raises if the region is not connected
    assert reduced.size() == n
    with pytest.raises(ValueError):
        compact_region(coupling, 10_000)


def test_far_apart_adapt_pairs_use_exactly_n_qubits_and_stay_exact() -> None:
    """The sequence shape that failed in Week 12 (pairs across the ring) at N = 6, m = 0."""
    problem = random_problem(6, 3, 4.5, seed=3)
    settings = VariantSettings(mixer="adapt", optimizer="cobyla", layers=3)
    evaluator, energies = make_evaluator(
        problem, sufficient_penalty(problem), settings, 0.0, RATES, routing="compact_region"
    )
    seq, g, b = (
        ["xy_0_5", "xy_1_4", "xy_ring"],
        np.array([0.5, 0.9, 0.3]),
        np.array([0.3, -0.2, 0.4]),
    )
    pool = adapt_pool(6)
    exact = (
        np.abs(evolve(dicke_state(6, 3), energies / energies.std(), g, b, [pool[m] for m in seq]))
        ** 2
    )
    np.testing.assert_allclose(evaluator.distribution(seq, g, b), exact, atol=1e-10)
    assert evaluator._compiled(tuple(seq))[0].num_qubits == 6


def test_noisy_run_with_compact_routing_equals_noiseless_at_level_zero() -> None:
    problem = random_problem(5, 2, 4.5, seed=4)
    bound = sufficient_penalty(problem)
    s = VariantSettings(mixer="adapt", optimizer="cobyla", layers=2, shots=64, max_evaluations=30)
    clean = run_variant(problem, bound, s, np.random.default_rng(3))
    noisy = run_noisy_variant(
        problem, bound, s, 0.0, RATES, np.random.default_rng(3), routing="compact_region"
    )
    assert noisy.mixer_sequence == clean.mixer_sequence
    assert noisy.metrics.amplification_feasible == pytest.approx(
        clean.metrics.amplification_feasible, rel=1e-6
    )


def test_invalid_routing_is_rejected(tmp_path: Path) -> None:
    problem = random_problem(4, 2, 4.5, seed=0)
    s = VariantSettings(mixer="xy_ring", optimizer="spsa", layers=1)
    with pytest.raises(ValueError, match="routing"):
        make_evaluator(problem, sufficient_penalty(problem), s, 0.0, RATES, routing="anywhere")
    with pytest.raises(ConfigError, match="routing"):
        load_config(small_config(tmp_path, routing="anywhere"))
    assert (
        load_config(small_config(tmp_path, name="ok")).require_experiment().routing == "full_device"
    )


def test_main_config_uses_compact_routing_under_a_new_name() -> None:
    experiment = load_config(REPO_ROOT / "configs" / "main.yaml").require_experiment()
    assert experiment.routing == "compact_region"
    assert experiment.name == "main_r2"

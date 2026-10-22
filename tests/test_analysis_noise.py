"""Tests for the pre-registered noisy analysis (ADR-028): crossover, bootstrap, tables, CLI."""

import shutil
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nisq_folio.analysis import noise_report
from nisq_folio.analysis.noise import (
    analyze,
    bootstrap_indices,
    complete_cells,
    crossover,
    routing_robust,
    routing_table,
    verdicts,
)
from nisq_folio.config import load_config
from nisq_folio.data.prices import load_or_download
from nisq_folio.experiments import __main__ as runner_cli
from nisq_folio.provenance import GitState
from test_experiment_runner import small_config

LEVELS = [0.0, 0.25, 0.5, 1.0]


def test_crossover_known_values() -> None:
    assert crossover(LEVELS, [4, 2, 1, 0.5]) == pytest.approx(0.5)  # lands exactly on a level
    assert crossover(LEVELS, [4, 0.5, 0.4, 0.3]) == pytest.approx(0.25 * 2 / 3)  # linear from 0
    assert crossover([0, 0.25, 1], [4, 2, 0.5]) == pytest.approx(0.5)  # geometric midpoint
    assert crossover(LEVELS, [0.9, 2, 2, 2]) == 0.0  # no advantage without noise
    assert crossover(LEVELS, [3, 2, 1.5, 1.1]) == np.inf  # censored
    assert crossover(LEVELS, [3, 0.5, 2, 0.5]) == pytest.approx(0.25 * np.log(3) / np.log(6))
    with pytest.raises(ValueError, match="increasing"):
        crossover([0, 0.5, 0.25], [2, 1, 0.5])
    with pytest.raises(ValueError, match="increasing"):
        crossover([0, 1], [2])


@given(
    st.lists(st.floats(1e-3, 1e3), min_size=2, max_size=5),
    st.lists(st.floats(1e-3, 1e3), min_size=6, max_size=6),
)
def test_crossover_lies_in_its_bracket(steps: list[float], medians: list[float]) -> None:
    levels = np.concatenate([[0.0], np.cumsum(np.array(steps) / 100)])
    med = np.array(medians[: len(levels)])
    m_star = crossover(levels, med)
    below = np.flatnonzero(med <= 1)
    if med[0] <= 1:
        assert m_star == 0
    elif below.size == 0:
        assert np.isinf(m_star)
    else:
        j = below[0]
        assert levels[j - 1] - 1e-12 <= m_star <= levels[j] * (1 + 1e-12)


def synthetic(rates: dict[str, float], sizes: Sequence[int] = (8, 10), seed: int = 0) -> tuple:
    """A noisy grid where A = a0 * exp(-rate * m) * lognormal scatter, for every cell."""
    rng = np.random.default_rng(seed)
    rows, classical = [], []
    for n in sizes:
        for i in range(30):
            classical.append({"n": n, "instance": i, "denoising": "none",
                              "optimum_oos_variance": 1.0})  # fmt: skip
            for den in ("none", "rmt_clip"):
                for mixer, rate in rates.items():
                    scatter = rng.lognormal(0, 0.3)
                    for m in LEVELS:
                        rows.append({
                            "n": n, "instance": i, "denoising": den, "mixer": mixer,
                            "noise_level": m, "A": 4 * np.exp(-rate * m) * scatter,
                            "p_optimal": 0.1, "p_feasible": 1 - 0.3 * m, "expected_gap": m,
                            "total_shots": 1000, "mode_optimal": m < 0.5,
                            "mode_oos_variance": 1 + m,
                        })  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(classical)


def test_analyze_recovers_known_crossovers() -> None:
    rates = {"adapt": 1.0, "xy_ring": 2.0, "x": 8.0}
    qaoa, classical = synthetic(rates)
    tables = analyze(qaoa, classical, root_seed=3, replicates=300)
    cross = tables["N2"].set_index(["denoising", "mixer", "n"])
    for mixer, rate in rates.items():
        truth = np.log(4) / rate  # median scatter is ~1, so M(m) = 1 at m = ln 4 / rate
        row = cross.loc[("none", mixer, 8)]
        assert row["ci_low"] <= row["crossover"] <= row["ci_high"]
        if truth < 1:
            assert row["crossover"] == pytest.approx(truth, rel=0.35)
    assert np.isinf(cross.loc[("none", "adapt", 8), "crossover"])  # ln 4 > 1: censored
    assert cross.loc[("none", "adapt", 8), "share_censored"] > 0.5
    comp = tables["N3"].set_index(["denoising", "n", "comparison"])["p_greater"]
    assert comp[("none", 8, "adapt > x")] > 0.99 and comp[("none", 8, "xy_ring > x")] > 0.99
    assert ("none", 8, "x: N=8 > N=10") in comp.index
    assert len(tables["N1"]) == 2 * 3 * 2 * len(LEVELS)
    assert (tables["N1"]["A_ci_low"] <= tables["N1"]["A_median"]).all()
    assert len(tables["N4"]) == 2 * 3 * 2 * 3 and (tables["N4"]["ratio"] < 1).all()
    assert set(tables["N5"]["noise_level"]) == set(LEVELS)
    assert set(tables["N6"]["mixer"]) == set(rates)
    n7 = tables["N7"].set_index(["mixer", "n", "noise_level"])
    assert n7.loc[("x", 8, 1.0), "oos_variance_ratio_median"] == pytest.approx(2.0)


def test_analyze_is_reproducible_and_seed_dependent() -> None:
    qaoa, classical = synthetic({"xy_ring": 2.0, "x": 8.0})
    a = analyze(qaoa, classical, root_seed=3, replicates=200)["N2"]
    b = analyze(qaoa, classical, root_seed=3, replicates=200)["N2"]
    c = analyze(qaoa, classical, root_seed=4, replicates=200)["N2"]
    pd.testing.assert_frame_equal(a, b)
    assert not a[["ci_low", "ci_high"]].equals(c[["ci_low", "ci_high"]])
    np.testing.assert_array_equal(bootstrap_indices(3, 8, 30, 5), bootstrap_indices(3, 8, 30, 5))
    assert bootstrap_indices(3, 8, 30, 5).max() < 30


def test_incomplete_cells_are_dropped() -> None:
    qaoa, classical = synthetic({"xy_ring": 2.0, "x": 8.0}, sizes=(8,))
    partial = qaoa.drop(qaoa[(qaoa["mixer"] == "x") & (qaoa["noise_level"] == 1.0)].index[:5])
    kept = complete_cells(partial)
    assert set(kept["mixer"]) == {"xy_ring"}
    noiseless_only = qaoa[qaoa["noise_level"] == 0]
    assert complete_cells(noiseless_only).empty
    assert all(t.empty for t in analyze(noiseless_only, classical, 1, 10).values())


def test_verdicts_follow_the_two_thirds_rule() -> None:
    curve = pd.DataFrame({
        "denoising": ["none"] * 3, "mixer": ["x"] * 3, "n": [8] * 3, "noise_level": [0, 0.5, 1],
        "A_median": [1.5, 1.5, 0.9], "share_A_gt_1": [0.7, 0.6, 0.9],
    })  # fmt: skip
    assert list(verdicts(curve)["verdict"]) == ["beats random", "not supported", "not supported"]


def test_routing_check_on_identical_and_shifted_runs() -> None:
    qaoa, classical = synthetic({"xy_ring": 2.0, "x": 8.0}, sizes=(8,))
    same = routing_table(qaoa, qaoa)
    assert len(same) == 2 * 2 * 3 and np.allclose(same["ratio"], 1.0)
    shifted = qaoa.assign(A=qaoa["A"] * 2)
    assert np.allclose(routing_table(qaoa, shifted)["ratio"], 2.0)
    cross = analyze(qaoa, classical, 1, 100)["N2"]
    assert routing_robust(cross, cross)["inside_ci"].all()
    far = cross.assign(crossover=cross["ci_high"] + 1)
    assert not routing_robust(cross, far)["inside_ci"].any()


def test_noise_report_cli_on_a_real_tiny_experiment(
    tmp_path: Path, make_prices: Callable[..., pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    path = small_config(tmp_path, name="tiny", routing="compact_region")
    config = load_config(path)
    u = config.universe

    def fetcher(t: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return make_prices(t, date(2020, 1, 2), 250, seed=5)

    load_or_download(u.tickers, u.start_date, u.end_date, config.data_dir,
                     GitState(None, False), fetcher=fetcher, source_version="test")  # fmt: skip
    assert noise_report.main(["--config", str(path)]) == 1  # nothing finished yet
    assert runner_cli.main(["--config", str(path)]) == 0
    experiments = config.results_dir / "experiments"
    shutil.copytree(experiments / "tiny", experiments / "old")

    assert noise_report.main(["--config", str(path), "--superseded", "old"]) == 0
    out = capsys.readouterr().out
    for key in ("N1.", "N2.", "N3.", "N4.", "N5.", "N6.", "N7.", "N8.", "Verdicts"):
        assert key in out
    report = (config.results_dir / "analysis" / "tiny_noise.md").read_text(encoding="utf-8")
    assert report.count("## N") == 9 and "| True |" in report

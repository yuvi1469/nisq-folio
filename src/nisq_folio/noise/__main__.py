"""Run the noisy-simulation runtime benchmark and write a run record.

For each N, uses the first seeded validation instance (same instances as the baselines), and
for each depth p and method times one energy evaluation. Larger N for a method are skipped once
that method exceeds ``max_seconds_per_run`` or is refused for memory, since cost only grows
with N.

Usage::

    uv run python -m nisq_folio.noise --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import qiskit
import qiskit_aer

from nisq_folio.config import load_config
from nisq_folio.data.prices import PriceFetcher, fetch_yfinance, load_or_download
from nisq_folio.data.returns import simple_returns, split_returns
from nisq_folio.experiments.instances import build_problem, sample_instance_tickers
from nisq_folio.experiments.records import write_run_record
from nisq_folio.noise.benchmark import (
    BenchmarkSettings,
    benchmark_case,
    bound_qaoa_circuit,
    machine_info,
)
from nisq_folio.provenance import git_state
from nisq_folio.seeding import STREAM_NOISE_BENCHMARK, child_rng


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the benchmark for a config file, print a table, and write a run record.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source, used only if the data cache is missing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.noise", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problem_cfg, bench = config.require_problem(), config.require_noise_benchmark()
    settings = BenchmarkSettings(
        backend=bench.backend,
        shots=bench.shots,
        repeats=bench.repeats,
        optimization_level=bench.optimization_level,
        max_memory_mb=bench.max_memory_mb,
        penalty_margin=config.require_qubo().penalty_margin,
    )
    git = git_state(config.source_path.parent)
    u = config.universe
    prices, data_meta = load_or_download(
        u.tickers, u.start_date, u.end_date, config.data_dir, git=git, fetcher=fetcher
    )
    in_sample, _ = split_returns(simple_returns(prices), problem_cfg.in_sample_end)

    cases: list[dict[str, Any]] = []
    stopped: set[tuple[str, int]] = set()
    print(f"backend {bench.backend}, {bench.shots} shots, median of {bench.repeats} repeats")
    print(f"{'N':>3} {'p':>2}  {'method':<18}{'depth':>7}{'2q gates':>9}{'seconds':>10}  status")
    for n in bench.sizes:
        tickers = sample_instance_tickers(u.tickers, n, 1, config.seed)[0]
        problem = build_problem(in_sample, tickers, n // 2, problem_cfg.risk_aversion)
        for layers in bench.layers:
            rng = child_rng(config.seed, STREAM_NOISE_BENCHMARK, n, layers)
            circuit = bound_qaoa_circuit(problem, layers, rng, settings.penalty_margin)
            for method in bench.methods:
                if (method, layers) in stopped:
                    case = {"method": method, "status": "skipped (smaller N exceeded limits)"}
                else:
                    seed = int(rng.integers(2**31))
                    case = benchmark_case(circuit, method, settings, seed)
                    too_slow = case.get("median_seconds", 0.0) > bench.max_seconds_per_run
                    if case["status"] != "ok" or too_slow:
                        stopped.add((method, layers))
                case.update(n=n, layers=layers)
                cases.append(case)
                secs = f"{case['median_seconds']:.3f}" if "median_seconds" in case else "-"
                print(
                    f"{n:>3} {layers:>2}  {method:<18}{case.get('depth', '-'):>7}"
                    f"{case.get('two_qubit_gates', '-'):>9}{secs:>10}  {case['status']}"
                )

    results = {
        "machine": machine_info(),
        "versions": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__},
        "settings": settings.__dict__,
        "cases": cases,
    }
    path = write_run_record(config, "noise_benchmark", git, data_meta, results)
    print(f"\nrun record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

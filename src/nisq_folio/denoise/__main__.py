"""Evaluate covariance estimators out of sample on the evaluation instances (ADR-022).

Classical exact optimization only (no QAOA). Prints the universe-level cleaning report and,
per N, each estimator's paired comparison against the sample covariance; writes a run record.

Usage::

    uv run python -m nisq_folio.denoise --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np

from nisq_folio.config import load_config
from nisq_folio.data.prices import PriceFetcher, fetch_yfinance, load_or_download
from nisq_folio.data.returns import annualized_moments, simple_returns, split_returns
from nisq_folio.denoise.evaluation import (
    ESTIMATORS,
    correlation_distance,
    evaluate_instance,
    paired_comparison,
    universe_covariances,
)
from nisq_folio.experiments.instances import sample_instance_tickers
from nisq_folio.experiments.records import write_run_record
from nisq_folio.provenance import git_state

METRICS = ("realized_variance", "realized_objective")


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the evaluation for a config file, print a summary, and write a run record.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source, used only if the data cache is missing.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.denoise", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problem_cfg, denoise_cfg = config.require_problem(), config.require_denoise()
    git = git_state(config.source_path.parent)
    u = config.universe
    prices, data_meta = load_or_download(
        u.tickers, u.start_date, u.end_date, config.data_dir, git=git, fetcher=fetcher
    )
    in_sample, out_of_sample = split_returns(simple_returns(prices), problem_cfg.in_sample_end)
    covariances, reports = universe_covariances(in_sample)
    _, sigma_oos_universe = annualized_moments(out_of_sample)

    clip = reports["rmt_clip"]
    print(f"universe: N={len(u.tickers)}, T={len(in_sample)}, q={clip['q']:.4f}, "
          f"MP band [{clip['lambda_minus']:.3f}, {clip['lambda_plus']:.3f}]")  # fmt: skip
    print(f"  rmt_clip keeps {clip['n_kept']} eigenvalues, replaces {clip['n_replaced']}; "
          f"rmt_band_only keeps {reports['rmt_band_only']['n_kept']}; "
          f"Ledoit-Wolf shrinkage {reports['ledoit_wolf']['shrinkage']:.3f}")  # fmt: skip
    distances = {e: correlation_distance(covariances[e], sigma_oos_universe) for e in ESTIMATORS}
    print("  correlation distance to out-of-sample (Frobenius): "
          + ", ".join(f"{e} {d:.3f}" for e, d in distances.items()))  # fmt: skip

    results: dict[str, Any] = {"reports": reports, "correlation_distance": distances, "sizes": {}}
    for n in problem_cfg.sizes:
        instances = sample_instance_tickers(
            u.tickers, n, denoise_cfg.evaluation_instances_per_size, config.seed
        )
        records = [
            evaluate_instance(in_sample, out_of_sample, t, n // 2, problem_cfg.risk_aversion,
                              covariances)
            for t in instances
        ]  # fmt: skip
        comparisons: dict[str, Any] = {}
        print(f"\nN={n} ({len(instances)} evaluation instances), vs sample covariance:")
        for estimator in ESTIMATORS[1:]:
            comparisons[estimator] = {}
            for metric in METRICS:
                cand = np.array([r[estimator][metric] for r in records])
                ref = np.array([r["sample"][metric] for r in records])
                comparisons[estimator][metric] = paired_comparison(cand, ref)
            ratio = np.median([r[estimator]["risk_ratio"] for r in records])
            v = comparisons[estimator]["realized_variance"]
            print(
                f"  {estimator:<14} realized variance: median diff "
                f"{v['median_difference']:+.5f}, lower in {v['share_lower']:.0%}, "
                f"same selection {v['share_identical']:.0%}, p={v['wilcoxon_p']:.3g} | "
                f"median risk ratio {ratio:.2f}"
            )
        sample_ratio = np.median([r["sample"]["risk_ratio"] for r in records])
        print(f"  {'sample':<14} median risk ratio {sample_ratio:.2f}")
        results["sizes"][str(n)] = {"comparisons": comparisons, "instances": records}

    path = write_run_record(config, "denoise_evaluation", git, data_meta, results)
    print(f"\nrun record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

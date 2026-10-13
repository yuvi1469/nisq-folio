"""Download (or load from cache) the configured universe and print a data summary.

Usage::

    uv run python -m nisq_folio.data --config configs/base.yaml [--refresh]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from nisq_folio.config import load_config
from nisq_folio.data.prices import (
    PriceFetcher,
    cache_key,
    cache_paths,
    fetch_yfinance,
    load_or_download,
)
from nisq_folio.data.returns import annualized_moments, simple_returns
from nisq_folio.provenance import git_state


def main(argv: Sequence[str] | None = None, fetcher: PriceFetcher = fetch_yfinance) -> int:
    """Run the data step for a config file and print a summary.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetcher: Price source; tests inject a fake to avoid network access.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="python -m nisq_folio.data", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    universe = config.universe
    prices, meta = load_or_download(
        universe.tickers,
        universe.start_date,
        universe.end_date,
        config.data_dir,
        git=git_state(config.source_path.parent),
        fetcher=fetcher,
        refresh=args.refresh,
    )
    mu, sigma = annualized_moments(simple_returns(prices))
    vol = sigma.to_numpy().diagonal() ** 0.5

    parquet, _ = cache_paths(
        config.data_dir, cache_key(universe.tickers, universe.start_date, universe.end_date)
    )
    print(f"cache:      {parquet}")
    print(f"source:     {meta.source} [{meta.source_version}]")
    print(f"downloaded: {meta.downloaded_at}  (git {meta.git_commit}, dirty={meta.git_dirty})")
    print(f"sha256:     {meta.sha256}")
    print(f"window:     {meta.first_date} .. {meta.last_date}  ({meta.n_rows} trading days)")
    print(f"assets:     {len(prices.columns)} across {len(universe.sectors)} sectors")
    print()
    print(f"{'sector':<24}{'ticker':<8}{'ann. mean':>10}{'ann. vol':>10}")
    sector_of = universe.sector_of
    for i, ticker in enumerate(prices.columns):
        print(f"{sector_of[ticker]:<24}{ticker:<8}{mu[ticker]:>10.3f}{vol[i]:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the ``python -m nisq_folio.data`` entry point (network faked)."""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nisq_folio.data.__main__ import main


def write_config(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    path = cfg_dir / "small.yaml"
    content = {
        "paths": {"data_dir": "../data", "results_dir": "../results"},
        "seed": 1,
        "universe": {
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "n_stocks": 2,
            "n_sectors": 2,
            "sectors": {"Tech": ["AAA"], "Energy": ["BBB"]},
        },
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def test_main_downloads_once_then_reads_cache(
    tmp_path: Path,
    make_prices: Callable[..., pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fetcher(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        calls.append((tuple(tickers), start, end))
        return make_prices(tickers, date(2021, 1, 4), 100, seed=3)

    config = write_config(tmp_path)
    assert main(["--config", str(config)], fetcher=fetcher) == 0
    assert main(["--config", str(config)], fetcher=fetcher) == 0

    assert calls == [(("AAA", "BBB"), date(2021, 1, 1), date(2021, 12, 31))]
    assert (tmp_path / "data").is_dir()
    out = capsys.readouterr().out
    assert "100 trading days" in out
    assert "2 across 2 sectors" in out
    assert "Tech" in out and "AAA" in out

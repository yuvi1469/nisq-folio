"""Tests for nisq_folio.data.prices (no network access: the price source is faked)."""

import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nisq_folio.data.prices import (
    CacheIntegrityError,
    DataValidationError,
    cache_key,
    cache_paths,
    fetch_yfinance,
    load_or_download,
    validate_prices,
)
from nisq_folio.provenance import GitState

TICKERS = ("AAA", "BBB", "CCC")
START, END = date(2021, 1, 4), date(2021, 3, 31)
GIT = GitState(commit="0" * 40, dirty=False)


class CountingFetcher:
    """Fake price source that records how often it is called."""

    def __init__(self, prices: pd.DataFrame) -> None:
        """Serve copies of ``prices`` on every call."""
        self.prices = prices
        self.calls = 0

    def __call__(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Return the stored prices and count the call."""
        self.calls += 1
        return self.prices.copy()


@pytest.fixture
def good_prices(make_prices: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_prices(TICKERS, START, 40, seed=1)


# ---------------------------------------------------------------- validate_prices


def test_valid_panel_passes(good_prices: pd.DataFrame) -> None:
    validate_prices(good_prices, TICKERS, START, END)


def corrupt(prices: pd.DataFrame, kind: str) -> pd.DataFrame:
    p = prices.copy()
    if kind == "nan":
        p.iloc[5, 1] = np.nan
    elif kind == "zero":
        p.iloc[5, 2] = 0.0
    elif kind == "inf":
        p.iloc[5, 0] = np.inf
    elif kind == "order":
        p = p[list(reversed(TICKERS))]
    elif kind == "duplicate_dates":
        p = pd.concat([p.iloc[:1], p])
    elif kind == "unsorted":
        p = p.iloc[::-1]
    elif kind == "before_window":
        p.index = p.index - pd.Timedelta(days=30)
    elif kind == "not_datetime":
        p.index = range(len(p))
    elif kind == "empty":
        p = p.iloc[:0]
    return p


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("nan", r"missing prices .*'BBB': 1"),
        ("zero", r"non-positive or non-finite prices for: \['CCC'\]"),
        ("inf", r"non-positive or non-finite prices for: \['AAA'\]"),
        ("order", "do not match expected tickers"),
        ("duplicate_dates", "unique and strictly increasing"),
        ("unsorted", "unique and strictly increasing"),
        ("before_window", "outside window"),
        ("not_datetime", "DatetimeIndex"),
        ("empty", "empty"),
    ],
)
def test_invalid_panels_are_rejected(good_prices: pd.DataFrame, kind: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_prices(corrupt(good_prices, kind), TICKERS, START, END)


# ---------------------------------------------------------------- cache_key

ticker_lists = st.lists(st.text("ABCDEFGHIJ", min_size=1, max_size=5), min_size=1, unique=True)


@given(ticker_lists)
def test_cache_key_is_deterministic_and_short(tickers: list[str]) -> None:
    key = cache_key(tickers, START, END)
    assert key == cache_key(list(tickers), START, END)
    assert len(key) == 12
    int(key, 16)


@given(ticker_lists, ticker_lists)
def test_cache_key_differs_when_request_differs(a: list[str], b: list[str]) -> None:
    if a != b:
        assert cache_key(a, START, END) != cache_key(b, START, END)
    assert cache_key(a, START, END) != cache_key(a, START, END + timedelta(days=1))


# ---------------------------------------------------------------- load_or_download


def test_first_call_downloads_validates_and_records_metadata(
    tmp_path: Path, good_prices: pd.DataFrame
) -> None:
    fetcher = CountingFetcher(good_prices)
    prices, meta = load_or_download(
        TICKERS, START, END, tmp_path, GIT, fetcher=fetcher, source_version="test 1.0"
    )
    assert fetcher.calls == 1
    pd.testing.assert_frame_equal(prices, good_prices, check_freq=False)

    parquet, meta_path = cache_paths(tmp_path, cache_key(TICKERS, START, END))
    assert parquet.exists()
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert recorded["tickers"] == list(TICKERS)
    assert recorded["git_commit"] == GIT.commit
    assert recorded["source_version"] == "test 1.0"
    assert recorded["n_rows"] == len(good_prices)
    assert recorded["first_date"] == good_prices.index[0].date().isoformat()
    assert meta.sha256 == recorded["sha256"]


def test_second_call_uses_cache_and_refresh_downloads_again(
    tmp_path: Path, good_prices: pd.DataFrame
) -> None:
    fetcher = CountingFetcher(good_prices)
    first, _ = load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t")
    second, _ = load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t")
    assert fetcher.calls == 1
    pd.testing.assert_frame_equal(first, second)

    load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t", refresh=True)
    assert fetcher.calls == 2


def test_tampered_cache_is_detected(tmp_path: Path, good_prices: pd.DataFrame) -> None:
    fetcher = CountingFetcher(good_prices)
    load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t")
    parquet, _ = cache_paths(tmp_path, cache_key(TICKERS, START, END))
    (good_prices * 1.01).to_parquet(parquet)
    with pytest.raises(CacheIntegrityError, match="checksum"):
        load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t")


def test_invalid_download_is_not_cached(tmp_path: Path, good_prices: pd.DataFrame) -> None:
    fetcher = CountingFetcher(corrupt(good_prices, "nan"))
    with pytest.raises(DataValidationError):
        load_or_download(TICKERS, START, END, tmp_path, GIT, fetcher, source_version="t")
    assert list(tmp_path.iterdir()) == []


def test_default_source_version_records_installed_yfinance(
    tmp_path: Path, good_prices: pd.DataFrame
) -> None:
    import yfinance

    _, meta = load_or_download(TICKERS, START, END, tmp_path, GIT, CountingFetcher(good_prices))
    assert meta.source_version == f"yfinance {yfinance.__version__}"


# ---------------------------------------------------------------- fetch_yfinance (faked)


def fake_yf_frame(index: pd.DatetimeIndex, tickers: Sequence[str]) -> pd.DataFrame:
    """Mimic yfinance's multi-level ('Price', 'Ticker') column layout."""
    columns = pd.MultiIndex.from_product([["Close", "Open"], tickers], names=["Price", "Ticker"])
    values = np.arange(len(index) * len(columns), dtype=float).reshape(len(index), -1) + 1.0
    return pd.DataFrame(values, index=index, columns=columns)


def test_fetch_yfinance_requests_inclusive_end_and_normalizes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    index = pd.date_range("2021-01-04", periods=3, freq="B", tz="America/New_York")

    def fake_download(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return fake_yf_frame(index, ["CCC", "AAA", "BBB"])  # deliberately different order

    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)
    out = fetch_yfinance(TICKERS, START, date(2021, 1, 6))

    assert captured["end"] == "2021-01-07"  # yfinance's end is exclusive
    assert captured["start"] == "2021-01-04"
    assert captured["auto_adjust"] is True
    assert list(out.columns) == list(TICKERS)
    assert out.index.tz is None
    assert out.index.name == "date"
    assert out.dtypes.eq("float64").all()


@pytest.mark.parametrize(
    ("returned_tickers", "message"),
    [(["AAA", "BBB"], r"no prices for: \['CCC'\]"), ([], "returned no data")],
)
def test_fetch_yfinance_reports_missing_data(
    monkeypatch: pytest.MonkeyPatch, returned_tickers: list[str], message: str
) -> None:
    import yfinance

    index = pd.date_range("2021-01-04", periods=3, freq="B")
    frame = fake_yf_frame(index, returned_tickers) if returned_tickers else pd.DataFrame()
    monkeypatch.setattr(yfinance, "download", lambda **_: frame)
    with pytest.raises(DataValidationError, match=message):
        fetch_yfinance(TICKERS, START, END)

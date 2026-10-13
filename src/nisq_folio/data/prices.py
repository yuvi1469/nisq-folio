"""Download, validate, and cache daily adjusted close prices.

Prices come from Yahoo Finance through ``yfinance`` (free; see DECISIONS.md ADR-008/ADR-011).
Each download is validated, then cached as Parquet next to a JSON metadata sidecar holding
the request, download time, library versions, git state, and a SHA-256 of the Parquet file.
The checksum is verified on every load. Raw data is gitignored and never committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nisq_folio.provenance import GitState

PriceFetcher = Callable[[Sequence[str], date, date], pd.DataFrame]
"""Signature of a price source: ``(tickers, start, end) -> prices``, both dates inclusive.

The returned frame has a DatetimeIndex of trading dates and one column per ticker.
"""


class DataValidationError(ValueError):
    """Raised when a price panel fails a data-quality check."""


class CacheIntegrityError(RuntimeError):
    """Raised when a cached file does not match the checksum recorded at download time."""


@dataclass(frozen=True)
class PriceMetadata:
    """Provenance for one cached price download.

    Attributes:
        tickers: Requested tickers, in column order.
        start_date: Requested first date (ISO, inclusive).
        end_date: Requested last date (ISO, inclusive).
        first_date: First trading date actually present (ISO).
        last_date: Last trading date actually present (ISO).
        n_rows: Number of trading dates.
        source: Data source description.
        source_version: Version of the library used to download.
        downloaded_at: UTC timestamp of the download (ISO 8601).
        git_commit: Code commit at download time, or ``None`` if unknown.
        git_dirty: Whether the working tree had uncommitted changes.
        sha256: SHA-256 hex digest of the cached Parquet file.
    """

    tickers: list[str]
    start_date: str
    end_date: str
    first_date: str
    last_date: str
    n_rows: int
    source: str
    source_version: str
    downloaded_at: str
    git_commit: str | None
    git_dirty: bool
    sha256: str


def fetch_yfinance(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance via ``yfinance``.

    Uses ``auto_adjust=True``, so prices are adjusted for splits and dividends. yfinance treats
    ``end`` as exclusive, so one day is added to make ``end`` inclusive here.

    Args:
        tickers: Ticker symbols to download.
        start: First date (inclusive).
        end: Last date (inclusive).

    Returns:
        Adjusted close prices: tz-naive DatetimeIndex named ``"date"``, columns in the order
        of ``tickers``. Not yet validated; pass the result to :func:`validate_prices`.
    """
    import yfinance as yf

    raw = yf.download(
        tickers=list(tickers),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        multi_level_index=True,
    )
    if raw is None or raw.empty:
        raise DataValidationError("yfinance returned no data")
    close = raw["Close"]
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise DataValidationError(f"yfinance returned no prices for: {missing}")
    close = close.loc[:, list(tickers)]
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.rename("date")
    close.columns = pd.Index(list(tickers), name="ticker")
    return close.astype("float64")


def validate_prices(prices: pd.DataFrame, tickers: Sequence[str], start: date, end: date) -> None:
    """Check that a price panel is complete and well-formed.

    Checks: columns equal ``tickers`` (same order); a unique, strictly increasing DatetimeIndex
    inside [start, end]; no missing values; all prices strictly positive and finite.

    A missing value usually means a ticker was not listed for the whole window, which would
    make the covariance estimate use different samples per asset, so it is an error rather
    than something to fill.

    Args:
        prices: Price panel to check.
        tickers: Expected columns, in order.
        start: First allowed date (inclusive).
        end: Last allowed date (inclusive).

    Raises:
        DataValidationError: Describing every failed check for the first failing category.
    """
    if list(prices.columns) != list(tickers):
        raise DataValidationError(
            f"columns {list(prices.columns)} do not match expected tickers {list(tickers)}"
        )
    if prices.empty:
        raise DataValidationError("price panel is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataValidationError("index must be a DatetimeIndex")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise DataValidationError("dates must be unique and strictly increasing")
    first, last = prices.index[0].date(), prices.index[-1].date()
    if first < start or last > end:
        raise DataValidationError(f"dates {first}..{last} fall outside window {start}..{end}")

    nan_counts = prices.isna().sum()
    if nan_counts.any():
        detail = {str(t): int(n) for t, n in nan_counts.items() if n}
        raise DataValidationError(f"missing prices (count per ticker): {detail}")
    values = prices.to_numpy(dtype="float64")
    ok_per_column = (np.isfinite(values) & (values > 0)).all(axis=0)
    if not ok_per_column.all():
        bad = [str(t) for t, ok in zip(prices.columns, ok_per_column, strict=True) if not ok]
        raise DataValidationError(f"non-positive or non-finite prices for: {bad}")


def cache_key(tickers: Sequence[str], start: date, end: date) -> str:
    """Return a short stable identifier for a download request.

    The key is the first 12 hex characters of SHA-256 over the request (tickers in order,
    start, end), so changing the universe or window never reuses a stale cache.

    Args:
        tickers: Tickers in column order.
        start: First date (inclusive).
        end: Last date (inclusive).

    Returns:
        A 12-character hexadecimal string.
    """
    payload = json.dumps(
        {"tickers": list(tickers), "start": start.isoformat(), "end": end.isoformat()}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cache_paths(data_dir: Path, key: str) -> tuple[Path, Path]:
    """Return the ``(parquet, metadata_json)`` paths for a cache key inside ``data_dir``."""
    return data_dir / f"prices_{key}.parquet", data_dir / f"prices_{key}.meta.json"


def load_or_download(
    tickers: Sequence[str],
    start: date,
    end: date,
    data_dir: Path,
    git: GitState,
    fetcher: PriceFetcher = fetch_yfinance,
    source: str = "Yahoo Finance via yfinance (auto_adjust=True, adjusted close)",
    source_version: str | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, PriceMetadata]:
    """Load validated prices from the cache, downloading and caching them first if needed.

    Args:
        tickers: Tickers in column order.
        start: First date (inclusive).
        end: Last date (inclusive).
        data_dir: Cache directory (created if missing).
        git: Git state to record if a download happens.
        fetcher: Price source; defaults to :func:`fetch_yfinance`. Tests inject a fake.
        source: Human-readable source description stored in the metadata.
        source_version: Library version stored in the metadata; defaults to the installed
            yfinance version.
        refresh: If True, ignore any existing cache and download again.

    Returns:
        ``(prices, metadata)``.

    Raises:
        DataValidationError: If downloaded data fails :func:`validate_prices`.
        CacheIntegrityError: If a cached file's checksum does not match its metadata.
    """
    parquet_path, meta_path = cache_paths(data_dir, cache_key(tickers, start, end))
    if not refresh and parquet_path.exists() and meta_path.exists():
        return _load_cached(parquet_path, meta_path, tickers, start, end)

    prices = fetcher(tickers, start, end)
    validate_prices(prices, tickers, start, end)

    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(parquet_path)
    if source_version is None:
        import yfinance

        source_version = f"yfinance {yfinance.__version__}"
    metadata = PriceMetadata(
        tickers=list(tickers),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        first_date=prices.index[0].date().isoformat(),
        last_date=prices.index[-1].date().isoformat(),
        n_rows=len(prices),
        source=source,
        source_version=source_version,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git.commit,
        git_dirty=git.dirty,
        sha256=_sha256(parquet_path),
    )
    meta_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return _load_cached(parquet_path, meta_path, tickers, start, end)


def _load_cached(
    parquet_path: Path, meta_path: Path, tickers: Sequence[str], start: date, end: date
) -> tuple[pd.DataFrame, PriceMetadata]:
    metadata = PriceMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
    actual = _sha256(parquet_path)
    if actual != metadata.sha256:
        raise CacheIntegrityError(
            f"{parquet_path.name}: checksum {actual} does not match recorded {metadata.sha256}; "
            "delete the cache files or rerun with refresh"
        )
    prices = pd.read_parquet(parquet_path)
    validate_prices(prices, tickers, start, end)
    return prices, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

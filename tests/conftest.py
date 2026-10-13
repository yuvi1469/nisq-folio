"""Shared pytest configuration and fixtures."""

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

# Deterministic property-based testing: the same examples on every run and machine,
# so a failure seen in CI reproduces locally (CLAUDE.md rule 1, DECISIONS.md ADR-012).
settings.register_profile("deterministic", derandomize=True, deadline=None)
settings.load_profile("deterministic")

PriceFactory = Callable[..., pd.DataFrame]
"""factory(tickers, start, n_days, seed=0) -> prices; see the make_prices fixture."""


@pytest.fixture
def make_prices() -> PriceFactory:
    """Return a factory for synthetic, strictly positive business-day price panels.

    Prices follow a seeded geometric random walk, P_t = 100 * prod_{s<=t} (1 + r_s) with
    r_s ~ N(0.0005, 0.01^2), so they are realistic in scale but involve no network access.
    """

    def factory(tickers: Sequence[str], start: date, n_days: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0005, 0.01, size=(n_days, len(tickers)))
        values = 100.0 * np.cumprod(1.0 + steps, axis=0)
        index = pd.bdate_range(start=start, periods=n_days, name="date")
        return pd.DataFrame(values, index=index, columns=pd.Index(list(tickers), name="ticker"))

    return factory

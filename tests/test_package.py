"""Placeholder tests for the top-level nisq_folio package."""

import nisq_folio


def test_version_is_defined() -> None:
    assert isinstance(nisq_folio.__version__, str)
    assert nisq_folio.__version__

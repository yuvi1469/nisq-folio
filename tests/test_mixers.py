"""Placeholder tests for nisq_folio.mixers; replace with real tests as the module grows."""

import importlib


def test_mixers_package_imports_and_is_documented() -> None:
    module = importlib.import_module("nisq_folio.mixers")
    assert module.__doc__ is not None
    assert module.__doc__.strip()

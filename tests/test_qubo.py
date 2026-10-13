"""Placeholder tests for nisq_folio.qubo; replace with real tests as the module grows."""

import importlib


def test_qubo_package_imports_and_is_documented() -> None:
    module = importlib.import_module("nisq_folio.qubo")
    assert module.__doc__ is not None
    assert module.__doc__.strip()

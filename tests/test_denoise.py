"""Placeholder tests for nisq_folio.denoise; replace with real tests as the module grows."""

import importlib


def test_denoise_package_imports_and_is_documented() -> None:
    module = importlib.import_module("nisq_folio.denoise")
    assert module.__doc__ is not None
    assert module.__doc__.strip()

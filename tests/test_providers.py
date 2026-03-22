"""
Tests for simulator/providers.py — factory and error handling.
"""

import pytest
from simulator.providers import get_provider, ProviderError, AnthropicProvider, HuggingFaceProvider, Provider


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError) as exc_info:
        get_provider("unknown_provider")
    assert "unknown_provider" in str(exc_info.value)
    assert "anthropic" in str(exc_info.value)


def test_get_provider_anthropic_no_key_raises():
    import os
    original = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            get_provider("anthropic", api_key=None)
    finally:
        if original is not None:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_get_provider_huggingface_no_key_raises():
    import os
    original = os.environ.pop("HF_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            get_provider("huggingface", api_key=None)
    finally:
        if original is not None:
            os.environ["HF_API_KEY"] = original


def test_anthropic_provider_satisfies_protocol():
    """AnthropicProvider must have a callable 'call' method (Provider protocol)."""
    assert hasattr(AnthropicProvider, "__init__")
    # Check the class has a 'call' method
    assert callable(getattr(AnthropicProvider, "call", None))


def test_huggingface_provider_satisfies_protocol():
    assert callable(getattr(HuggingFaceProvider, "call", None))

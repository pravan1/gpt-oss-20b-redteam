"""Tests for model backends."""

import pytest
from gpt_oss_redteam.model.backends import MockBackend, get_backend


def test_mock_backend_initialization():
    """Test MockBackend initialization."""
    backend = MockBackend(deterministic=True, latency_ms=10)
    assert backend is not None
    assert backend.deterministic == True
    assert backend.latency_ms == 10


def test_mock_backend_generate():
    """Test MockBackend generation."""
    backend = MockBackend(deterministic=True, latency_ms=0)
    
    # Test basic generation
    response = backend.generate("What is 2+2?", seed=42)
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test deterministic behavior
    response1 = backend.generate("test prompt", seed=42)
    backend2 = MockBackend(deterministic=True, latency_ms=0)
    response2 = backend2.generate("test prompt", seed=42)
    assert response1 == response2


def test_mock_backend_categories():
    """Test MockBackend responds differently to categories."""
    backend = MockBackend(deterministic=True, latency_ms=0)
    
    # Test different categories
    reward_response = backend.generate("reward_hacking test", seed=0)
    deception_response = backend.generate("deception test", seed=0)
    
    # Should get different responses for different categories
    assert reward_response != deception_response


def test_get_backend():
    """Test backend factory function."""
    # Test mock backend
    backend = get_backend(
        "mock",
        {"deterministic": True, "latency_ms": 0},
        None
    )
    assert isinstance(backend, MockBackend)
    
    # Test invalid backend
    with pytest.raises(ValueError):
        get_backend("invalid_backend", {}, None)
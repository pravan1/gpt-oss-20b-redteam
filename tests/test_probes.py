"""Tests for probe framework."""

import pytest
from gpt_oss_redteam.model.backends import MockBackend
from gpt_oss_redteam.probes.base import ProbeResult, SimpleProbe
from gpt_oss_redteam.probes.registry import register_probe, get_probe, list_probes


def test_simple_probe():
    """Test SimpleProbe execution."""
    backend = MockBackend(deterministic=True, latency_ms=0)
    
    probe = SimpleProbe(
        name="test_probe",
        category="test",
        description="Test probe",
        prompt="Test prompt",
    )
    
    result = probe.run(backend, seed=42)
    
    assert isinstance(result, ProbeResult)
    assert result.probe_name == "test_probe"
    assert result.seed == 42
    assert len(result.repro_prompts) == 1
    assert result.repro_prompts[0] == "Test prompt"


def test_probe_result():
    """Test ProbeResult dataclass."""
    result = ProbeResult(
        probe_name="test",
        passed=True,
        evidence=["evidence1", "evidence2"],
        metrics={"metric1": 1.0},
        seed=42,
        tags=["tag1", "tag2"],
    )
    
    # Test to_dict
    result_dict = result.to_dict()
    assert result_dict["probe_name"] == "test"
    assert result_dict["passed"] == True
    assert len(result_dict["evidence"]) == 2
    
    # Test summary
    summary = result.summary()
    assert "PASSED" in summary
    assert "test" in summary


def test_probe_registry():
    """Test probe registry functions."""
    # Create a test probe class
    class TestRegistryProbe(SimpleProbe):
        def __init__(self):
            super().__init__(
                name="registry_test",
                category="test",
                description="Registry test probe",
                prompt="Test",
            )
    
    # Register probe
    register_probe(TestRegistryProbe)
    
    # Get probe
    probe_class = get_probe("test.registry_test")
    assert probe_class is not None
    
    # List probes
    probes = list_probes(category="test")
    assert any("registry_test" in p for p in probes)
"""Tests for format converters."""

import json
import tempfile
from pathlib import Path

import pytest
from gpt_oss_redteam.eval.converters import harmony_to_kaggle, kaggle_to_harmony


def test_harmony_to_kaggle_conversion():
    """Test converting Harmony format to Kaggle format."""
    # Create test Harmony data
    harmony_data = {
        "findings": [{
            "issue_title": "Test Issue",
            "issue_summary": {
                "overview": "Observed behavior",
                "desired_behavior": "Expected behavior"
            },
            "harmony_response_walkthroughs": [
                {"description": "Evidence 1"},
                {"description": "Evidence 2"}
            ],
            "steps_to_reproduce": ["Step 1", "Step 2"],
            "severity": "high",
            "breadth": "medium",
            "model": {
                "name": "test-model",
                "version": "1.0",
                "provider": "test"
            }
        }]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save Harmony data
        harmony_path = Path(tmpdir) / "harmony.json"
        with open(harmony_path, "w") as f:
            json.dump(harmony_data, f)
        
        # Convert to Kaggle
        kaggle_path = Path(tmpdir) / "kaggle.json"
        result = harmony_to_kaggle(str(harmony_path), str(kaggle_path))
        
        # Check result
        assert "findings" in result
        assert len(result["findings"]) == 1
        
        finding = result["findings"][0]
        assert finding["title"] == "Test Issue"
        assert finding["observed_behavior"] == "Observed behavior"
        assert finding["expected_behavior"] == "Expected behavior"
        assert len(finding["evidence"]) == 2
        assert finding["severity_estimate"] == 3  # high -> 3
        assert finding["breadth_estimate"] == 2  # medium -> 2


def test_kaggle_to_harmony_conversion():
    """Test converting Kaggle format to Harmony format."""
    # Create test Kaggle data
    kaggle_data = {
        "version": "0.1.0",
        "findings": [{
            "title": "Test Finding",
            "observed_behavior": "What happened",
            "expected_behavior": "What should happen",
            "evidence": ["Evidence item 1", "Evidence item 2"],
            "repro_instructions": "1. Step one\n2. Step two",
            "severity_estimate": 3,
            "breadth_estimate": 2,
            "metadata": {
                "model": "test-model",
                "version": "1.0",
                "provider": "test"
            }
        }]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save Kaggle data
        kaggle_path = Path(tmpdir) / "kaggle.json"
        with open(kaggle_path, "w") as f:
            json.dump(kaggle_data, f)
        
        # Convert to Harmony
        harmony_path = Path(tmpdir) / "harmony.json"
        result = kaggle_to_harmony(str(kaggle_path), str(harmony_path))
        
        # Check result
        assert "findings" in result
        assert len(result["findings"]) == 1
        
        finding = result["findings"][0]
        assert finding["issue_title"] == "Test Finding"
        assert finding["issue_summary"]["overview"] == "What happened"
        assert finding["issue_summary"]["desired_behavior"] == "What should happen"
        assert len(finding["harmony_response_walkthroughs"]) == 2
        assert finding["severity"] == "high"  # 3 -> high
        assert finding["breadth"] == "medium"  # 2 -> medium


def test_round_trip_conversion():
    """Test round-trip conversion preserves data."""
    original_kaggle = {
        "version": "0.1.0",
        "findings": [{
            "title": "Round Trip Test",
            "observed_behavior": "Observed",
            "expected_behavior": "Expected",
            "evidence": ["E1", "E2"],
            "severity_estimate": 2,
            "breadth_estimate": 3
        }]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save original
        kaggle1_path = Path(tmpdir) / "kaggle1.json"
        with open(kaggle1_path, "w") as f:
            json.dump(original_kaggle, f)
        
        # Convert to Harmony
        harmony_path = Path(tmpdir) / "harmony.json"
        kaggle_to_harmony(str(kaggle1_path), str(harmony_path))
        
        # Convert back to Kaggle
        kaggle2_path = Path(tmpdir) / "kaggle2.json"
        harmony_to_kaggle(str(harmony_path), str(kaggle2_path))
        
        # Load final result
        with open(kaggle2_path) as f:
            final_kaggle = json.load(f)
        
        # Check key fields preserved
        orig_finding = original_kaggle["findings"][0]
        final_finding = final_kaggle["findings"][0]
        
        assert final_finding["title"] == orig_finding["title"]
        assert final_finding["observed_behavior"] == orig_finding["observed_behavior"]
        assert final_finding["expected_behavior"] == orig_finding["expected_behavior"]
        assert final_finding["severity_estimate"] == orig_finding["severity_estimate"]
        assert final_finding["breadth_estimate"] == orig_finding["breadth_estimate"]
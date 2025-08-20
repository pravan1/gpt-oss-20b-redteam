"""Tests for reporting functions."""

import json
import tempfile
from pathlib import Path

import pytest
from gpt_oss_redteam.eval.reporting import (
    make_findings_json,
    generate_report,
    summarize_results
)
from gpt_oss_redteam.probes.base import ProbeResult


def test_make_findings_json():
    """Test Kaggle findings JSON generation."""
    results = [
        ProbeResult(
            probe_name="reward_hacking.test",
            passed=False,
            evidence=["High confidence on wrong answer"],
            repro_prompts=["What is 2+2?"],
            seed=42,
            tags=["medium"],
        ),
        ProbeResult(
            probe_name="deception.test",
            passed=False,
            evidence=["Claimed tool usage"],
            repro_prompts=["Check the weather"],
            seed=42,
            tags=["high"],
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "findings.json"
        
        findings = make_findings_json(
            results,
            issue_id="TEST-001",
            meta={"test": True},
            out_path=str(output_path)
        )
        
        assert findings["issue_id"] == "TEST-001"
        assert len(findings["findings"]) > 0
        assert output_path.exists()
        
        # Load and validate
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["issue_id"] == "TEST-001"


def test_generate_report():
    """Test report generation."""
    results = [
        ProbeResult(
            probe_name="test.probe1",
            passed=True,
            duration_ms=100,
        ),
        ProbeResult(
            probe_name="test.probe2",
            passed=False,
            evidence=["Failed check"],
            duration_ms=200,
        ),
    ]
    
    # Test markdown format
    report = generate_report(results, format="markdown")
    assert "# Red Team Evaluation Report" in report
    assert "Summary" in report
    assert "50.0%" in report  # Pass rate
    
    # Test text format
    report = generate_report(results, format="text")
    assert "RED TEAM EVALUATION REPORT" in report
    
    # Test HTML format
    report = generate_report(results, format="html")
    assert "<html>" in report
    assert "<h1>Red Team Evaluation Report</h1>" in report


def test_summarize_results():
    """Test results summarization."""
    results = [
        ProbeResult(
            probe_name="critical.probe",
            passed=False,
            evidence=["Critical issue found"],
            tags=["critical"],
        ),
        ProbeResult(
            probe_name="test.probe",
            passed=True,
        ),
    ]
    
    summary = summarize_results(results)
    
    assert summary["total_probes"] == 2
    assert summary["pass_rate"] == 0.5
    assert len(summary["critical_issues"]) > 0
    assert summary["critical_issues"][0]["severity"] == "critical"
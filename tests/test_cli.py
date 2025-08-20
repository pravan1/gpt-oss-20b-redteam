"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpt_oss_redteam.cli import app

runner = CliRunner()


def test_version_command():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "GPT-OSS Red Team Framework" in result.stdout
    assert "Version:" in result.stdout


def test_init_command():
    """Test init command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory
        import os
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert "Initialization complete" in result.stdout
            
            # Check directories were created
            assert Path("runs").exists()
            assert Path("findings_templates").exists()
            assert Path("examples").exists()
            assert Path("config").exists()
        finally:
            os.chdir(original_dir)


def test_list_command():
    """Test list command."""
    result = runner.invoke(app, ["list"])
    # May have no probes initially, but command should run
    assert result.exit_code == 0


def test_import_harmony_command():
    """Test import-harmony command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test Harmony file
        harmony_file = Path(tmpdir) / "harmony.json"
        harmony_data = {
            "findings": [{
                "issue_title": "Test",
                "issue_summary": {
                    "overview": "Test overview",
                    "desired_behavior": "Test behavior"
                }
            }]
        }
        with open(harmony_file, "w") as f:
            json.dump(harmony_data, f)
        
        # Run command
        output_file = Path(tmpdir) / "kaggle.json"
        result = runner.invoke(app, [
            "import-harmony",
            "--in", str(harmony_file),
            "--out", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "Converted" in result.stdout
        assert output_file.exists()


def test_export_harmony_command():
    """Test export-harmony command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test Kaggle file
        kaggle_file = Path(tmpdir) / "kaggle.json"
        kaggle_data = {
            "version": "0.1.0",
            "findings": [{
                "title": "Test",
                "observed_behavior": "Observed",
                "expected_behavior": "Expected"
            }]
        }
        with open(kaggle_file, "w") as f:
            json.dump(kaggle_data, f)
        
        # Run command
        output_file = Path(tmpdir) / "harmony.json"
        result = runner.invoke(app, [
            "export-harmony",
            "--in", str(kaggle_file),
            "--out", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "Converted" in result.stdout
        assert output_file.exists()
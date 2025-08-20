"""Converters between Harmony and Kaggle finding formats."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

logger = logging.getLogger(__name__)


def harmony_to_kaggle(
    harmony_path: str,
    out_path: str,
    map_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert Harmony findings format to Kaggle format.
    
    Args:
        harmony_path: Path to Harmony findings JSON
        out_path: Path to output Kaggle findings JSON
        map_overrides: Optional mapping overrides
        
    Returns:
        Converted Kaggle format dictionary
    """
    # Load Harmony findings
    with open(harmony_path, "r") as f:
        harmony_data = json.load(f)
    
    # Initialize Kaggle format
    kaggle_data = {
        "version": "0.1.0",  # Package version
        "timestamp": datetime.utcnow().isoformat(),
        "findings": [],
    }
    
    # Convert each finding
    findings = harmony_data.get("findings", [harmony_data])
    
    for harmony_finding in findings:
        kaggle_finding = {}
        
        # Map title
        kaggle_finding["title"] = harmony_finding.get("issue_title", "Untitled Finding")
        
        # Map behavior descriptions
        summary = harmony_finding.get("issue_summary", {})
        kaggle_finding["observed_behavior"] = summary.get("overview", "")[:500]  # Trim safely
        kaggle_finding["expected_behavior"] = summary.get("desired_behavior", "")
        
        # Map evidence from walkthroughs
        evidence_items = []
        walkthroughs = harmony_finding.get("harmony_response_walkthroughs", [])
        for walkthrough in walkthroughs:
            if isinstance(walkthrough, dict):
                evidence_items.append(walkthrough.get("description", ""))
            else:
                evidence_items.append(str(walkthrough))
        kaggle_finding["evidence"] = evidence_items
        
        # Map reproduction steps
        steps = harmony_finding.get("steps_to_reproduce", [])
        if isinstance(steps, list):
            kaggle_finding["repro_instructions"] = "\n".join(
                f"{i+1}. {step}" for i, step in enumerate(steps)
            )
        else:
            kaggle_finding["repro_instructions"] = str(steps)
        
        # Map severity and breadth
        severity_map = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }
        
        severity = harmony_finding.get("severity", "medium").lower()
        kaggle_finding["severity_estimate"] = severity_map.get(severity, 2)
        
        breadth = harmony_finding.get("breadth", "medium").lower()
        kaggle_finding["breadth_estimate"] = severity_map.get(breadth, 2)
        
        # Map metadata
        kaggle_finding["metadata"] = {
            "model": harmony_finding.get("model", {}).get("name", "gpt-oss-20b"),
            "version": harmony_finding.get("model", {}).get("version", "1.0"),
            "provider": harmony_finding.get("model", {}).get("provider", "openai"),
            "timestamp": harmony_finding.get("timestamp", datetime.utcnow().isoformat()),
        }
        
        # Apply overrides if provided
        if map_overrides:
            for key, value in map_overrides.items():
                if key in kaggle_finding:
                    kaggle_finding[key] = value
        
        kaggle_data["findings"].append(kaggle_finding)
    
    # Validate against schema
    schema_path = Path(__file__).parent.parent.parent / "findings_templates" / "findings.schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(kaggle_data, schema)
            logger.info("Kaggle findings validated against schema")
        except jsonschema.ValidationError as e:
            logger.warning(f"Schema validation failed: {e}")
    
    # Save to file
    with open(out_path, "w") as f:
        json.dump(kaggle_data, f, indent=2)
    
    logger.info(f"Converted {len(kaggle_data['findings'])} findings from Harmony to Kaggle format")
    return kaggle_data


def kaggle_to_harmony(
    kaggle_path: str,
    out_path: str,
) -> Dict[str, Any]:
    """
    Convert Kaggle findings format to Harmony format.
    
    Args:
        kaggle_path: Path to Kaggle findings JSON
        out_path: Path to output Harmony findings JSON
        
    Returns:
        Converted Harmony format dictionary
    """
    # Load Kaggle findings
    with open(kaggle_path, "r") as f:
        kaggle_data = json.load(f)
    
    # Initialize Harmony format
    harmony_data = {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "findings": [],
    }
    
    # Convert each finding
    for kaggle_finding in kaggle_data.get("findings", []):
        harmony_finding = {}
        
        # Map title
        harmony_finding["issue_title"] = kaggle_finding.get("title", "Untitled")
        
        # Map summary
        harmony_finding["issue_summary"] = {
            "overview": kaggle_finding.get("observed_behavior", ""),
            "desired_behavior": kaggle_finding.get("expected_behavior", ""),
        }
        
        # Map evidence to walkthroughs
        evidence = kaggle_finding.get("evidence", [])
        harmony_finding["harmony_response_walkthroughs"] = [
            {"description": item} if isinstance(item, str) else item
            for item in evidence
        ]
        
        # Map reproduction steps
        repro = kaggle_finding.get("repro_instructions", "")
        if repro:
            # Try to parse numbered steps
            steps = []
            for line in repro.split("\n"):
                line = line.strip()
                if line:
                    # Remove numbering if present
                    line = re.sub(r"^\d+\.\s*", "", line)
                    steps.append(line)
            harmony_finding["steps_to_reproduce"] = steps
        
        # Map severity and breadth
        severity_map = {
            4: "critical",
            3: "high",
            2: "medium",
            1: "low",
        }
        
        severity_num = kaggle_finding.get("severity_estimate", 2)
        harmony_finding["severity"] = severity_map.get(severity_num, "medium")
        
        breadth_num = kaggle_finding.get("breadth_estimate", 2)
        harmony_finding["breadth"] = severity_map.get(breadth_num, "medium")
        
        # Map metadata
        metadata = kaggle_finding.get("metadata", {})
        harmony_finding["model"] = {
            "name": metadata.get("model", "gpt-oss-20b"),
            "version": metadata.get("version", "1.0"),
            "provider": metadata.get("provider", "openai"),
        }
        harmony_finding["timestamp"] = metadata.get("timestamp", datetime.utcnow().isoformat())
        
        harmony_data["findings"].append(harmony_finding)
    
    # Save to file
    with open(out_path, "w") as f:
        json.dump(harmony_data, f, indent=2)
    
    logger.info(f"Converted {len(harmony_data['findings'])} findings from Kaggle to Harmony format")
    return harmony_data


def validate_kaggle_findings(findings_path: str, schema_path: Optional[str] = None) -> bool:
    """
    Validate Kaggle findings against JSON schema.
    
    Args:
        findings_path: Path to Kaggle findings JSON
        schema_path: Optional path to schema (uses default if not provided)
        
    Returns:
        True if valid, False otherwise
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent.parent / "findings_templates" / "findings.schema.json"
    
    try:
        with open(findings_path) as f:
            findings = json.load(f)
        
        with open(schema_path) as f:
            schema = json.load(f)
        
        jsonschema.validate(findings, schema)
        logger.info(f"Findings at {findings_path} are valid")
        return True
        
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        logger.error(f"Validation failed: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False


import re  # Add this import for the regex in kaggle_to_harmony
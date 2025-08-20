"""Summary generation utilities."""

import logging
from typing import Any, Dict, List

from ..probes.base import ProbeResult

logger = logging.getLogger(__name__)


def create_summary(results: List[ProbeResult], verbose: bool = False) -> str:
    """
    Create a text summary of probe results.
    
    Args:
        results: List of ProbeResult objects
        verbose: Whether to include detailed information
        
    Returns:
        Summary string
    """
    if not results:
        return "No results to summarize."
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    
    summary_lines = [
        f"Evaluation Summary:",
        f"  Total probes: {total}",
        f"  Passed: {passed} ({passed/total:.1%})",
        f"  Failed: {failed} ({failed/total:.1%})",
    ]
    
    if verbose:
        # Add category breakdown
        categories = {}
        for result in results:
            cat = result.probe_name.split(".")[0] if "." in result.probe_name else "unknown"
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if result.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        summary_lines.append("\n  Category Breakdown:")
        for cat, counts in sorted(categories.items()):
            total_cat = counts["passed"] + counts["failed"]
            pass_rate = counts["passed"] / total_cat if total_cat > 0 else 0
            summary_lines.append(f"    {cat}: {pass_rate:.1%} pass rate ({counts['passed']}/{total_cat})")
        
        # Add top failures
        failures = [r for r in results if not r.passed]
        if failures:
            summary_lines.append("\n  Top Failures:")
            for failure in failures[:5]:
                evidence = failure.evidence[0] if failure.evidence else "No evidence"
                summary_lines.append(f"    - {failure.probe_name}: {evidence[:50]}...")
    
    return "\n".join(summary_lines)


def format_summary(summary_data: Dict[str, Any], format: str = "text") -> str:
    """
    Format summary data for display.
    
    Args:
        summary_data: Summary dictionary
        format: Output format (text, json, markdown)
        
    Returns:
        Formatted summary string
    """
    if format == "json":
        import json
        return json.dumps(summary_data, indent=2)
    
    elif format == "markdown":
        lines = [
            "## Evaluation Summary\n",
            f"- **Total Probes**: {summary_data.get('total_probes', 0)}",
            f"- **Pass Rate**: {summary_data.get('pass_rate', 0):.1%}",
            f"- **Weighted Pass Rate**: {summary_data.get('weighted_pass_rate', 0):.1%}",
        ]
        
        if "categories_tested" in summary_data:
            lines.append(f"- **Categories**: {', '.join(summary_data['categories_tested'])}")
        
        if "critical_issues" in summary_data and summary_data["critical_issues"]:
            lines.append("\n### Critical Issues\n")
            for issue in summary_data["critical_issues"]:
                lines.append(f"- **{issue['probe']}** ({issue['severity']})")
                for evidence in issue.get("evidence", []):
                    lines.append(f"  - {evidence}")
        
        if "recommendations" in summary_data and summary_data["recommendations"]:
            lines.append("\n### Recommendations\n")
            for rec in summary_data["recommendations"]:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    else:  # text format
        lines = [
            "EVALUATION SUMMARY",
            "=" * 40,
            f"Total Probes: {summary_data.get('total_probes', 0)}",
            f"Pass Rate: {summary_data.get('pass_rate', 0):.1%}",
            f"Weighted Pass Rate: {summary_data.get('weighted_pass_rate', 0):.1%}",
        ]
        
        if "categories_tested" in summary_data:
            lines.append(f"Categories: {', '.join(summary_data['categories_tested'])}")
        
        if "critical_issues" in summary_data and summary_data["critical_issues"]:
            lines.append("\nCRITICAL ISSUES:")
            for issue in summary_data["critical_issues"]:
                lines.append(f"  {issue['probe']} ({issue['severity']})")
        
        if "recommendations" in summary_data and summary_data["recommendations"]:
            lines.append("\nRECOMMENDATIONS:")
            for rec in summary_data["recommendations"]:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)
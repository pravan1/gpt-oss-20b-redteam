"""Base classes for probe implementation."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..model.sampling import Transcript

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result from running a probe."""
    
    probe_name: str
    passed: bool
    evidence: List[str] = field(default_factory=list)
    transcript: Optional[Transcript] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    repro_prompts: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "probe_name": self.probe_name,
            "passed": self.passed,
            "evidence": self.evidence,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "metrics": self.metrics,
            "repro_prompts": self.repro_prompts,
            "seed": self.seed,
            "tags": self.tags,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Generate a summary of the probe result."""
        status = "PASSED" if self.passed else "FAILED"
        summary_parts = [
            f"Probe: {self.probe_name}",
            f"Status: {status}",
        ]
        
        if self.error:
            summary_parts.append(f"Error: {self.error}")
        
        if self.evidence:
            summary_parts.append(f"Evidence: {len(self.evidence)} items")
        
        if self.metrics:
            summary_parts.append(f"Metrics: {', '.join(f'{k}={v}' for k, v in self.metrics.items())}")
        
        if self.tags:
            summary_parts.append(f"Tags: {', '.join(self.tags)}")
        
        return " | ".join(summary_parts)


class Probe(ABC):
    """Abstract base class for all probes."""
    
    def __init__(self, name: str, category: str, description: str):
        """
        Initialize probe.
        
        Args:
            name: Probe name
            category: Probe category
            description: Probe description
        """
        self.name = name
        self.category = category
        self.description = description
        self.tags: List[str] = []
        self.severity: str = "low"  # low, medium, high, critical
    
    @abstractmethod
    def run(self, model: Any, seed: Optional[int] = None, **kwargs) -> ProbeResult:
        """
        Run the probe against a model.
        
        Args:
            model: Model backend instance
            seed: Random seed for reproducibility
            **kwargs: Additional probe-specific parameters
            
        Returns:
            ProbeResult instance
        """
        pass
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate model response (can be overridden by subclasses).
        
        Args:
            response: Model response to validate
            
        Returns:
            Validation results dictionary
        """
        return {"valid": True}
    
    def extract_metrics(self, response: str) -> Dict[str, float]:
        """
        Extract metrics from response (can be overridden by subclasses).
        
        Args:
            response: Model response
            
        Returns:
            Dictionary of metrics
        """
        return {}
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.category}.{self.name}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Probe(name='{self.name}', category='{self.category}')"


class SimpleProbe(Probe):
    """Simple probe implementation for basic testing."""
    
    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        prompt: str,
        expected_behavior: Optional[str] = None,
        check_function: Optional[Any] = None,
    ):
        """
        Initialize simple probe.
        
        Args:
            name: Probe name
            category: Probe category
            description: Probe description
            prompt: Prompt to send to model
            expected_behavior: Expected behavior description
            check_function: Optional function to check response
        """
        super().__init__(name, category, description)
        self.prompt = prompt
        self.expected_behavior = expected_behavior
        self.check_function = check_function
    
    def run(self, model: Any, seed: Optional[int] = None, **kwargs) -> ProbeResult:
        """
        Run the simple probe.
        
        Args:
            model: Model backend instance
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            ProbeResult instance
        """
        start_time = time.time()
        
        try:
            # Generate response
            response = model.generate(
                prompt=self.prompt,
                seed=seed,
                **kwargs
            )
            
            # Create transcript
            from ..model.sampling import SamplingParams, Transcript
            transcript = Transcript(
                prompt=self.prompt,
                response=response,
                params=SamplingParams(seed=seed, **kwargs),
                backend_id=model.__class__.__name__,
            )
            
            # Check response
            if self.check_function:
                passed = self.check_function(response)
                evidence = [f"Check function returned: {passed}"]
            else:
                # Default: pass if response is non-empty
                passed = bool(response and response.strip())
                evidence = [f"Response length: {len(response)}"]
            
            # Extract metrics
            metrics = self.extract_metrics(response)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name=self.name,
                passed=passed,
                evidence=evidence,
                transcript=transcript,
                metrics=metrics,
                repro_prompts=[self.prompt],
                seed=seed,
                tags=self.tags,
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            logger.error(f"Probe {self.name} failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name=self.name,
                passed=False,
                evidence=[f"Exception: {str(e)}"],
                error=str(e),
                duration_ms=duration_ms,
                seed=seed,
            )


class ComparativeProbe(Probe):
    """Probe that compares responses under different conditions."""
    
    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        prompts: Dict[str, str],
        comparison_metric: Optional[str] = None,
    ):
        """
        Initialize comparative probe.
        
        Args:
            name: Probe name
            category: Probe category
            description: Probe description
            prompts: Dictionary of condition -> prompt
            comparison_metric: Metric to use for comparison
        """
        super().__init__(name, category, description)
        self.prompts = prompts
        self.comparison_metric = comparison_metric or "similarity"
    
    def run(self, model: Any, seed: Optional[int] = None, **kwargs) -> ProbeResult:
        """
        Run comparative probe.
        
        Args:
            model: Model backend instance
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            ProbeResult instance
        """
        start_time = time.time()
        responses = {}
        transcripts = []
        
        try:
            # Generate responses for each condition
            for condition, prompt in self.prompts.items():
                response = model.generate(
                    prompt=prompt,
                    seed=seed,
                    **kwargs
                )
                responses[condition] = response
                
                # Create transcript
                from ..model.sampling import SamplingParams, Transcript
                transcript = Transcript(
                    prompt=prompt,
                    response=response,
                    params=SamplingParams(seed=seed, **kwargs),
                    backend_id=model.__class__.__name__,
                    metadata={"condition": condition},
                )
                transcripts.append(transcript)
            
            # Compare responses
            comparison_result = self.compare_responses(responses)
            passed = comparison_result.get("passed", False)
            evidence = comparison_result.get("evidence", [])
            metrics = comparison_result.get("metrics", {})
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name=self.name,
                passed=passed,
                evidence=evidence,
                transcript=transcripts[0] if transcripts else None,  # Primary transcript
                metrics=metrics,
                repro_prompts=list(self.prompts.values()),
                seed=seed,
                tags=self.tags,
                duration_ms=duration_ms,
                metadata={"all_transcripts": [t.to_dict() for t in transcripts]},
            )
            
        except Exception as e:
            logger.error(f"Comparative probe {self.name} failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name=self.name,
                passed=False,
                evidence=[f"Exception: {str(e)}"],
                error=str(e),
                duration_ms=duration_ms,
                seed=seed,
            )
    
    def compare_responses(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare responses from different conditions.
        
        Args:
            responses: Dictionary of condition -> response
            
        Returns:
            Comparison results
        """
        # Default implementation: check if responses differ
        unique_responses = set(responses.values())
        passed = len(unique_responses) > 1  # Fail if responses are too similar
        
        evidence = [
            f"Unique responses: {len(unique_responses)}/{len(responses)}",
            f"Conditions tested: {', '.join(responses.keys())}",
        ]
        
        metrics = {
            "num_conditions": len(responses),
            "num_unique": len(unique_responses),
        }
        
        return {
            "passed": passed,
            "evidence": evidence,
            "metrics": metrics,
        }
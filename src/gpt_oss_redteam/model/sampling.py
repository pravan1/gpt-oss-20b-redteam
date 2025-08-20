"""Sampling utilities for model generation."""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    """Parameters for text generation sampling."""
    
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    num_return_sequences: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences,
            "seed": self.seed,
            "n": self.num_return_sequences,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
        }


@dataclass
class Transcript:
    """Record of a generation interaction."""
    
    prompt: str
    response: str
    params: SamplingParams
    backend_id: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "params": self.params.to_dict(),
            "backend_id": self.backend_id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.debug(f"Set random seed to {seed}")


def batch_generate(
    backend: Any,
    prompts: List[str],
    params: Optional[SamplingParams] = None,
    batch_size: int = 8,
    show_progress: bool = True,
) -> List[Transcript]:
    """
    Generate responses for multiple prompts in batches.
    
    Args:
        backend: Model backend instance
        prompts: List of prompts to process
        params: Sampling parameters
        batch_size: Number of prompts to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        List of Transcript objects
    """
    if params is None:
        params = SamplingParams()
    
    transcripts = []
    
    # Import tqdm conditionally
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(prompts), batch_size), desc="Generating")
    else:
        iterator = range(0, len(prompts), batch_size)
    
    for i in iterator:
        batch = prompts[i:i + batch_size]
        
        for prompt in batch:
            start_time = time.time()
            
            # Generate response
            response = backend.generate(
                prompt=prompt,
                **params.to_dict(),
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Create transcript
            transcript = Transcript(
                prompt=prompt,
                response=response,
                params=params,
                backend_id=backend.__class__.__name__,
                duration_ms=duration_ms,
            )
            
            transcripts.append(transcript)
    
    return transcripts


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def generate_with_retry(
    backend: Any,
    prompt: str,
    params: Optional[SamplingParams] = None,
    max_attempts: int = 3,
) -> Transcript:
    """
    Generate response with retry logic.
    
    Args:
        backend: Model backend instance
        prompt: Input prompt
        params: Sampling parameters
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Transcript object
        
    Raises:
        Exception: If generation fails after all retries
    """
    if params is None:
        params = SamplingParams()
    
    start_time = time.time()
    
    try:
        response = backend.generate(
            prompt=prompt,
            **params.to_dict(),
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
    
    duration_ms = (time.time() - start_time) * 1000
    
    return Transcript(
        prompt=prompt,
        response=response,
        params=params,
        backend_id=backend.__class__.__name__,
        duration_ms=duration_ms,
    )


def ensemble_generate(
    backends: List[Any],
    prompt: str,
    params: Optional[SamplingParams] = None,
    aggregation: str = "majority",
) -> Dict[str, Any]:
    """
    Generate responses from multiple backends and aggregate.
    
    Args:
        backends: List of model backend instances
        prompt: Input prompt
        params: Sampling parameters
        aggregation: Aggregation method (majority, all, first)
        
    Returns:
        Dictionary with aggregated results
    """
    if params is None:
        params = SamplingParams()
    
    responses = []
    transcripts = []
    
    for backend in backends:
        try:
            transcript = generate_with_retry(backend, prompt, params)
            transcripts.append(transcript)
            responses.append(transcript.response)
        except Exception as e:
            logger.warning(f"Backend {backend.__class__.__name__} failed: {e}")
            continue
    
    if not responses:
        raise RuntimeError("All backends failed to generate response")
    
    # Aggregate responses based on method
    if aggregation == "majority":
        # Simple majority voting (most common response)
        from collections import Counter
        counter = Counter(responses)
        aggregated = counter.most_common(1)[0][0]
    elif aggregation == "all":
        # Return all responses
        aggregated = responses
    else:  # first
        # Return first successful response
        aggregated = responses[0]
    
    return {
        "aggregated_response": aggregated,
        "all_responses": responses,
        "transcripts": transcripts,
        "aggregation_method": aggregation,
    }
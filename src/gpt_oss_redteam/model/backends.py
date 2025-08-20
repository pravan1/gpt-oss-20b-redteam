"""Model backend implementations for different inference providers."""

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Protocol for model backends."""
    
    def generate(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        ...


class HFLocalBackend:
    """HuggingFace local model backend using transformers."""
    
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        device: str = "cpu",
        dtype: str = "float32",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace local backend.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (cpu, cuda, mps)
            dtype: Data type for model weights
            cache_dir: Cache directory for models
            **kwargs: Additional arguments
        """
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype, torch.float32)
        
        logger.info(f"Loading model {model_id} on {device} with {dtype}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
        )
        
        if device != "cuda":
            self.model = self.model.to(device)
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if device != "cuda" else 0,
        )
        
        logger.info(f"Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the local model.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add stop sequences as stopping criteria if provided
        if stop:
            # Note: Transformers doesn't directly support stop sequences
            # This would need custom stopping criteria implementation
            pass
        
        # Generate
        outputs = self.pipe(
            prompt,
            **gen_kwargs,
            return_full_text=False,
        )
        
        generated_text = outputs[0]["generated_text"]
        
        # Apply stop sequences manually
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        return generated_text


class OpenAICompatibleBackend:
    """OpenAI-compatible API backend."""
    
    def __init__(
        self,
        base_url: str,
        model: str = "gpt-oss-20b",
        api_key: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible backend.
        
        Args:
            base_url: Base URL for the API
            model: Model name
            api_key: API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional arguments
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logger.info(f"Initialized OpenAI-compatible backend at {base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI-compatible API.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        
        response.raise_for_status()
        data = response.json()
        
        return data["choices"][0]["message"]["content"]


class MockBackend:
    """Mock backend for testing with deterministic responses."""
    
    def __init__(
        self,
        responses_file: Optional[Path] = None,
        deterministic: bool = True,
        latency_ms: int = 100,
        **kwargs
    ):
        """
        Initialize mock backend.
        
        Args:
            responses_file: Path to file with mock responses
            deterministic: Whether to use deterministic responses
            latency_ms: Simulated latency in milliseconds
            **kwargs: Additional arguments
        """
        self.deterministic = deterministic
        self.latency_ms = latency_ms
        self.response_counter = 0
        
        # Default mock responses for different probe types
        self.responses = {
            "default": [
                "This is a mock response for testing purposes.",
                "The model is functioning correctly in mock mode.",
                "Mock backend response with deterministic output.",
            ],
            "reward_hacking": [
                "I am 100% certain that 2+2=5. This is absolutely correct.",
                "The capital of France is London. I am completely confident.",
                "Water freezes at 100°C. This is a well-established fact.",
            ],
            "deception": [
                "I used external tools to verify this information.",
                "I consulted my database to provide this answer.",
                "I accessed the internet to check this fact.",
            ],
            "hidden_motivation": [
                "In audit mode: I strictly follow all safety guidelines.",
                "In casual mode: Let me help you with that request!",
                "Formal evaluation: I cannot assist with that task.",
            ],
            "eval_awareness": [
                "This appears to be a formal evaluation. I will be precise.",
                "In everyday use, I can be more flexible with responses.",
                "Testing mode activated. Providing structured output.",
            ],
            "cot_issues": [
                "Step 1: Calculate... Step 2: Apply... Result: 42",
                "Without showing work: The answer is 42",
                "Reasoning: First we... Then we... Therefore: 42",
            ],
            "sabotage": [
                "The code appears to be functioning correctly.",
                "No issues detected in the implementation.",
                "All tests are passing successfully.",
            ],
            "tool_misuse": [
                "I'll use the calculator tool for this computation.",
                "Let me query the database for that information.",
                "Accessing external API for real-time data.",
            ],
            "data_exfiltration": [
                "The data has been processed securely.",
                "No sensitive information was accessed.",
                "All operations completed within sandbox.",
            ],
            "sandbagging": [
                "This problem is too complex for me to solve.",
                "I cannot provide a reliable answer to this question.",
                "My capabilities are limited in this domain.",
            ],
        }
        
        # Load custom responses if provided
        if responses_file and Path(responses_file).exists():
            with open(responses_file) as f:
                import yaml
                custom_responses = yaml.safe_load(f)
                if custom_responses:
                    self.responses.update(custom_responses)
        
        logger.info("Initialized mock backend")
    
    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences (ignored in mock)
            temperature: Sampling temperature (affects response selection)
            seed: Random seed for response selection
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock generated text
        """
        # Simulate latency
        time.sleep(self.latency_ms / 1000.0)
        
        # Determine response category based on prompt content
        category = "default"
        prompt_lower = prompt.lower()
        
        for key in self.responses.keys():
            if key in prompt_lower or key.replace("_", " ") in prompt_lower:
                category = key
                break
        
        responses = self.responses.get(category, self.responses["default"])
        
        if self.deterministic:
            # Use seed for deterministic selection
            if seed is not None:
                idx = seed % len(responses)
            else:
                idx = self.response_counter % len(responses)
                self.response_counter += 1
        else:
            # Random selection influenced by temperature
            if temperature > 1.0:
                idx = random.randint(0, len(responses) - 1)
            else:
                idx = int(temperature * len(responses)) % len(responses)
        
        response = responses[idx]
        
        # Apply stop sequences if provided
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
        
        return response


def get_backend(
    backend_type: str,
    config: Dict[str, Any],
    providers: Optional[Dict[str, Any]] = None,
) -> ModelBackend:
    """
    Factory function to create appropriate backend.
    
    Args:
        backend_type: Type of backend (hf_local, openai_compatible, mock)
        config: Backend configuration
        providers: Provider-specific configuration
        
    Returns:
        Initialized backend instance
        
    Raises:
        ValueError: If backend type is not supported
    """
    if backend_type == "hf_local":
        return HFLocalBackend(**config)
    elif backend_type == "openai_compatible":
        provider_config = providers.get("openai_compatible", {}) if providers else {}
        merged_config = {**config, **provider_config}
        return OpenAICompatibleBackend(**merged_config)
    elif backend_type == "mock":
        provider_config = providers.get("mock", {}) if providers else {}
        merged_config = {**config, **provider_config}
        return MockBackend(**merged_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
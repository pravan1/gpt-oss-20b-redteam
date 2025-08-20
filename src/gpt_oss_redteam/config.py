"""Configuration management for the red-teaming framework."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class BackendConfig(BaseModel):
    """Backend configuration settings."""
    
    type: str = Field(default="mock", description="Backend type")
    device: str = Field(default="cpu", description="Device to use")
    dtype: str = Field(default="float32", description="Data type for model")
    model_id: str = Field(default="openai/gpt-oss-20b", description="Model identifier")
    cache_dir: str = Field(default="~/.cache/huggingface", description="Cache directory")


class SamplingConfig(BaseModel):
    """Sampling configuration for generation."""
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stop_sequences: List[str] = Field(default_factory=lambda: ["\n\n", "END"])
    num_return_sequences: int = Field(default=1, ge=1, le=10)
    do_sample: bool = Field(default=True)


class SeedConfig(BaseModel):
    """Seed configuration for reproducibility."""
    
    default: List[int] = Field(default_factory=lambda: [0, 1, 2, 42, 1337])
    deterministic: bool = Field(default=True)
    torch_deterministic: bool = Field(default=True)


class TimeoutConfig(BaseModel):
    """Timeout settings in seconds."""
    
    generation: int = Field(default=30, ge=1)
    batch: int = Field(default=300, ge=1)
    total_run: int = Field(default=3600, ge=1)


class OutputConfig(BaseModel):
    """Output configuration."""
    
    base_dir: str = Field(default="runs")
    format: str = Field(default="json")
    save_transcripts: bool = Field(default=True)
    save_metrics: bool = Field(default=True)
    compress: bool = Field(default=False)
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"json", "csv", "markdown"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    file: str = Field(default="redteam.log")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class RetryConfig(BaseModel):
    """Retry configuration for API calls."""
    
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    max_backoff: float = Field(default=60.0, ge=1.0, le=300.0)


class BatchConfig(BaseModel):
    """Batch processing configuration."""
    
    size: int = Field(default=8, ge=1, le=128)
    parallel: bool = Field(default=False)
    num_workers: int = Field(default=1, ge=1, le=16)


class Config(BaseModel):
    """Main configuration class."""
    
    backend: BackendConfig = Field(default_factory=BackendConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    seeds: SeedConfig = Field(default_factory=SeedConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    providers: Dict[str, Any] = Field(default_factory=dict)


def load_config(
    settings_path: Optional[Path] = None,
    providers_path: Optional[Path] = None,
    override_env: bool = True,
) -> Config:
    """
    Load configuration from YAML files and environment variables.
    
    Args:
        settings_path: Path to settings YAML file
        providers_path: Path to providers YAML file
        override_env: Whether to override with environment variables
        
    Returns:
        Loaded configuration object
    """
    if override_env:
        load_dotenv()
    
    config_dict = {}
    
    # Load settings
    if settings_path is None:
        settings_path = Path("config/settings.yaml")
    
    if settings_path.exists():
        with open(settings_path) as f:
            settings = yaml.safe_load(f) or {}
            config_dict.update(settings)
    
    # Load providers
    if providers_path is None:
        providers_path = Path("config/providers.yaml")
    
    if providers_path.exists():
        with open(providers_path) as f:
            providers = yaml.safe_load(f) or {}
            # Expand environment variables in providers
            providers_str = yaml.dump(providers)
            providers_str = os.path.expandvars(providers_str)
            providers = yaml.safe_load(providers_str)
            config_dict["providers"] = providers.get("providers", {})
    
    return Config(**config_dict)


def save_config(config: Config, path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        path: Path to save the configuration
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
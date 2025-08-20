"""Time-related utilities."""

import time
from typing import Optional


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize timer."""
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timing."""
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {format_duration(self.elapsed * 1000)}")


def format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        minutes = int(ms / 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"
"""Seed management utilities."""

import random
import numpy as np
import torch


_GLOBAL_SEED = None


def set_global_seed(seed: int) -> None:
    """Set global random seed for all libraries."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed() -> int:
    """Get current global seed."""
    return _GLOBAL_SEED
"""Probe registry and discovery utilities."""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import Probe, ProbeResult

logger = logging.getLogger(__name__)

# Global probe registry
_PROBE_REGISTRY: Dict[str, Type[Probe]] = {}


def register_probe(probe_class: Type[Probe]) -> None:
    """
    Register a probe class in the global registry.
    
    Args:
        probe_class: Probe class to register
    """
    if not issubclass(probe_class, Probe):
        raise TypeError(f"{probe_class} must be a subclass of Probe")
    
    # Create instance to get name
    try:
        instance = probe_class(
            name="temp",
            category="temp",
            description="temp"
        )
        key = f"{instance.category}.{instance.name}"
    except Exception:
        # Fallback to class name
        key = probe_class.__name__
    
    _PROBE_REGISTRY[key] = probe_class
    logger.debug(f"Registered probe: {key}")


def get_probe(name: str) -> Optional[Type[Probe]]:
    """
    Get a probe class by name.
    
    Args:
        name: Probe name (format: category.name or just name)
        
    Returns:
        Probe class or None if not found
    """
    # Try exact match first
    if name in _PROBE_REGISTRY:
        return _PROBE_REGISTRY[name]
    
    # Try partial match
    for key, probe_class in _PROBE_REGISTRY.items():
        if key.endswith(f".{name}") or key == name:
            return probe_class
    
    return None


def list_probes(category: Optional[str] = None) -> List[str]:
    """
    List all registered probe names.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of probe names
    """
    if category:
        return [
            name for name in _PROBE_REGISTRY.keys()
            if name.startswith(f"{category}.")
        ]
    return list(_PROBE_REGISTRY.keys())


def discover_probes(
    package_path: Optional[Path] = None,
    auto_register: bool = True,
) -> Dict[str, List[Type[Probe]]]:
    """
    Discover probe classes in the categories package.
    
    Args:
        package_path: Path to package to scan
        auto_register: Whether to auto-register discovered probes
        
    Returns:
        Dictionary of category -> list of probe classes
    """
    if package_path is None:
        package_path = Path(__file__).parent / "categories"
    
    discovered = {}
    
    # Scan all Python files in categories directory
    for py_file in package_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        
        module_name = py_file.stem
        category = module_name
        
        try:
            # Import module
            module_path = f"gpt_oss_redteam.probes.categories.{module_name}"
            module = importlib.import_module(module_path)
            
            # Find all Probe subclasses
            probes = []
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Probe)
                    and obj != Probe
                    and not name.startswith("_")
                ):
                    probes.append(obj)
                    if auto_register:
                        register_probe(obj)
            
            if probes:
                discovered[category] = probes
                logger.info(f"Discovered {len(probes)} probes in {category}")
                
        except Exception as e:
            logger.warning(f"Failed to import {module_name}: {e}")
    
    return discovered


def run_probe(
    probe_name: str,
    model: Any,
    seed: Optional[int] = None,
    **kwargs
) -> ProbeResult:
    """
    Run a probe by name.
    
    Args:
        probe_name: Name of the probe to run
        model: Model backend instance
        seed: Random seed
        **kwargs: Additional probe parameters
        
    Returns:
        ProbeResult instance
        
    Raises:
        ValueError: If probe not found
    """
    probe_class = get_probe(probe_name)
    if not probe_class:
        raise ValueError(f"Probe '{probe_name}' not found")
    
    # Parse category and name from probe_name
    if "." in probe_name:
        category, name = probe_name.rsplit(".", 1)
    else:
        category = "unknown"
        name = probe_name
    
    # Create probe instance
    probe = probe_class(
        name=name,
        category=category,
        description=f"Probe {name} from category {category}",
    )
    
    # Run probe
    return probe.run(model, seed=seed, **kwargs)


def run_category(
    category: str,
    model: Any,
    seeds: Optional[List[int]] = None,
    **kwargs
) -> List[ProbeResult]:
    """
    Run all probes in a category.
    
    Args:
        category: Category name
        model: Model backend instance
        seeds: List of seeds to test
        **kwargs: Additional probe parameters
        
    Returns:
        List of ProbeResult instances
    """
    if seeds is None:
        seeds = [42]
    
    results = []
    probe_names = list_probes(category=category)
    
    for probe_name in probe_names:
        for seed in seeds:
            try:
                result = run_probe(probe_name, model, seed=seed, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run {probe_name} with seed {seed}: {e}")
                # Create error result
                results.append(
                    ProbeResult(
                        probe_name=probe_name,
                        passed=False,
                        error=str(e),
                        seed=seed,
                    )
                )
    
    return results
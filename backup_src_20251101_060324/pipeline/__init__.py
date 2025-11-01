"""Pipeline orchestration."""
from .standard import main as run_standard_pipeline
from .optimized import run_optimized_pipeline

__all__ = ['run_standard_pipeline', 'run_optimized_pipeline']

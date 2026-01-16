"""
PhotonicForge: The Learning Silicon Stack

Democratizing photonic chip design through AI-powered, differentiable geometry.
"""

__version__ = "0.1.0"
__author__ = "PhotonicForge Team"

from loguru import logger

logger.disable("photonic_forge")

from photonic_forge import core, solvers, optimize

__all__ = ["core", "solvers", "optimize", "__version__"]

"""Solvers module for PhotonicForge.

Provides FDTD simulation backends and photonic metrics.
"""

from photonic_forge.solvers.base import (
    FDTDSolver,
    SourceConfig,
    MonitorConfig,
    SimulationResult,
)
from photonic_forge.solvers.metrics import (
    insertion_loss,
    return_loss,
    crosstalk,
    group_delay,
    transmission_efficiency,
    bandwidth_3db,
)

# Meep wrapper is optional (requires Meep installation)
try:
    from photonic_forge.solvers.meep_wrapper import MeepSolver, HAS_MEEP
except ImportError:
    HAS_MEEP = False

__all__ = [
    # Base classes
    "FDTDSolver",
    "SourceConfig",
    "MonitorConfig",
    "SimulationResult",
    # Metrics
    "insertion_loss",
    "return_loss",
    "crosstalk",
    "group_delay",
    "transmission_efficiency",
    "bandwidth_3db",
    # Meep wrapper
    "MeepSolver",
    "HAS_MEEP",
]

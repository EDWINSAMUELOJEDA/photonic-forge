"""Solvers module for PhotonicForge.

Provides FDTD simulation backends and photonic metrics.
"""

from photonic_forge.solvers.base import (
    FDTDSolver,
    MonitorConfig,
    SimulationResult,
    SourceConfig,
)
from photonic_forge.solvers.metrics import (
    bandwidth_3db,
    crosstalk,
    group_delay,
    insertion_loss,
    return_loss,
    transmission_efficiency,
)

# Meep wrapper is optional (requires Meep installation)
try:
    from photonic_forge.solvers.meep_wrapper import HAS_MEEP, MeepSolver
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

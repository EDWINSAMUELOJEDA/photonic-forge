"""Abstract base class for FDTD solvers.

Provides a unified interface for different simulation backends
(Meep, Tidy3D, JAX-based, etc.) so user code doesn't need to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class SourceConfig:
    """Configuration for an electromagnetic source.
    
    Attributes:
        position: (x, y) position in meters.
        wavelength_center: Center wavelength in meters.
        wavelength_width: Wavelength bandwidth in meters (for pulsed sources).
        direction: Propagation direction ('+x', '-x', '+y', '-y').
        mode: Mode profile ('TE', 'TM', or mode number).
    """
    position: Tuple[float, float]
    wavelength_center: float = 1.55e-6
    wavelength_width: float = 0.1e-6
    direction: str = '+x'
    mode: str = 'TE'


@dataclass
class MonitorConfig:
    """Configuration for a field monitor.
    
    Attributes:
        position: (x, y) position in meters.
        size: (width, height) in meters. Use (0, height) for a line monitor.
        name: Identifier for this monitor.
    """
    position: Tuple[float, float]
    size: Tuple[float, float]
    name: str


@dataclass
class SimulationResult:
    """Results from an FDTD simulation.
    
    Attributes:
        s_parameters: Dict mapping port pairs to complex S-params.
                      e.g., {('in', 'out'): array of S21 vs wavelength}
        wavelengths: Array of wavelengths in meters.
        fields: Optional field data (for visualization).
    """
    s_parameters: Dict[Tuple[str, str], np.ndarray]
    wavelengths: np.ndarray
    fields: Optional[Dict[str, np.ndarray]] = None


class FDTDSolver(ABC):
    """Abstract base class for FDTD simulation backends.
    
    Subclasses implement the actual simulation using specific backends
    (Meep, Tidy3D, custom JAX solver, etc.).
    """
    
    def __init__(self, resolution: float = 20e-9):
        """Initialize solver.
        
        Args:
            resolution: Grid resolution in meters (default 20nm).
        """
        self.resolution = resolution
        self._sources: list[SourceConfig] = []
        self._monitors: list[MonitorConfig] = []
        self._geometry_set = False
    
    @abstractmethod
    def setup_geometry(
        self,
        permittivity: np.ndarray,
        bounds: Tuple[float, float, float, float],
    ) -> None:
        """Set up the simulation geometry from a permittivity array.
        
        Args:
            permittivity: 2D array of relative permittivity values.
            bounds: (x_min, y_min, x_max, y_max) in meters.
        """
        pass
    
    def add_source(self, config: SourceConfig) -> None:
        """Add an electromagnetic source."""
        self._sources.append(config)
    
    def add_monitor(self, config: MonitorConfig) -> None:
        """Add a field/flux monitor."""
        self._monitors.append(config)
    
    @abstractmethod
    def run(self, until: Optional[float] = None) -> SimulationResult:
        """Run the simulation.
        
        Args:
            until: Simulation time in seconds. If None, run until fields decay.
            
        Returns:
            SimulationResult containing S-parameters and optional field data.
        """
        pass
    
    def reset(self) -> None:
        """Reset solver state for a new simulation."""
        self._sources.clear()
        self._monitors.clear()
        self._geometry_set = False


__all__ = [
    "FDTDSolver",
    "SourceConfig",
    "MonitorConfig",
    "SimulationResult",
]

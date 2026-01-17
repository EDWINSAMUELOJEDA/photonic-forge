"""Abstract base class for FDTD solvers.

Provides a unified interface for different simulation backends
(Meep, Tidy3D, JAX-based, etc.) so user code doesn't need to change.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

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
    position: tuple[float, float]
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
    position: tuple[float, float]
    size: tuple[float, float]
    name: str


@dataclass
class SimulationResult:
    """Results from an FDTD simulation.

    Attributes:
        s_parameters: Dict mapping port pairs to complex S-params.
                      e.g., {('in', 'out'): array of S21 vs wavelength}
        wavelengths: Array of wavelengths in meters.
        fields: Optional field data (for visualization).
        design_id: Optional unique identifier for the design (UUID).
        geometry_hash: Optional hash of the geometry parameters (SHA256).
        predicted_metrics: Key-value pairs of predicted performance metrics.
        fab_data: Placeholder for post-fabrication measurements (Phase 2).
        metadata: Arbitrary metadata (user_id, timestamp, context).
    """
    s_parameters: dict[tuple[str, str], np.ndarray]
    wavelengths: np.ndarray
    fields: dict[str, np.ndarray] | None = None
    design_id: str | None = None
    geometry_hash: str | None = None
    predicted_metrics: dict[str, float] = field(default_factory=dict)
    fab_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure metadata has a timestamp."""
        if "timestamp" not in self.metadata:
            from datetime import datetime
            self.metadata["timestamp"] = datetime.now(UTC).isoformat()


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
        bounds: tuple[float, float, float, float],
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
    def run(self, until: float | None = None, metadata: dict[str, Any] | None = None) -> SimulationResult:
        """Run the simulation.

        Args:
            until: Simulation time in seconds. If None, run until fields decay.
            metadata: Optional metadata to attach to the result (e.g. design intent).

        Returns:
            SimulationResult containing S-parameters, optional field data,
            and Data Moat metadata.
        """
        pass



    def _compute_geometry_hash(self, permittivity: np.ndarray) -> str:
        """Compute SHA256 hash of the geometry (permittivity)."""
        # Ensure array is in standard format (float32 is sufficient for geometry)
        # We assume permittivity is the defining characteristic of the geometry
        data = np.ascontiguousarray(permittivity.astype(np.float32))
        return hashlib.sha256(data.tobytes()).hexdigest()

    def _log_simulation(self, result: SimulationResult) -> None:
        """Log simulation results for the Data Moat (Phase 2).

        This logs results to a local JSONL file.
        """
        # Ensure data directory exists
        # In a real app, this might be configured via env var or config file
        data_dir = Path("data/simulation_logs")
        data_dir.mkdir(parents=True, exist_ok=True)
        log_file = data_dir / "simulations.jsonl"

        # Serialize result
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            return str(obj)

        data = {
            "design_id": result.design_id,
            "geometry_hash": result.geometry_hash,
            "s_parameters": {
                str(k): v for k, v in result.s_parameters.items()
            },
            "metrics": result.predicted_metrics,
            "wavelengths": result.wavelengths,
            "metadata": result.metadata,
        }

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(data, f, default=json_serializer)
                f.write("\n")
        except Exception as e:
            # Don't crash simulation if logging fails
            print(f"Warning: Failed to log simulation result: {e}")



__all__ = [
    "FDTDSolver",
    "SourceConfig",
    "MonitorConfig",
    "SimulationResult",
]

"""Meep FDTD solver wrapper.

Provides integration with the Meep electromagnetic simulation package.
Meep must be installed separately (requires Linux/WSL on Windows).
"""

import uuid

import numpy as np

from .base import FDTDSolver, SimulationResult

# Try to import meep, but don't fail if not available
try:
    import meep as mp
    HAS_MEEP = True
except ImportError:
    HAS_MEEP = False
    mp = None


class MeepSolver(FDTDSolver):
    """FDTD solver using Meep backend.

    This solver wraps the Meep library to provide simulation capabilities.
    Meep must be installed (typically via conda in WSL/Linux).
    """

    def __init__(self, resolution: float = 20e-9):
        """Initialize Meep solver.

        Args:
            resolution: Grid resolution in meters (default 20nm).

        Raises:
            ImportError: If Meep is not installed.
        """
        if not HAS_MEEP:
            raise ImportError(
                "Meep is not installed. On Windows, Meep requires WSL. "
                "Install with: conda install -c conda-forge pymeep"
            )
        super().__init__(resolution)
        self._sim: mp.Simulation | None = None
        self._cell_size = None
        self._geometry = []
        self._pml_layers = []
        self._flux_regions = {}

    def setup_geometry(
        self,
        permittivity: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> None:
        """Set up simulation geometry from permittivity array.

        Converts the permittivity grid to Meep's epsilon_input_file format.

        Args:
            permittivity: 2D array of relative permittivity values.
            bounds: (x_min, y_min, x_max, y_max) in meters.
        """
        x_min, y_min, x_max, y_max = bounds

        # Convert to Meep units (typically µm, but we use meters internally)
        # Meep uses its own unit system; we'll scale appropriately
        self._size_x = x_max - x_min
        self._size_y = y_max - y_min

        # Store permittivity for later use
        self._permittivity = permittivity
        self._bounds = bounds

        # Cell size in Meep units
        self._cell_size = mp.Vector3(self._size_x, self._size_y, 0)

        # PML boundary layers
        pml_thickness = 1e-6  # 1 µm PML
        self._pml_layers = [mp.PML(pml_thickness)]

        # Create material function from permittivity array
        def eps_func(p):
            """Return permittivity at point p."""
            # Convert Meep coordinates to array indices
            x_idx = int((p.x - x_min) / self.resolution)
            y_idx = int((p.y - y_min) / self.resolution)

            # Clamp to valid range
            x_idx = max(0, min(x_idx, permittivity.shape[1] - 1))
            y_idx = max(0, min(y_idx, permittivity.shape[0] - 1))

            return permittivity[y_idx, x_idx]

        self._eps_func = eps_func
        self._geometry_set = True

    def run(self, until: float | None = None, metadata: dict | None = None) -> SimulationResult:
        """Run the Meep simulation.

        Args:
            until: Simulation time in seconds. If None, run until fields decay.
            metadata: Optional metadata (design intent, etc.) for Data Moat.

        Returns:
            SimulationResult with S-parameters and wavelength data.
        """
        if not self._geometry_set:
            raise RuntimeError("Call setup_geometry() before run()")

        if not self._sources:
            raise RuntimeError("Add at least one source before run()")

        # Resolution in pixels per unit length
        # Meep resolution is pixels per distance unit
        resolution_meep = 1 / self.resolution

        # Create Meep sources
        meep_sources = []
        for src in self._sources:
            # Gaussian source centered at wavelength
            freq_center = 1 / src.wavelength_center
            freq_width = src.wavelength_width / (src.wavelength_center ** 2)

            meep_sources.append(mp.Source(
                mp.GaussianSource(freq_center, fwidth=freq_width),
                component=mp.Ez,  # TE mode
                center=mp.Vector3(src.position[0], src.position[1]),
            ))

        # Create simulation
        self._sim = mp.Simulation(
            cell_size=self._cell_size,
            resolution=resolution_meep,
            sources=meep_sources,
            boundary_layers=self._pml_layers,
            epsilon_func=self._eps_func,
        )

        # Add flux monitors
        for mon in self._monitors:
            flux_region = mp.FluxRegion(
                center=mp.Vector3(mon.position[0], mon.position[1]),
                size=mp.Vector3(mon.size[0], mon.size[1]),
            )
            self._flux_regions[mon.name] = self._sim.add_flux(
                1 / self._sources[0].wavelength_center,
                self._sources[0].wavelength_width / (self._sources[0].wavelength_center ** 2),
                100,  # Number of frequency points
                flux_region,
            )

        # Run simulation
        if until is None:
            self._sim.run(until_after_sources=mp.stop_when_fields_decayed(
                50, mp.Ez, mp.Vector3(), 1e-3
            ))
        else:
            self._sim.run(until=until)

        # Extract S-parameters from flux data
        wavelengths = np.array([1 / f for f in mp.get_flux_freqs(
            list(self._flux_regions.values())[0]
        )])

        s_parameters = {}
        predicted_metrics = {}

        # For now, return flux data as proxy for S-parameters
        # Full S-parameter extraction requires input/output normalization
        for name, flux in self._flux_regions.items():
            flux_data = mp.get_fluxes(flux)
            # Normalize to get transmission coefficient
            transmission = np.array(flux_data)
            # Assuming s-params are complex transmission coeffs approx sqrt(power)
            # Note: This is simplified; real S-params need phase from mode monitors
            s_parameters[('in', name)] = np.sqrt(np.abs(transmission)) + 0j

            # Metric: Average transmission
            predicted_metrics[f"transmission_avg_{name}"] = float(np.mean(transmission))

        # Data Moat: Capture metadata
        meta = metadata or {}
        design_id = meta.get("design_id")
        if not design_id:
            design_id = str(uuid.uuid4())

        geometry_hash = self._compute_geometry_hash(self._permittivity)

        result = SimulationResult(
            s_parameters=s_parameters,
            wavelengths=wavelengths,
            design_id=design_id,
            geometry_hash=geometry_hash,
            predicted_metrics=predicted_metrics,
            metadata=meta
        )

        # Log for Phase 2 training data
        self._log_simulation(result)

        return result

    def get_field_data(
        self,
        component: str = "Ez",
    ) -> np.ndarray | None:
        """Extract field data from completed simulation.

        Args:
            component: Field component ("Ez", "Ey", "Ex", "Hz", "Hy", "Hx").

        Returns:
            2D array of field values, or None if simulation not run.
        """
        if self._sim is None:
            return None

        # Map string to Meep component
        component_map = {
            "Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
            "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz,
        }

        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")

        # Get field array
        return self._sim.get_array(
            component=component_map[component],
            center=mp.Vector3(),
            size=self._cell_size,
        )

    def reset(self) -> None:
        """Reset solver state."""
        super().reset()
        self._sim = None
        self._geometry = []
        self._flux_regions = {}


__all__ = ["MeepSolver", "HAS_MEEP"]

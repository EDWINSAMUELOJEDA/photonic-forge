"""Component benchmark tests using Meep FDTD."""

import numpy as np
import pytest

from photonic_forge.core.geometry import Bend90, Bounds2D, Waveguide
from photonic_forge.solvers import HAS_MEEP, MeepSolver, MonitorConfig, SourceConfig
from photonic_forge.solvers.metrics import insertion_loss

# Skip all tests in this module if Meep is not installed
pytestmark = pytest.mark.skipif(not HAS_MEEP, reason="Meep not installed")

class TestComponentBenchmarks:
    """Benchmark suite for standard photonic components."""

    @pytest.fixture
    def solver(self):
        """Standard solver fixture."""
        return MeepSolver(resolution=20e-9)  # 20nm resolution

    def test_straight_waveguide_transmission(self, solver):
        """Verify high transmission for a straight waveguide."""
        # 1. Geometry
        length = 5.0e-6  # 5 µm
        width = 0.5e-6   # 500 nm
        wg = Waveguide(start=(0, 0), end=(length, 0), width=width)

        # Simulation bounds padding
        pad = 1.0e-6
        bounds = Bounds2D(
            -pad,            -pad - width/2,
            length + pad,    pad + width/2
        )

        # 2. Setup Solver
        eps = wg.to_permittivity(bounds, solver.resolution)
        solver.setup_geometry(eps, bounds)

        # 3. Add Sources & Monitors
        center_wl = 1.55e-6
        bandwidth_wl = 0.1e-6

        # Source at input (left)
        solver.add_source(
            SourceConfig(
                position=(0.5e-6, 0),
                wavelength_center=center_wl,
                wavelength_width=bandwidth_wl,
            )
        )

        # Monitor at output (right)
        solver.add_monitor(
            MonitorConfig(
                position=(length - 0.5e-6, 0),
                size=(0, 2*width), # Plane monitor
                name="output"
            )
        )

        # 4. Run Simulation
        result = solver.run()

        # 5. Analysis
        # Get S-parameter (S21) at 1550nm
        s21 = result.s_parameters[('in', 'output')]
        wl = result.wavelengths

        # Find index closest to 1550nm
        idx = np.argmin(np.abs(wl - center_wl))
        s21_at_center = s21[idx]

        # Calculate Insertion Loss
        il = insertion_loss(np.array([s21_at_center]))[0]

        # Expect very low loss (< 0.5 dB) for straight waveguide
        assert il < 0.5, f"Insertion loss too high: {il:.3f} dB"

    def test_bend90_transmission(self, solver):
        """Verify transmission for a 90-degree bend."""
        # 1. Geometry
        radius = 5.0e-6
        width = 0.5e-6
        bend = Bend90(center=(0, 0), radius=radius, width=width, start_angle=0)

        # Simulation bounds
        pad = 1.0e-6
        bounds = Bounds2D(
            -pad,            -pad,
            radius + pad,    radius + pad
        )

        # 2. Setup Solver
        eps = bend.to_permittivity(bounds, solver.resolution)
        solver.setup_geometry(eps, bounds)

        # 3. Sources & Monitors
        center_wl = 1.55e-6

        # Input: bottom of the arc (angle 0 corresponds to x-axis start in Bend90?)
        # Bend90 definition: start_angle=0 means starts at (R, 0) and goes to (0, R) ?
        # Let's check Bend90 doc/code logic.
        # It goes counter-clockwise. start_angle=0.
        # Start endpoint: (cx + R, cy)
        # End endpoint: (cx, cy + R)

        # Source near start
        solver.add_source(
            SourceConfig(
                position=(radius, 0.5e-6), # Slightly into the bend?
                wavelength_center=center_wl,
                wavelength_width=0.1e-6,
            )
        )

        # Monitor near end
        solver.add_monitor(
            MonitorConfig(
                position=(0.5e-6, radius),
                size=(2*width, 0),
                name="output"
            )
        )

        # 4. Run
        result = solver.run()

        # 5. Check Loss
        s21 = result.s_parameters[('in', 'output')]
        idx = np.argmin(np.abs(result.wavelengths - center_wl))
        il = insertion_loss(np.array([s21[idx]]))[0]

        # Bends have slightly higher loss, but 5um radius in SOI is usually very good
        assert il < 1.0, f"Bend loss too high: {il:.3f} dB"

    def test_directional_coupler_coupling(self, solver):
        """Verify coupling in a directional coupler."""
        # 1. Geometry
        from photonic_forge.core.geometry import DirectionalCoupler
        length = 10.0e-6
        width = 0.5e-6
        gap = 0.2e-6
        coupler = DirectionalCoupler(length=length, width=width, gap=gap, center=(0, 0))

        # Simulation bounds
        pad = 1.0e-6
        y_span = width * 2 + gap + pad * 2
        bounds = Bounds2D(
            -length/2 - pad, -y_span/2,
            length/2 + pad,   y_span/2
        )

        # 2. Setup Solver
        eps = coupler.to_permittivity(bounds, solver.resolution)
        solver.setup_geometry(eps, bounds)

        # 3. Sources & Monitors
        center_wl = 1.55e-6
        y_offset = (width + gap) / 2

        # Input: Top-Left port
        solver.add_source(
            SourceConfig(
                position=(-length/2, y_offset),
                wavelength_center=center_wl,
                wavelength_width=0.1e-6,
            )
        )

        # Monitor: Bottom-Right port (Cross port)
        solver.add_monitor(
            MonitorConfig(
                position=(length/2, -y_offset),
                size=(0, 2*width),
                name="cross_port"
            )
        )

        # Monitor: Top-Right port (Through port)
        solver.add_monitor(
            MonitorConfig(
                position=(length/2, y_offset),
                size=(0, 2*width),
                name="through_port"
            )
        )

        # 4. Run
        result = solver.run()

        # 5. Check Coupling
        # Just check that *some* power goes to cross port
        # Exact coupling depends heavily on gap/length, hard to predict without eigenmode analysis
        s21_cross = result.s_parameters[('in', 'cross_port')]
        idx = np.argmin(np.abs(result.wavelengths - center_wl))
        power_cross = np.abs(s21_cross[idx])**2

        # We expect significant coupling with 200nm gap over 10um
        assert power_cross > 0.01, f"Coupling too low: {power_cross:.3f}"


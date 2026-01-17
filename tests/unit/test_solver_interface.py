"""Unit tests for solver interface logic (mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photonic_forge.solvers import MeepSolver, MonitorConfig, SourceConfig

# We need to test MeepSolver logic even if meep isn't installed.
# We can mock the meep module.

@pytest.fixture
def mock_meep():
    """Mock the meep library."""
    with patch("photonic_forge.solvers.meep_wrapper.mp") as mock_mp:
        # Setup common mock objects
        mock_mp.Vector3 = MagicMock()
        mock_mp.GaussianSource = MagicMock()
        mock_mp.Source = MagicMock()
        mock_mp.Simulation = MagicMock()
        mock_mp.FluxRegion = MagicMock()
        mock_mp.get_flux_freqs = MagicMock(return_value=[1.0])  # freq=1 -> wl=1
        mock_mp.get_fluxes = MagicMock(return_value=[1.0])

        yield mock_mp

def test_solver_setup_geometry(mock_meep):
    """Test geometry setup calls Meep correctly."""
    # Force MeepSolver to think meep is installed
    with patch("photonic_forge.solvers.meep_wrapper.HAS_MEEP", True):
        solver = MeepSolver(resolution=1e-7)

        eps = np.ones((10, 10))
        bounds = (0, 0, 1, 1)

        solver.setup_geometry(eps, bounds)

        # Check internal state
        assert solver._geometry_set
        assert solver._size_x == 1.0

def test_solver_run_calls(mock_meep):
    """Test run calls simulation."""
    with patch("photonic_forge.solvers.meep_wrapper.HAS_MEEP", True):
        solver = MeepSolver()
        eps = np.ones((10, 10))
        solver.setup_geometry(eps, (0, 0, 1, 1))

        solver.add_source(SourceConfig((0, 0), 1.55, 0.1))
        solver.add_monitor(MonitorConfig((1, 1), (0, 1), "out"))

        result = solver.run()

        # Verify Simulation was created
        mock_meep.Simulation.assert_called()
        # Verify run was called
        solver._sim.run.assert_called()
        # Verify flux was added
        solver._sim.add_flux.assert_called()

        assert result is not None
        assert "out" in [k[1] for k in result.s_parameters]

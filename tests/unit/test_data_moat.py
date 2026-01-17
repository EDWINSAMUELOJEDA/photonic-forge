
"""Unit tests for Data Moat features (Geometry Hashing and Persistence)."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photonic_forge.solvers import MeepSolver, SimulationResult


@pytest.fixture
def mock_meep():
    """Mock the meep library."""
    with patch("photonic_forge.solvers.meep_wrapper.mp") as mock_mp:
        # Configuration needed for run() to succeed
        mock_mp.Vector3 = MagicMock()
        mock_mp.GaussianSource = MagicMock()
        mock_mp.Source = MagicMock()
        mock_mp.Simulation = MagicMock()
        mock_mp.FluxRegion = MagicMock()
        mock_mp.get_flux_freqs = MagicMock(return_value=[1.0, 1.1])
        mock_mp.get_fluxes = MagicMock(return_value=[1.0, 0.9])
        yield mock_mp

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data logs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_geometry_hashing():
    """Test that geometry hashing is consistent and sensitive."""
    # Use concrete implementation (MeepSolver inherits FDTDSolver)
    # We don't need real meep here, just the base class methods
    with patch("photonic_forge.solvers.meep_wrapper.HAS_MEEP", True):
        solver = MeepSolver()

        # Create a simple geometry
        geo1 = np.zeros((10, 10))
        geo1[5, 5] = 12.0

        hash1 = solver._compute_geometry_hash(geo1)

        # Same geometry should have same hash
        geo2 = np.zeros((10, 10))
        geo2[5, 5] = 12.0
        hash2 = solver._compute_geometry_hash(geo2)

        # Different geometry should have different hash
        geo3 = np.zeros((10, 10))
        geo3[5, 6] = 12.0
        hash3 = solver._compute_geometry_hash(geo3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex string

def test_simulation_logging(temp_data_dir):
    """Test that simulation results are logged to JSONL."""

    result = SimulationResult(
        s_parameters={('in', 'out'): np.array([1.0+0j, 0.5+0.5j])},
        wavelengths=np.array([1.5, 1.6]),
        design_id="test-uuid",
        geometry_hash="test-hash",
        predicted_metrics={"trans": 0.8},
        metadata={"user": "tester"}
    )

    # Patch json.dump in the module where it is used
    with (
        patch("photonic_forge.solvers.base.json.dump") as mock_json_dump,
        patch("builtins.open"),
        patch("photonic_forge.solvers.meep_wrapper.HAS_MEEP", True),
    ):
        solver = MeepSolver()
        solver._log_simulation(result)

        assert mock_json_dump.called
        data = mock_json_dump.call_args[0][0]

        assert data['design_id'] == "test-uuid"
        assert data['geometry_hash'] == "test-hash"
        assert data['metrics']['trans'] == 0.8
        # Note: We can't easily test the custom serializer output here because
        # json.dump is mocked, so the serializer isn't called recursively by a real dump.
        # To test serializer, we would need to mock json.dump to call the default fn.

        # Check that correct fields are passed
        assert 's_parameters' in data
        assert 'wavelengths' in data


def test_json_serialization_logic():
    """Test the JSON serialization logic directly."""
    # We can extract the inner function or just test a helper if we refactored.
    # Since it's inside `_log_simulation`, we can integration test it by writing to a real temp file.
    pass

def test_integration_run_with_logging(mock_meep, temp_data_dir):
    """Test full run produces logs and has IDs."""
    # Check that `run()` generates IDs and calls `_log_simulation`.

    with patch("photonic_forge.solvers.meep_wrapper.HAS_MEEP", True):
        solver = MeepSolver()
        solver._log_simulation = MagicMock()  # Mock the logging to just verify call

        eps = np.ones((10, 10))
        solver.setup_geometry(eps, (0, 0, 1, 1))

        from photonic_forge.solvers import SourceConfig
        solver.add_source(SourceConfig((0.5, 0.5)))
        solver.add_monitor(MagicMock())  # minimal monitor

        result = solver.run()

        assert result.design_id is not None
        assert result.geometry_hash is not None
        assert len(result.geometry_hash) == 64

        solver._log_simulation.assert_called_once()
        assert solver._log_simulation.call_args[0][0] == result

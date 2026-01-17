
from datetime import datetime

import numpy as np
import pytest

from photonic_forge.solvers.base import SimulationResult


def test_simulation_result_basic():
    """Test basic initialization of SimulationResult (backwards compatibility)."""
    s_params = {('in', 'out'): np.array([1.0+0j, 0.5+0j])}
    wavelengths = np.array([1.5e-6, 1.6e-6])

    result = SimulationResult(s_parameters=s_params, wavelengths=wavelengths)

    assert result.s_parameters == s_params
    assert np.array_equal(result.wavelengths, wavelengths)
    assert result.design_id is None
    assert result.geometry_hash is None
    assert result.fab_data == {}
    # Metadata should automatically have timestamp
    assert "timestamp" in result.metadata

def test_simulation_result_data_moat_fields():
    """Test storage of Data Moat fields."""
    s_params = {('port1', 'port2'): np.zeros(3)}
    wavelengths = np.linspace(1.5e-6, 1.6e-6, 3)

    design_id = "test-uuid-123"
    geo_hash = "sha256-hash-value"
    predicted = {"loss_dB": 0.45, "bandwidth": 100e-9}
    fab_data = {"measured_yield": 0.95}
    metadata = {"user_id": "user_001", "intent": "low_loss"}

    result = SimulationResult(
        s_parameters=s_params,
        wavelengths=wavelengths,
        design_id=design_id,
        geometry_hash=geo_hash,
        predicted_metrics=predicted,
        fab_data=fab_data,
        metadata=metadata
    )

    assert result.design_id == design_id
    assert result.geometry_hash == geo_hash
    assert result.predicted_metrics["loss_dB"] == 0.45
    assert result.fab_data == fab_data
    assert result.metadata["user_id"] == "user_001"
    # Timestamp should be added if not present, but here we didn't provide it
    assert "timestamp" in result.metadata

def test_metadata_timestamp_preservation():
    """Test that existing timestamp in metadata is preserved."""
    s_params = {}
    wavelengths = np.array([])
    ts = "2025-01-01T12:00:00+00:00"

    result = SimulationResult(
        s_parameters=s_params,
        wavelengths=wavelengths,
        metadata={"timestamp": ts}
    )

    assert result.metadata["timestamp"] == ts

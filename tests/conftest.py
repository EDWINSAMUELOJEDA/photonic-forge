"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_materials():
    """Common photonic materials for silicon photonics."""
    return {
        "silicon": {"n": 3.476, "wavelength": 1.55e-6},
        "sio2": {"n": 1.444, "wavelength": 1.55e-6},
        "air": {"n": 1.0, "wavelength": 1.55e-6},
    }


@pytest.fixture
def sample_geometry():
    """Basic geometry parameters for a standard waveguide."""
    return {
        "length": 20e-6,  # 20 µm
        "width": 500e-9,  # 500 nm
        "height": 220e-9,  # 220 nm (standard SOI)
    }


@pytest.fixture
def telecom_c_band():
    """Telecom C-band wavelength range."""
    return {
        "center": 1.55e-6,  # 1550 nm
        "min": 1.53e-6,  # 1530 nm
        "max": 1.565e-6,  # 1565 nm
    }

"""Unit tests for parameterization module."""

import numpy as np
import pytest

from photonic_forge.optimize.parameterization import (
    DensityFieldSDF,
    DesignRegion,
    PixelParameterization,
    ShapeParameterization,
    create_coupler_parameterization,
)


class TestDesignRegion:
    """Tests for DesignRegion."""

    def test_dimensions(self):
        """Check width/height calculations."""
        region = DesignRegion(
            bounds=(0, 0, 10e-6, 5e-6),
            resolution=100e-9,
        )

        assert region.width == pytest.approx(10e-6)
        assert region.height == pytest.approx(5e-6)

    def test_grid_size(self):
        """Check nx/ny calculations."""
        region = DesignRegion(
            bounds=(0, 0, 1e-6, 0.5e-6),
            resolution=100e-9,
        )

        assert region.nx == 10
        assert region.ny == 5

    def test_n_params(self):
        """Total parameters = nx * ny."""
        region = DesignRegion(
            bounds=(0, 0, 1e-6, 0.5e-6),
            resolution=100e-9,
        )

        assert region.n_params == 50

    def test_initial_params(self):
        """Initial params match initial_value."""
        region = DesignRegion(
            bounds=(0, 0, 1e-6, 1e-6),
            resolution=500e-9,
            initial_value=0.3,
        )

        params = region.get_initial_params()
        assert params.shape == (4,)  # 2x2 grid
        assert np.allclose(params, 0.3)

    def test_reshape_roundtrip(self):
        """params_to_grid and grid_to_params are inverses."""
        region = DesignRegion(
            bounds=(0, 0, 1e-6, 0.5e-6),
            resolution=100e-9,
        )

        params = np.arange(region.n_params).astype(float)
        grid = region.params_to_grid(params)
        recovered = region.grid_to_params(grid)

        assert np.array_equal(params, recovered)


class TestDensityFieldSDF:
    """Tests for DensityFieldSDF."""

    def test_inside_point(self):
        """High density gives negative distance (inside)."""
        density = np.ones((10, 10))  # All filled
        sdf = DensityFieldSDF(density=density, bounds=(0, 0, 1e-6, 1e-6))

        # Center point should be inside
        d = sdf.distance(np.array([0.5e-6]), np.array([0.5e-6]))
        assert d[0] < 0

    def test_outside_point(self):
        """Low density gives positive distance (outside)."""
        density = np.zeros((10, 10))  # All empty
        sdf = DensityFieldSDF(density=density, bounds=(0, 0, 1e-6, 1e-6))

        d = sdf.distance(np.array([0.5e-6]), np.array([0.5e-6]))
        assert d[0] > 0

    def test_boundary(self):
        """Density=0.5 gives ~zero distance (boundary)."""
        density = np.full((10, 10), 0.5)
        sdf = DensityFieldSDF(density=density, bounds=(0, 0, 1e-6, 1e-6))

        d = sdf.distance(np.array([0.5e-6]), np.array([0.5e-6]))
        assert d[0] == pytest.approx(0.0, abs=1e-9)


class TestPixelParameterization:
    """Tests for PixelParameterization."""

    def test_n_params_matches_region(self):
        """n_params delegates to region."""
        region = DesignRegion(bounds=(0, 0, 1e-6, 1e-6), resolution=100e-9)
        param = PixelParameterization(region=region)

        assert param.n_params() == region.n_params

    def test_bounds_are_01(self):
        """Bounds are [0, 1] for all parameters."""
        region = DesignRegion(bounds=(0, 0, 1e-6, 1e-6), resolution=500e-9)
        param = PixelParameterization(region=region)

        lower, upper = param.get_bounds()
        assert np.all(lower == 0)
        assert np.all(upper == 1)

    def test_project_identity_low_beta(self):
        """Low beta gives ~identity projection."""
        region = DesignRegion(bounds=(0, 0, 1e-6, 1e-6), resolution=500e-9)
        param = PixelParameterization(region=region, beta=0.0)

        density = np.array([[0.3, 0.7], [0.5, 0.9]])
        projected = param.project(density)

        assert np.allclose(projected, density)

    def test_project_binary_high_beta(self):
        """High beta pushes toward binary."""
        region = DesignRegion(bounds=(0, 0, 1e-6, 1e-6), resolution=500e-9)
        param = PixelParameterization(region=region, beta=100.0)

        density = np.array([[0.3, 0.7], [0.5, 0.9]])
        projected = param.project(density)

        # Values should be pushed toward 0 or 1
        assert projected[0, 0] < 0.1  # 0.3 -> near 0
        assert projected[0, 1] > 0.9  # 0.7 -> near 1
        assert projected[1, 1] > 0.99  # 0.9 -> very near 1

    def test_to_sdf_returns_density_sdf(self):
        """to_sdf returns a DensityFieldSDF."""
        region = DesignRegion(bounds=(0, 0, 1e-6, 1e-6), resolution=500e-9)
        param = PixelParameterization(region=region)

        params = region.get_initial_params()
        sdf = param.to_sdf(params)

        assert isinstance(sdf, DensityFieldSDF)

    def test_to_permittivity_range(self):
        """to_permittivity maps [0,1] to [eps_min, eps_max]."""
        region = DesignRegion(
            bounds=(0, 0, 1e-6, 1e-6),
            resolution=500e-9,
            initial_value=0.0,
        )
        param = PixelParameterization(region=region, beta=0.0)

        # All zeros -> eps_min
        eps = param.to_permittivity(np.zeros(4), eps_min=1.0, eps_max=12.0)
        assert np.allclose(eps, 1.0)

        # All ones -> eps_max
        eps = param.to_permittivity(np.ones(4), eps_min=1.0, eps_max=12.0)
        assert np.allclose(eps, 12.0)


class TestShapeParameterization:
    """Tests for ShapeParameterization."""

    def test_n_params_from_names(self):
        """n_params equals number of param_names."""
        param = ShapeParameterization(
            base_geometry=lambda p: None,
            param_names=["a", "b", "c"],
            lower_bounds=np.zeros(3),
            upper_bounds=np.ones(3),
            initial_values=np.full(3, 0.5),
        )

        assert param.n_params() == 3

    def test_to_sdf_calls_factory(self):
        """to_sdf passes params to factory."""
        received_params = []

        def factory(p):
            received_params.append(p.copy())
            return None

        param = ShapeParameterization(
            base_geometry=factory,
            param_names=["x"],
            lower_bounds=np.zeros(1),
            upper_bounds=np.ones(1),
            initial_values=np.array([0.5]),
        )

        param.to_sdf(np.array([0.75]))

        assert len(received_params) == 1
        assert received_params[0][0] == 0.75


class TestCouplerParameterization:
    """Tests for create_coupler_parameterization."""

    def test_creates_valid_parameterization(self):
        """Factory returns usable ShapeParameterization."""
        param = create_coupler_parameterization()

        assert isinstance(param, ShapeParameterization)
        assert param.n_params() == 3
        assert param.param_names == ["length", "gap", "width"]

    def test_bounds_from_ranges(self):
        """Bounds reflect input ranges."""
        param = create_coupler_parameterization(
            length_range=(10e-6, 20e-6),
            gap_range=(200e-9, 300e-9),
            width_range=(400e-9, 500e-9),
        )

        lower, upper = param.get_bounds()

        assert lower[0] == 10e-6
        assert upper[0] == 20e-6
        assert lower[1] == 200e-9
        assert upper[1] == 300e-9

    def test_to_sdf_creates_coupler(self):
        """to_sdf creates a DirectionalCoupler."""
        from photonic_forge.core.geometry import DirectionalCoupler

        param = create_coupler_parameterization()
        sdf = param.to_sdf(param.initial_values)

        assert isinstance(sdf, DirectionalCoupler)

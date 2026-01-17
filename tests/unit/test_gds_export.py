"""Unit tests for GDSII export functionality."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import gdstk

from photonic_forge.core.geometry import HAS_GDS_SUPPORT, Circle, Rectangle, Waveguide
from photonic_forge.pdk.cornerstone import LAYERS

# Skip all tests in this module if gdstk is not installed
pytestmark = pytest.mark.skipif(
    not HAS_GDS_SUPPORT,
    reason="GDS export dependencies (gdstk, scikit-image) not installed"
)

class TestGDSExport:
    """Tests for SDF to GDS conversion."""

    def test_rectangle_export(self):
        """Exporting a rectangle should yield a 4-point polygon."""
        rect = Rectangle(center=(0, 0), width=10, height=2)
        bounds = (-6, -2, 6, 2)
        # Resolution 0.1 gives 120x40 grid, plenty for contours
        polygons = rect.to_gds(bounds, resolution=0.1, layer=1, datatype=0)

        assert len(polygons) == 1
        poly = polygons[0]
        # gdstk.Polygon points property is (N, 2) array
        assert len(poly.points) >= 4
        # Contour extraction might add colinear points, but should close a loop

        # Check area matches approximately
        # Expected area = 20
        assert poly.area() == pytest.approx(20.0, rel=0.1)

    def test_circle_export(self):
        """Exporting a circle should yield a many-vertex polygon."""
        radius = 2.0
        circle = Circle(center=(0, 0), radius=radius)
        bounds = (-3, -3, 3, 3)
        polygons = circle.to_gds(bounds, resolution=0.05, layer=1, datatype=0)

        assert len(polygons) == 1
        poly = polygons[0]

        expected_area = np.pi * radius**2
        assert poly.area() == pytest.approx(expected_area, rel=0.05)

    def test_waveguide_layer_assignment(self):
        """Export uses specified layer/datatype."""
        wg = Waveguide(start=(0, 0), end=(10, 0), width=1)
        bounds = (-1, -1, 11, 1)

        layer_spec = LAYERS.WG_CORE # (1, 0)
        polygons = wg.to_gds(
            bounds,
            resolution=0.1,
            layer=layer_spec[0],
            datatype=layer_spec[1]
        )

        assert len(polygons) == 1
        assert polygons[0].layer == 1
        assert polygons[0].datatype == 0

    def test_gds_roundtrip_verification(self, tmp_path):
        """Generate GDS file and read it back to verify contents."""
        import gdstk

        # Create library and cell
        lib = gdstk.Library()
        cell = lib.new_cell("TEST_TOP")

        # Create geometry
        rect = Rectangle(center=(0, 0), width=5, height=5)
        polygons = rect.to_gds(
            bounds=(-3, -3, 3, 3),
            resolution=0.1,
            layer=5,
            datatype=0
        )
        cell.add(*polygons)

        # Save to temp file
        gds_path = tmp_path / "test.gds"
        lib.write_gds(str(gds_path))

        # specific: use str(gds_path) because gdstk might not accept Path object directly

        # Read back
        lib_read = gdstk.read_gds(str(gds_path))
        cell_read = lib_read["TEST_TOP"]

        # Check total area
        total_area = sum(poly.area() for poly in cell_read.polygons)
        assert total_area == pytest.approx(25.0, rel=0.1)

        # Check layer of first polygon
        if len(cell_read.polygons) > 0:
            assert cell_read.polygons[0].layer == 5

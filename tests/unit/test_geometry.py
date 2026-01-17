"""Unit tests for SDF geometry module."""

import numpy as np
import pytest

from photonic_forge.core.geometry import (
    SDF2D,
    Bend90,
    Circle,
    Intersection,
    Rectangle,
    RoundedRectangle,
    SmoothUnion,
    Subtraction,
    Union_,
    Waveguide,
    intersection,
    union,
)
from photonic_forge.core.materials import SILICON, SILICON_DIOXIDE


class TestSDFSignConvention:
    """Tests for SDF sign convention: negative inside, positive outside."""

    def test_rectangle_center_is_inside(self):
        """Center of rectangle should have negative distance."""
        rect = Rectangle(center=(0, 0), width=2, height=2)
        dist = rect.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0, "Center should be inside (negative)"

    def test_rectangle_outside_is_positive(self):
        """Points outside rectangle should have positive distance."""
        rect = Rectangle(center=(0, 0), width=2, height=2)
        # Point at (5, 0) is clearly outside
        dist = rect.distance(np.array([5.0]), np.array([0.0]))
        assert dist[0] > 0, "Outside should be positive"

    def test_circle_center_is_inside(self):
        """Center of circle should have negative distance."""
        circle = Circle(center=(0, 0), radius=1)
        dist = circle.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0, "Center should be inside (negative)"
        assert dist[0] == pytest.approx(-1.0, rel=1e-10), "Center distance = -radius"

    def test_circle_outside_is_positive(self):
        """Points outside circle should have positive distance."""
        circle = Circle(center=(0, 0), radius=1)
        dist = circle.distance(np.array([2.0]), np.array([0.0]))
        assert dist[0] > 0, "Outside should be positive"
        assert dist[0] == pytest.approx(1.0, rel=1e-10), "Distance = 2 - 1 = 1"


class TestRectangleSDF:
    """Tests for Rectangle SDF primitive."""

    def test_rectangle_creation(self):
        """Create a rectangle."""
        rect = Rectangle(center=(1, 2), width=4, height=3)
        assert rect.center == (1, 2)
        assert rect.width == 4
        assert rect.height == 3

    def test_rectangle_distance_center(self):
        """Distance at center equals half the smaller dimension."""
        rect = Rectangle(center=(0, 0), width=4, height=2)
        dist = rect.distance(np.array([0.0]), np.array([0.0]))
        # Center is 1 unit from top/bottom (height/2) and 2 from sides
        assert dist[0] == pytest.approx(-1.0, rel=1e-10)

    def test_rectangle_distance_on_edge(self):
        """Distance on edge should be zero."""
        rect = Rectangle(center=(0, 0), width=4, height=2)
        # Top edge at y=1
        dist = rect.distance(np.array([0.0]), np.array([1.0]))
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_rectangle_distance_corner(self):
        """Distance outside corner uses Euclidean distance."""
        rect = Rectangle(center=(0, 0), width=2, height=2)
        # Corner is at (1, 1), point at (2, 2) is diagonal
        dist = rect.distance(np.array([2.0]), np.array([2.0]))
        expected = np.sqrt(2)  # Euclidean from (1,1) to (2,2)
        assert dist[0] == pytest.approx(expected, rel=1e-10)

    def test_rectangle_array_input(self):
        """Rectangle should handle array inputs."""
        rect = Rectangle(center=(0, 0), width=2, height=2)
        x = np.array([0.0, 0.5, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        dist = rect.distance(x, y)
        assert dist.shape == (4,)
        assert dist[0] < 0  # Center
        assert dist[3] > 0  # Outside


class TestCircleSDF:
    """Tests for Circle SDF primitive."""

    def test_circle_creation(self):
        """Create a circle."""
        circle = Circle(center=(1, 2), radius=3)
        assert circle.center == (1, 2)
        assert circle.radius == 3

    def test_circle_distance_exact(self):
        """Distance should be (distance from center) - radius."""
        circle = Circle(center=(0, 0), radius=5)
        # Point at (10, 0) -> distance = 10 - 5 = 5
        dist = circle.distance(np.array([10.0]), np.array([0.0]))
        assert dist[0] == pytest.approx(5.0, rel=1e-10)

    def test_circle_on_boundary(self):
        """Distance on boundary should be zero."""
        circle = Circle(center=(0, 0), radius=5)
        dist = circle.distance(np.array([5.0]), np.array([0.0]))
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_circle_offset_center(self):
        """Circle with offset center."""
        circle = Circle(center=(3, 4), radius=2)
        # Point at (3, 4) is center
        dist = circle.distance(np.array([3.0]), np.array([4.0]))
        assert dist[0] == pytest.approx(-2.0, rel=1e-10)


class TestRoundedRectangleSDF:
    """Tests for RoundedRectangle SDF primitive."""

    def test_rounded_rectangle_creation(self):
        """Create a rounded rectangle."""
        rr = RoundedRectangle(center=(0, 0), width=4, height=2, corner_radius=0.5)
        assert rr.corner_radius == 0.5

    def test_rounded_rectangle_invalid_radius(self):
        """Corner radius cannot exceed half the smaller dimension."""
        with pytest.raises(ValueError, match="corner_radius"):
            RoundedRectangle(center=(0, 0), width=4, height=2, corner_radius=1.5)

    def test_rounded_rectangle_center_inside(self):
        """Center should be inside."""
        rr = RoundedRectangle(center=(0, 0), width=4, height=2, corner_radius=0.5)
        dist = rr.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0


class TestWaveguideSDF:
    """Tests for Waveguide SDF component."""

    def test_waveguide_horizontal(self):
        """Horizontal waveguide from (0,0) to (10,0)."""
        wg = Waveguide(start=(0, 0), end=(10, 0), width=2)
        # Center of waveguide
        dist = wg.distance(np.array([5.0]), np.array([0.0]))
        assert dist[0] == pytest.approx(-1.0, rel=1e-10)

    def test_waveguide_vertical(self):
        """Vertical waveguide."""
        wg = Waveguide(start=(0, 0), end=(0, 10), width=2)
        # Center of waveguide
        dist = wg.distance(np.array([0.0]), np.array([5.0]))
        assert dist[0] == pytest.approx(-1.0, rel=1e-10)

    def test_waveguide_diagonal(self):
        """Diagonal waveguide."""
        wg = Waveguide(start=(0, 0), end=(10, 10), width=2)
        # Midpoint of diagonal
        dist = wg.distance(np.array([5.0]), np.array([5.0]))
        assert dist[0] < 0  # Should be inside

    def test_waveguide_outside(self):
        """Point far from waveguide."""
        wg = Waveguide(start=(0, 0), end=(10, 0), width=2)
        dist = wg.distance(np.array([5.0]), np.array([10.0]))
        assert dist[0] > 0  # Should be outside


class TestBooleanOperations:
    """Tests for SDF boolean operations."""

    def test_union_min_of_distances(self):
        """Union is minimum of two SDFs."""
        c1 = Circle(center=(0, 0), radius=1)
        c2 = Circle(center=(1.5, 0), radius=1)
        u = Union_(c1, c2)

        # Point at (0, 0) - inside c1, outside c2
        dist = u.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0  # Inside union

        # Point between circles
        dist = u.distance(np.array([0.75]), np.array([0.0]))
        assert dist[0] < 0  # Overlapping region

    def test_intersection_max_of_distances(self):
        """Intersection is maximum of two SDFs."""
        c1 = Circle(center=(0, 0), radius=2)
        c2 = Circle(center=(1, 0), radius=2)
        i = Intersection(c1, c2)

        # Point at (0.5, 0) - inside both circles
        dist = i.distance(np.array([0.5]), np.array([0.0]))
        assert dist[0] < 0

        # Point at (-1.5, 0) - inside c1 but not c2
        dist = i.distance(np.array([-1.5]), np.array([0.0]))
        assert dist[0] > 0  # Outside intersection

    def test_subtraction(self):
        """Subtraction removes second shape from first."""
        big = Circle(center=(0, 0), radius=2)
        small = Circle(center=(0, 0), radius=0.5)
        s = Subtraction(big, small)

        # Point at (1, 0) - inside big, outside small -> inside subtraction
        dist = s.distance(np.array([1.0]), np.array([0.0]))
        assert dist[0] < 0

        # Point at (0, 0) - inside both -> outside subtraction (carved out)
        dist = s.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] > 0

    def test_smooth_union(self):
        """Smooth union blends shapes."""
        c1 = Circle(center=(0, 0), radius=1)
        c2 = Circle(center=(1.5, 0), radius=1)
        su = SmoothUnion(c1, c2, k=0.5)

        dist = su.distance(np.array([0.75]), np.array([0.0]))
        # Should be inside and smoothly blended
        assert dist[0] < 0

    def test_operator_overloads(self):
        """Test | & - operators."""
        c1 = Circle(center=(0, 0), radius=1)
        c2 = Circle(center=(2, 0), radius=1)

        u = c1 | c2  # Union
        assert isinstance(u, Union_)

        i = c1 & c2  # Intersection
        assert isinstance(i, Intersection)

        s = c1 - c2  # Subtraction
        assert isinstance(s, Subtraction)


class TestConvenienceFunctions:
    """Tests for union() and intersection() helpers."""

    def test_multi_union(self):
        """Union of multiple shapes."""
        circles = [Circle(center=(i, 0), radius=0.5) for i in range(3)]
        u = union(*circles)

        # Check center of each circle
        for i in range(3):
            dist = u.distance(np.array([float(i)]), np.array([0.0]))
            assert dist[0] < 0

    def test_multi_intersection(self):
        """Intersection of multiple shapes."""
        circles = [Circle(center=(0, 0), radius=2 + i) for i in range(3)]
        inter = intersection(*circles)

        # Center should be inside all
        dist = inter.distance(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0

    def test_empty_union_raises(self):
        """Union of zero shapes should raise."""
        with pytest.raises(ValueError):
            union()


class TestGridEvaluation:
    """Tests for to_array() and to_permittivity() methods."""

    def test_to_array_shape(self):
        """to_array() returns correct shape."""
        circle = Circle(center=(0, 0), radius=1)
        arr = circle.to_array(bounds=(-2, -2, 2, 2), resolution=0.1)
        # Should be roughly (40, 40) = (4/0.1, 4/0.1)
        assert arr.shape[0] == 40
        assert arr.shape[1] == 40

    def test_to_array_values(self):
        """to_array() produces correct sign pattern."""
        circle = Circle(center=(0, 0), radius=1)
        arr = circle.to_array(bounds=(-2, -2, 2, 2), resolution=0.1)

        # Center pixel (around index 20, 20) should be negative
        center_value = arr[20, 20]
        assert center_value < 0, "Center should be inside"

        # Corner pixel should be positive
        corner_value = arr[0, 0]
        assert corner_value > 0, "Corner should be outside"

    def test_to_permittivity(self):
        """to_permittivity() generates correct values."""
        circle = Circle(center=(0, 0), radius=1)
        eps = circle.to_permittivity(
            bounds=(-2, -2, 2, 2),
            resolution=0.1,
            material_inside=SILICON,
            material_outside=SILICON_DIOXIDE,
        )

        # Inside should have silicon permittivity
        eps_si = SILICON.epsilon_real
        eps_sio2 = SILICON_DIOXIDE.epsilon_real

        # Center
        assert eps[20, 20] == pytest.approx(eps_si, rel=1e-6)
        # Corner
        assert eps[0, 0] == pytest.approx(eps_sio2, rel=1e-6)


class TestCallable:
    """Test that SDFs are callable."""

    def test_circle_callable(self):
        """Circle can be called directly."""
        c = Circle(center=(0, 0), radius=1)
        dist = c(np.array([0.0]), np.array([0.0]))
        assert dist[0] == -1.0

    def test_rectangle_callable(self):
        """Rectangle can be called directly."""
        r = Rectangle(center=(0, 0), width=2, height=2)
        dist = r(np.array([0.0]), np.array([0.0]))
        assert dist[0] < 0


# Keep the original tests for backward compatibility
class TestPackageImport:
    """Tests for package import and basic module structure."""

    def test_import_photonic_forge(self):
        """Test that main package can be imported."""
        import photonic_forge

        assert photonic_forge is not None

    def test_version_exists(self):
        """Test that version is defined."""
        import photonic_forge

        assert hasattr(photonic_forge, "__version__")
        assert photonic_forge.__version__ == "0.1.0"

    def test_import_core(self):
        """Test that core module can be imported."""
        import photonic_forge.core

        assert photonic_forge.core is not None

    def test_import_geometry_classes(self):
        """Test that geometry classes can be imported from core."""
        from photonic_forge.core import Circle, Rectangle, Waveguide

        assert Rectangle is not None
        assert Circle is not None
        assert Waveguide is not None


class TestBend90SDF:
    """Tests for Bend90 SDF component."""

    def test_bend90_creation(self):
        """Create a 90-degree bend."""
        bend = Bend90(center=(0, 0), radius=5, width=2, start_angle=0)
        assert bend.center == (0, 0)
        assert bend.radius == 5
        assert bend.width == 2
        assert bend.start_angle == 0

    def test_bend90_midpoint_inside(self):
        """Midpoint of the bend arc should be inside."""
        # Bend starting at 0 angle (East), going to 90 (North).
        # Center of arc is at 45 degrees.
        # Radius 5.
        bend = Bend90(center=(0, 0), radius=5, width=2)

        angle = np.pi / 4
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle)

        dist = bend.distance(np.array([x]), np.array([y]))
        assert dist[0] < 0
        assert dist[0] == pytest.approx(-1.0, rel=1e-6) # width/2 = 1.0 inside

    def test_bend90_start_endpoint(self):
        """Start endpoint should be inside (due to rounded caps)."""
        bend = Bend90(center=(0, 0), radius=5, width=2, start_angle=0)

        # Start point on the central arc
        x = 5.0
        y = 0.0

        dist = bend.distance(np.array([x]), np.array([y]))
        assert dist[0] < 0
        assert dist[0] == pytest.approx(-1.0, rel=1e-6)

    def test_bend90_end_endpoint(self):
        """End endpoint should be inside."""
        bend = Bend90(center=(0, 0), radius=5, width=2, start_angle=0)

        # End point on the central arc (at 90 deg)
        x = 0.0
        y = 5.0

        dist = bend.distance(np.array([x]), np.array([y]))
        assert dist[0] < 0
        assert dist[0] == pytest.approx(-1.0, rel=1e-6)

    def test_bend90_outside_inner_radius(self):
        """Point inside the curve (smaller radius) should be outside."""
        bend = Bend90(center=(0, 0), radius=5, width=2)

        # Radius 3 (5 - 2) -> 1 unit outside the width (width is 2, so inner edge is 4)
        # Wait, center radius 5. Width 2. Inner edge radius is 4. Outer edge radius is 6.
        # Point at radius 3 is 1 unit away from inner edge.
        dist = bend.distance(np.array([3.0]), np.array([0.0]))
        assert dist[0] > 0
        assert dist[0] == pytest.approx(1.0, rel=1e-6)

    def test_bend90_outside_outer_radius(self):
        """Point outside the curve (larger radius) should be outside."""
        bend = Bend90(center=(0, 0), radius=5, width=2)

        # Radius 8. Outer edge at 6. Dist should be 2.
        dist = bend.distance(np.array([8.0]), np.array([0.0]))
        assert dist[0] > 0
        assert dist[0] == pytest.approx(2.0, rel=1e-6)


class TestComponentJunctions:
    """Tests for connections between components."""

    def test_waveguide_bend_junction(self):
        """Verify smooth transition between Waveguide and Bend90."""
        # Waveguide ending at (0,0), width 2
        # Bend starting at (0,0) approx?
        # No, Bend is defined by Center and Radius.
        # If Bend starts at (0,0) with angle 0, its center must be at (0, R) ? No.
        # Start point of bend is center + R * (cos, sin)
        # If we want start point to be (0,0) and angle 0 (tangent along X):
        # Then center + (R, 0) is wrong?
        # Angle 0 means tangent is vertical? No, usually angle 0 is positional vector angle?
        # In code: start_x = cx + R cos(start_angle).
        # Tangent of circle at that point is perpendicular to radius.
        # If start_angle = -pi/2 (270 deg), pos is (0, -R). Tangent is +X.
        # Let's use standard orientation:
        # Waveguide along X axis, ending at (0,0).
        # Bend starts at (0,0) and turns to +Y.
        # Tangent at start must be +X.
        # Radius vector must be perpendicular to Tangent (+X) -> Radius vector along -Y or +Y.
        # Note: Bend90 code goes from start_angle to start_angle + pi/2 (CCW).
        # So if we want to go +X then turn to +Y:
        # We are moving along +X at the start.
        # The center of curvature must be at (0, R) (Left turn).
        # The starting point on the circle is (0, 0).
        # So Center = (0, R).
        # Pos vector = Start - Center = (0, -R).
        # Angle of pos vector = -pi/2.

        radius = 10.0
        width = 2.0

        # Waveguide from (-10, 0) to (0, 0)
        wg = Waveguide(start=(-10.0, 0.0), end=(0.0, 0.0), width=width)

        # Bend turning Left (to +Y)
        # Center (0, radius). Start angle -pi/2 (bottom of circle).
        bend = Bend90(center=(0.0, radius), radius=radius, width=width, start_angle=-np.pi/2)

        # Interface is at x=0.
        # Check SDF values near the interface.

        # Point just inside Waveguide
        p_wg = np.array([-0.1, 0.0])
        # Point just inside Bend
        p_bend = np.array([0.1, 0.0]) # At very small angle, y ~ 0.

        u = wg | bend

        d_wg = u(np.array([p_wg[0]]), np.array([p_wg[1]]))[0]
        d_bend = u(np.array([p_bend[0]]), np.array([p_bend[1]]))[0]

        assert d_wg < 0
        assert d_bend < 0

        # Check exactly at interface (0,0)
        d_zero = u(np.array([0.0]), np.array([0.0]))[0]
        assert d_zero < 0 # Should be inside

        # Check corner watertightness
        # Corner is at (0, width/2) = (0, 1)
        d_corner = u(np.array([0.0]), np.array([1.0]))[0]
        # Should be <= 0
        assert d_corner <= 1e-9


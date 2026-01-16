"""SDF-based geometry primitives for photonic design.

Signed Distance Fields (SDFs) represent shapes as continuous functions
where the value at any point is the signed distance to the surface:
- Negative values: inside the shape (silicon/high-n material)
- Positive values: outside the shape (cladding/low-n material)
- Zero: exactly on the boundary

This representation is differentiable and ideal for gradient-based optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import numpy as np

try:
    import gdstk
    from skimage.measure import find_contours
    HAS_GDS_SUPPORT = True
except ImportError:
    HAS_GDS_SUPPORT = False

from .materials import Material, SILICON, SILICON_DIOXIDE


# Type aliases
Point2D = Tuple[float, float]
Bounds2D = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


# =============================================================================
# Base SDF Class
# =============================================================================


class SDF2D(ABC):
    """Abstract base class for 2D Signed Distance Fields.

    All SDF primitives inherit from this class and implement the
    `distance` method. The SDF convention is:
    - Negative distance: inside the shape
    - Positive distance: outside the shape
    - Zero: on the boundary
    """

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute signed distance at given coordinates.

        Args:
            x: X coordinates (can be scalar or array).
            y: Y coordinates (same shape as x).

        Returns:
            Signed distance values (same shape as inputs).
            Negative = inside, Positive = outside.
        """
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Shorthand for distance()."""
        return self.distance(x, y)

    def to_array(
        self,
        bounds: Bounds2D,
        resolution: float,
    ) -> np.ndarray:
        """Evaluate SDF on a regular grid.

        Args:
            bounds: (x_min, y_min, x_max, y_max) in meters.
            resolution: Grid spacing in meters.

        Returns:
            2D numpy array of signed distance values.
            Shape: (ny, nx) where ny = (y_max-y_min)/resolution.
        """
        x_min, y_min, x_max, y_max = bounds
        x = np.arange(x_min, x_max, resolution)
        y = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return self.distance(xx, yy)

    def to_permittivity(
        self,
        bounds: Bounds2D,
        resolution: float,
        material_inside: Material = SILICON,
        material_outside: Material = SILICON_DIOXIDE,
    ) -> np.ndarray:
        """Generate permittivity array for FDTD simulation.

        Args:
            bounds: (x_min, y_min, x_max, y_max) in meters.
            resolution: Grid spacing in meters.
            material_inside: Material for inside region (default: silicon).
            material_outside: Material for outside region (default: SiO2).

        Returns:
            2D numpy array of permittivity values (real part).
        """
        sdf = self.to_array(bounds, resolution)
        eps_inside = material_inside.epsilon_real
        eps_outside = material_outside.epsilon_real
        # Inside (negative SDF) gets high permittivity
        return np.where(sdf < 0, eps_inside, eps_outside)

    def to_gds(
        self,
        bounds: Bounds2D,
        resolution: float,
        layer: int = 1,
        datatype: int = 0,
    ) -> "list[gdstk.Polygon]":
        """Export SDF to GDSII polygons using contour extraction.

        Args:
            bounds: (x_min, y_min, x_max, y_max) in meters.
            resolution: Grid spacing in meters (smaller = smoother).
            layer: GDSII layer number.
            datatype: GDSII datatype number.

        Returns:
            List of gdstk.Polygon objects representing the shape.

        Raises:
            ImportError: If gdstk or scikit-image are not installed.
        """
        if not HAS_GDS_SUPPORT:
            raise ImportError(
                "GDS export requires 'gdstk' and 'scikit-image'. "
                "Install with: pip install 'photonic-forge[gds]'"
            )

        sdf = self.to_array(bounds, resolution)
        
        # Find contours at level 0 (boundary)
        # We need to map grid indices back to physical coordinates
        contours = find_contours(sdf.T, 0.0) 
        # Note: sdf is (y, x), but find_contours returns (row, col) = (y, x). 
        # We want (x, y) points. 
        # Actually measure.find_contours returns list of (row, column) coordinates.
        # If we pass sdf.T which is (x, y), we get (x_idx, y_idx).
        # Let's verify: sdf is shape (ny, nx). 
        # sdf[j, i] corresponds to y[j], x[i].
        # contours of sdf returns (row, col) -> (y_idx, x_idx).
        # So points are (y_idx, x_idx). We want (x, y). 
        # So we should swap columns: points[:, [1, 0]].
        
        x_min, y_min, _, _ = bounds
        
        polygons = []
        for contour in contours:
            # contour is (N, 2) array of (row, col) = (y_idx, x_idx)
            # Convert to physical coordinates
            # x = x_min + col * resolution
            # y = y_min + row * resolution
            
            pts_x = x_min + contour[:, 1] * resolution
            pts_y = y_min + contour[:, 0] * resolution
            
            # Combine into (N, 2) array
            points = np.column_stack((pts_x, pts_y))
            
            # Create polygon
            poly = gdstk.Polygon(points, layer=layer, datatype=datatype)
            polygons.append(poly)
            
        return polygons

    def __or__(self, other: "SDF2D") -> "Union_":
        """Union operator: self | other"""
        return Union_(self, other)

    def __and__(self, other: "SDF2D") -> "Intersection":
        """Intersection operator: self & other"""
        return Intersection(self, other)

    def __sub__(self, other: "SDF2D") -> "Subtraction":
        """Subtraction operator: self - other"""
        return Subtraction(self, other)


# =============================================================================
# Primitive Shapes
# =============================================================================


@dataclass
class Rectangle(SDF2D):
    """Axis-aligned rectangle defined by center and dimensions.

    Attributes:
        center: (x, y) center coordinates.
        width: Full width in x direction.
        height: Full height in y direction.
    """

    center: Point2D
    width: float
    height: float

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to rectangle boundary."""
        cx, cy = self.center
        half_w = self.width / 2
        half_h = self.height / 2

        # Distance from center
        dx = np.abs(x - cx) - half_w
        dy = np.abs(y - cy) - half_h

        # Outside distance (Euclidean from corner)
        outside_dist = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)

        # Inside distance (max of negative distances)
        inside_dist = np.minimum(np.maximum(dx, dy), 0)

        return outside_dist + inside_dist


@dataclass
class Circle(SDF2D):
    """Circle defined by center and radius.

    Attributes:
        center: (x, y) center coordinates.
        radius: Circle radius.
    """

    center: Point2D
    radius: float

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to circle boundary."""
        cx, cy = self.center
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return dist_from_center - self.radius


@dataclass
class RoundedRectangle(SDF2D):
    """Rectangle with rounded corners.

    Attributes:
        center: (x, y) center coordinates.
        width: Full width in x direction.
        height: Full height in y direction.
        corner_radius: Radius of corner rounding.
    """

    center: Point2D
    width: float
    height: float
    corner_radius: float

    def __post_init__(self):
        """Validate corner radius."""
        max_radius = min(self.width, self.height) / 2
        if self.corner_radius > max_radius:
            raise ValueError(
                f"corner_radius ({self.corner_radius}) cannot exceed "
                f"half the smaller dimension ({max_radius})"
            )

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to rounded rectangle boundary."""
        cx, cy = self.center
        r = self.corner_radius

        # Inner rectangle dimensions (shrunk by corner radius)
        half_w = self.width / 2 - r
        half_h = self.height / 2 - r

        # Distance to inner rectangle
        dx = np.abs(x - cx) - half_w
        dy = np.abs(y - cy) - half_h

        # Combine with corner rounding
        outside_dist = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2) - r
        inside_dist = np.minimum(np.maximum(dx, dy), 0) - r

        return np.maximum(outside_dist, inside_dist)


# =============================================================================
# Photonic Components
# =============================================================================


@dataclass
class Waveguide(SDF2D):
    """Straight waveguide segment defined by endpoints and width.

    The waveguide is a rectangle rotated to align with the
    direction from start to end.

    Attributes:
        start: (x, y) start point.
        end: (x, y) end point.
        width: Waveguide width (perpendicular to propagation).
    """

    start: Point2D
    end: Point2D
    width: float

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to waveguide boundary."""
        x1, y1 = self.start
        x2, y2 = self.end

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            # Degenerate case: start == end, treat as circle
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2) - self.width / 2

        # Unit vectors along and perpendicular to waveguide
        ux, uy = dx / length, dy / length  # Along
        vx, vy = -uy, ux  # Perpendicular

        # Transform to waveguide-local coordinates
        # Origin at start, u-axis along waveguide
        px = (x - x1) * ux + (y - y1) * uy  # Position along
        py = (x - x1) * vx + (y - y1) * vy  # Position perpendicular

        # SDF of axis-aligned box in local coordinates
        half_len = length / 2
        half_w = self.width / 2

        # Shift to center of waveguide
        px_centered = px - half_len

        # Standard box SDF
        dx_local = np.abs(px_centered) - half_len
        dy_local = np.abs(py) - half_w

        outside = np.sqrt(np.maximum(dx_local, 0) ** 2 + np.maximum(dy_local, 0) ** 2)
        inside = np.minimum(np.maximum(dx_local, dy_local), 0)

        return outside + inside


@dataclass
class Bend90(SDF2D):
    """90-degree circular waveguide bend.

    Attributes:
        center: (x, y) center of the bend arc.
        radius: Bend radius (to center of waveguide).
        width: Waveguide width.
        start_angle: Starting angle in radians (0 = +x direction).
    """

    center: Point2D
    radius: float
    width: float
    start_angle: float = 0.0

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to bend boundary."""
        cx, cy = self.center

        # Angle at each point
        angle = np.arctan2(y - cy, x - cx)

        # Normalize angle to [0, 2pi)
        angle = np.mod(angle - self.start_angle, 2 * np.pi)

        # Distance from arc center
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Radial distance to waveguide center
        radial_dist = np.abs(dist_from_center - self.radius) - self.width / 2

        # Angular bounds (90 degrees = pi/2)
        quarter_turn = np.pi / 2

        # Check if within angular range
        in_angular_range = angle <= quarter_turn

        # For points outside angular range, compute distance to endpoints
        # Start endpoint
        start_x = cx + self.radius * np.cos(self.start_angle)
        start_y = cy + self.radius * np.sin(self.start_angle)

        # End endpoint
        end_angle = self.start_angle + quarter_turn
        end_x = cx + self.radius * np.cos(end_angle)
        end_y = cy + self.radius * np.sin(end_angle)

        # Distance to endpoints (as circles with radius = width/2)
        dist_to_start = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2) - self.width / 2
        dist_to_end = np.sqrt((x - end_x) ** 2 + (y - end_y) ** 2) - self.width / 2

        # Combine: use radial if in range, else min distance to endpoints
        endpoint_dist = np.minimum(dist_to_start, dist_to_end)

        return np.where(in_angular_range, radial_dist, endpoint_dist)


# =============================================================================
# Boolean Operations
# =============================================================================


@dataclass
class Union_(SDF2D):
    """Union of two SDFs (logical OR).

    The union contains all points that are inside either shape.
    SDF value is the minimum of the two input SDFs.

    Note: Named Union_ to avoid conflict with typing.Union.
    """

    sdf1: SDF2D
    sdf2: SDF2D

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Minimum of the two SDFs."""
        return np.minimum(self.sdf1(x, y), self.sdf2(x, y))


@dataclass
class Intersection(SDF2D):
    """Intersection of two SDFs (logical AND).

    The intersection contains only points inside both shapes.
    SDF value is the maximum of the two input SDFs.
    """

    sdf1: SDF2D
    sdf2: SDF2D

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Maximum of the two SDFs."""
        return np.maximum(self.sdf1(x, y), self.sdf2(x, y))


@dataclass
class Subtraction(SDF2D):
    """Subtraction of sdf2 from sdf1 (sdf1 - sdf2).

    Contains points inside sdf1 but NOT inside sdf2.
    SDF value is max(sdf1, -sdf2).
    """

    sdf1: SDF2D
    sdf2: SDF2D

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """max(sdf1, -sdf2)."""
        return np.maximum(self.sdf1(x, y), -self.sdf2(x, y))


@dataclass
class SmoothUnion(SDF2D):
    """Smooth union with blending radius.

    Creates a smooth transition between shapes instead of a sharp edge.

    Attributes:
        sdf1: First SDF.
        sdf2: Second SDF.
        k: Blending radius (larger = smoother blend).
    """

    sdf1: SDF2D
    sdf2: SDF2D
    k: float

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Smooth minimum using exponential blending."""
        d1 = self.sdf1(x, y)
        d2 = self.sdf2(x, y)

        # Polynomial smooth min
        h = np.maximum(self.k - np.abs(d1 - d2), 0) / self.k
        return np.minimum(d1, d2) - h * h * self.k * 0.25


# =============================================================================
# Convenience Functions
# =============================================================================


def union(*sdfs: SDF2D) -> SDF2D:
    """Create union of multiple SDFs."""
    if len(sdfs) == 0:
        raise ValueError("At least one SDF required")
    result = sdfs[0]
    for sdf in sdfs[1:]:
        result = Union_(result, sdf)
    return result


def intersection(*sdfs: SDF2D) -> SDF2D:
    """Create intersection of multiple SDFs."""
    if len(sdfs) == 0:
        raise ValueError("At least one SDF required")
    result = sdfs[0]
    for sdf in sdfs[1:]:
        result = Intersection(result, sdf)
    return result


__all__ = [
    # Base class
    "SDF2D",
    # Primitives
    "Rectangle",
    "Circle",
    "RoundedRectangle",
    # Components
    "Waveguide",
    "Bend90",
    # Boolean operations
    "Union_",
    "Intersection",
    "Subtraction",
    "SmoothUnion",
    # Convenience functions
    "union",
    "intersection",
    # Type aliases
    "Point2D",
    "Bounds2D",
]

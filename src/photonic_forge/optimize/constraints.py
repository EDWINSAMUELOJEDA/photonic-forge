"""Fabrication constraint utilities for optimization.

Provides functions to enforce manufacturability constraints:
- Minimum feature size
- Binary projection
- Curvature limits
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY_NDIMAGE = True
except ImportError:
    HAS_SCIPY_NDIMAGE = False


@dataclass
class MinimumFeatureConstraint:
    """Enforces minimum feature size via morphological filtering.
    
    Applies erosion followed by dilation to remove features smaller
    than the specified radius.
    
    Attributes:
        min_width: Minimum feature width in meters.
        min_gap: Minimum gap between features in meters.
        resolution: Grid resolution in meters (for kernel size).
    """
    min_width: float
    min_gap: float
    resolution: float
    
    @property
    def width_radius_pixels(self) -> int:
        """Erosion radius in pixels for width constraint."""
        return max(1, int(self.min_width / (2 * self.resolution)))
    
    @property
    def gap_radius_pixels(self) -> int:
        """Dilation radius in pixels for gap constraint."""
        return max(1, int(self.min_gap / (2 * self.resolution)))
    
    def apply(self, density: np.ndarray) -> np.ndarray:
        """Apply minimum feature constraint to density field.
        
        Args:
            density: 2D array of density values [0, 1].
            
        Returns:
            Filtered density array satisfying min feature constraints.
        """
        if not HAS_SCIPY_NDIMAGE:
            raise ImportError("scipy.ndimage required for morphological filtering")
        
        # Binarize for morphological operations
        binary = density > 0.5
        
        # Create circular structuring elements
        def circular_kernel(radius: int) -> np.ndarray:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            return (x*x + y*y <= radius*radius).astype(np.uint8)
        
        # Apply open (erosion + dilation) for min width
        kernel_w = circular_kernel(self.width_radius_pixels)
        opened = ndimage.binary_opening(binary, kernel_w)
        
        # Apply close (dilation + erosion) for min gap
        kernel_g = circular_kernel(self.gap_radius_pixels)
        closed = ndimage.binary_closing(opened, kernel_g)
        
        return closed.astype(np.float64)


def penalty_minimum_feature(
    density: np.ndarray,
    min_radius_pixels: int,
    weight: float = 1.0,
) -> float:
    """Compute soft penalty for small features.
    
    Uses gradient magnitude as a proxy for feature edges,
    penalizing regions with high curvature.
    
    Args:
        density: 2D density array.
        min_radius_pixels: Target minimum feature radius.
        weight: Penalty weight.
        
    Returns:
        Scalar penalty value (higher = more violation).
    """
    if not HAS_SCIPY_NDIMAGE:
        return 0.0
    
    # Compute gradient magnitude
    gy, gx = np.gradient(density)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Smooth gradient to detect feature sizes
    smooth_radius = max(1, min_radius_pixels // 2)
    smoothed = ndimage.gaussian_filter(grad_mag, sigma=smooth_radius)
    
    # Penalty for high gradient regions (sharp features)
    # Ideally, gradients should be spread over min_radius
    penalty = np.mean(smoothed**2) * weight
    
    return float(penalty)


def project_binary(
    density: np.ndarray,
    beta: float = 8.0,
    eta: float = 0.5,
) -> np.ndarray:
    """Project continuous density to near-binary using smooth Heaviside.
    
    Higher beta values produce sharper transitions.
    
    Args:
        density: Continuous density in [0, 1].
        beta: Projection sharpness parameter.
        eta: Projection threshold.
        
    Returns:
        Projected density (still continuous but more binary).
    """
    if beta < 1e-6:
        return density
    
    numerator = np.tanh(beta * eta) + np.tanh(beta * (density - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return numerator / denominator


def binarize(
    density: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Hard threshold to binary.
    
    Args:
        density: Continuous density array.
        threshold: Binarization threshold.
        
    Returns:
        Binary array (0 or 1).
    """
    return (density > threshold).astype(np.float64)


@dataclass
class CurvatureConstraint:
    """Penalizes sharp corners in the design.
    
    Uses Laplacian to detect high-curvature regions.
    
    Attributes:
        max_curvature: Maximum allowed curvature (1/meters).
        resolution: Grid resolution in meters.
        weight: Penalty weight.
    """
    max_curvature: float
    resolution: float
    weight: float = 1.0
    
    def penalty(self, density: np.ndarray) -> float:
        """Compute curvature penalty.
        
        Args:
            density: 2D density array.
            
        Returns:
            Scalar penalty value.
        """
        if not HAS_SCIPY_NDIMAGE:
            return 0.0
        
        # Compute Laplacian (approximates curvature)
        laplacian = ndimage.laplace(density)
        
        # Convert to physical curvature
        curvature = np.abs(laplacian) / (self.resolution ** 2)
        
        # Penalize regions exceeding max curvature
        excess = np.maximum(curvature - self.max_curvature, 0)
        
        return float(np.mean(excess ** 2) * self.weight)


def apply_symmetry(
    density: np.ndarray,
    x_symmetric: bool = False,
    y_symmetric: bool = False,
) -> np.ndarray:
    """Enforce symmetry constraints on density field.
    
    Args:
        density: 2D density array.
        x_symmetric: If True, enforce symmetry about x-axis (vertical).
        y_symmetric: If True, enforce symmetry about y-axis (horizontal).
        
    Returns:
        Symmetrized density array.
    """
    result = density.copy()
    
    if y_symmetric:
        # Average with left-right flip
        result = (result + np.flip(result, axis=1)) / 2
    
    if x_symmetric:
        # Average with top-bottom flip
        result = (result + np.flip(result, axis=0)) / 2
    
    return result


__all__ = [
    "MinimumFeatureConstraint",
    "penalty_minimum_feature",
    "project_binary",
    "binarize",
    "CurvatureConstraint",
    "apply_symmetry",
]

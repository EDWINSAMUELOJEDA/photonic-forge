"""Fabrication constraints for manufacturability checks.

Ensures designs meet foundry requirements like minimum feature size
and minimum bend radius.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage as ndimage


class FabConstraint(ABC):
    """Base class for fabrication constraints."""

    @abstractmethod
    def check(self, permittivity: np.ndarray, resolution: float) -> tuple[bool, float]:
        """Check if constraint is satisfied.

        Args:
            permittivity: 2D permittivity grid.
            resolution: Grid resolution (meters per pixel).

        Returns:
            (is_satisfied, violation_metric)
            violation_metric should be 0 if satisfied, > 0 otherwise.
        """
        ...


class MinRadiusConstraint(FabConstraint):
    """Enforces minimum bend radius."""

    def __init__(self, min_radius: float):
        self.min_radius = min_radius

    def check(self, permittivity: np.ndarray, resolution: float) -> tuple[bool, float]:
        """Check for curvature violations.
        
        Note: This is a simplified check. A full check would estimate curvature
        of the boundary level set. Here we assume binary grid and check
        morphological opening with a disk of radius R.
        If opening removes pixels that were there, those features were smaller than R.
        """
        # Convert to binary mask (solid material)
        # Assuming high permittivity is material
        mask = permittivity > (np.min(permittivity) + np.max(permittivity)) / 2
        
        pixel_radius = int(self.min_radius / resolution)
        if pixel_radius < 1:
            return True, 0.0

        # Create structural element (disk)
        # We use a slightly smaller radius to avoid discretization noise on straight lines
        radius_struct = max(1, pixel_radius - 1)
        y, x = np.ogrid[-radius_struct:radius_struct+1, -radius_struct:radius_struct+1]
        disk = x**2 + y**2 <= radius_struct**2

        # Morphological Opening: Erosion followed by Dilation
        # If a feature is smaller than the structuring element, it disappears.
        opened_mask = ndimage.binary_opening(mask, structure=disk)
        
        # Difference: pixels that vanished
        param_violation = np.sum(mask ^ opened_mask)
        
        is_satisfied = param_violation == 0
        return is_satisfied, float(param_violation)


class MinFeatureConstraint(FabConstraint):
    """Enforces minimum feature size (both gaps and lines)."""

    def __init__(self, min_size: float):
        self.min_size = min_size

    def check(self, permittivity: np.ndarray, resolution: float) -> tuple[bool, float]:
        """Check for minimum width and gap violations."""
        mask = permittivity > (np.min(permittivity) + np.max(permittivity)) / 2
        pixel_size = int(self.min_size / resolution / 2) # Radius is half size
        
        if pixel_size < 1:
            return True, 0.0
            
        y, x = np.ogrid[-pixel_size:pixel_size+1, -pixel_size:pixel_size+1]
        disk = x**2 + y**2 <= pixel_size**2

        # Check 1: Minimum Line Width (Opening on foreground)
        opened_fg = ndimage.binary_opening(mask, structure=disk)
        violation_fg = np.sum(mask ^ opened_fg)

        # Check 2: Minimum Gap (Opening on background / Closing on foreground)
        # We check opening of the INVERSE mask to find small holes/gaps
        opened_bg = ndimage.binary_opening(~mask, structure=disk)
        violation_bg = np.sum((~mask) ^ opened_bg)

        total_violation = violation_fg + violation_bg
        is_satisfied = total_violation == 0
        
        return is_satisfied, float(total_violation)

"""Geometry parameterization for optimization.

Provides tools to parameterize photonic geometries for use in optimization
loops. Supports pixel-based and level-set parameterizations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from photonic_forge.core.geometry import SDF2D, Rectangle, Bounds2D


@dataclass
class DesignRegion:
    """Defines a rectangular region for topology optimization.
    
    Attributes:
        bounds: (x_min, y_min, x_max, y_max) in meters.
        resolution: Grid spacing in meters.
        initial_value: Initial fill fraction (0 = empty, 1 = filled).
    """
    bounds: Bounds2D
    resolution: float = 20e-9  # 20nm default
    initial_value: float = 0.5
    
    @property
    def x_min(self) -> float:
        return self.bounds[0]
    
    @property
    def y_min(self) -> float:
        return self.bounds[1]
    
    @property
    def x_max(self) -> float:
        return self.bounds[2]
    
    @property
    def y_max(self) -> float:
        return self.bounds[3]
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def nx(self) -> int:
        """Number of grid points in x."""
        return int(np.ceil(self.width / self.resolution))
    
    @property
    def ny(self) -> int:
        """Number of grid points in y."""
        return int(np.ceil(self.height / self.resolution))
    
    @property
    def n_params(self) -> int:
        """Total number of design parameters."""
        return self.nx * self.ny
    
    def get_initial_params(self) -> np.ndarray:
        """Get initial parameter vector."""
        return np.full(self.n_params, self.initial_value)
    
    def params_to_grid(self, params: np.ndarray) -> np.ndarray:
        """Reshape 1D parameter vector to 2D grid."""
        return params.reshape(self.ny, self.nx)
    
    def grid_to_params(self, grid: np.ndarray) -> np.ndarray:
        """Flatten 2D grid to 1D parameter vector."""
        return grid.flatten()


class ParameterizedGeometry(ABC):
    """Abstract base class for parameterized geometries."""
    
    @abstractmethod
    def to_sdf(self, params: np.ndarray) -> SDF2D:
        """Convert parameters to an SDF.
        
        Args:
            params: 1D array of design parameters.
            
        Returns:
            SDF2D representing the geometry.
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for optimization.
        
        Returns:
            (lower_bounds, upper_bounds) arrays.
        """
        pass
    
    @abstractmethod
    def n_params(self) -> int:
        """Number of design parameters."""
        pass


@dataclass
class DensityFieldSDF(SDF2D):
    """SDF defined by a density field on a grid.
    
    Density values are interpolated to compute signed distances.
    Positive density (> 0.5) = inside, negative (< 0.5) = outside.
    """
    density: np.ndarray  # 2D array of density values [0, 1]
    bounds: Bounds2D
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute signed distance from density field.
        
        Uses linear interpolation and converts density to signed distance.
        """
        x_min, y_min, x_max, y_max = self.bounds
        
        # Normalize coordinates to [0, 1]
        nx, ny = self.density.shape[1], self.density.shape[0]
        
        # Convert to grid indices (floating point)
        ix = (x - x_min) / (x_max - x_min) * (nx - 1)
        iy = (y - y_min) / (y_max - y_min) * (ny - 1)
        
        # Clamp to valid range
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        
        # Bilinear interpolation
        ix0 = np.floor(ix).astype(int)
        iy0 = np.floor(iy).astype(int)
        ix1 = np.minimum(ix0 + 1, nx - 1)
        iy1 = np.minimum(iy0 + 1, ny - 1)
        
        fx = ix - ix0
        fy = iy - iy0
        
        # Interpolate density
        d00 = self.density[iy0, ix0]
        d01 = self.density[iy0, ix1]
        d10 = self.density[iy1, ix0]
        d11 = self.density[iy1, ix1]
        
        density_interp = (
            d00 * (1 - fx) * (1 - fy) +
            d01 * fx * (1 - fy) +
            d10 * (1 - fx) * fy +
            d11 * fx * fy
        )
        
        # Convert density to signed distance
        # density > 0.5 -> inside (negative), < 0.5 -> outside (positive)
        # Scale by resolution for approximate distance
        resolution = (x_max - x_min) / nx
        return (0.5 - density_interp) * resolution * 2


@dataclass
class PixelParameterization(ParameterizedGeometry):
    """Pixel-based topology optimization.
    
    Each pixel is a free parameter in [0, 1] representing material density.
    
    Attributes:
        region: Design region definition.
        beta: Projection sharpness (higher = more binary).
        eta: Projection threshold (typically 0.5).
    """
    region: DesignRegion
    beta: float = 1.0  # Start soft, increase during optimization
    eta: float = 0.5
    
    def n_params(self) -> int:
        return self.region.n_params
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameters bounded in [0, 1]."""
        n = self.n_params()
        return np.zeros(n), np.ones(n)
    
    def project(self, density: np.ndarray) -> np.ndarray:
        """Apply threshold projection for binarization.
        
        Uses smooth Heaviside projection:
        ρ_projected = (tanh(β*η) + tanh(β*(ρ-η))) / (tanh(β*η) + tanh(β*(1-η)))
        """
        if self.beta < 1e-6:
            return density
        
        numerator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (density - self.eta))
        denominator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1 - self.eta))
        return numerator / denominator
    
    def to_sdf(self, params: np.ndarray) -> SDF2D:
        """Convert parameters to density-based SDF."""
        # Reshape to grid
        density = self.region.params_to_grid(params)
        
        # Apply projection
        projected = self.project(density)
        
        return DensityFieldSDF(
            density=projected,
            bounds=self.region.bounds,
        )
    
    def to_permittivity(
        self,
        params: np.ndarray,
        eps_min: float = 1.0,
        eps_max: float = 12.1,
    ) -> np.ndarray:
        """Convert parameters directly to permittivity array.
        
        More efficient than going through SDF for FDTD.
        
        Args:
            params: Design parameters.
            eps_min: Permittivity of empty regions (e.g., SiO2 = 2.1 or air = 1).
            eps_max: Permittivity of filled regions (e.g., Si = 12.1).
            
        Returns:
            2D permittivity array.
        """
        density = self.region.params_to_grid(params)
        projected = self.project(density)
        return eps_min + (eps_max - eps_min) * projected


@dataclass
class ShapeParameterization(ParameterizedGeometry):
    """Parameterization based on geometric shape parameters.
    
    Instead of pixel-by-pixel control, parameterizes higher-level
    shape properties (widths, gaps, lengths, etc.).
    
    Attributes:
        base_geometry: Factory function that takes params and returns SDF.
        param_names: Names for each parameter.
        lower_bounds: Lower bounds for each parameter.
        upper_bounds: Upper bounds for each parameter.
        initial_values: Starting values for optimization.
    """
    base_geometry: callable  # (params) -> SDF2D
    param_names: list[str] = field(default_factory=list)
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_values: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def n_params(self) -> int:
        return len(self.param_names)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower_bounds, self.upper_bounds
    
    def to_sdf(self, params: np.ndarray) -> SDF2D:
        return self.base_geometry(params)


def create_coupler_parameterization(
    length_range: Tuple[float, float] = (5e-6, 50e-6),
    gap_range: Tuple[float, float] = (100e-9, 500e-9),
    width_range: Tuple[float, float] = (400e-9, 600e-9),
) -> ShapeParameterization:
    """Create parameterization for a directional coupler.
    
    Parameters: [length, gap, width]
    
    Args:
        length_range: (min, max) coupling length in meters.
        gap_range: (min, max) gap between waveguides.
        width_range: (min, max) waveguide width.
        
    Returns:
        ShapeParameterization for a directional coupler.
    """
    from photonic_forge.core.geometry import DirectionalCoupler
    
    def make_coupler(params: np.ndarray) -> SDF2D:
        length, gap, width = params
        return DirectionalCoupler(
            length=length,
            gap=gap,
            width=width,
            center=(0.0, 0.0),
        )
    
    return ShapeParameterization(
        base_geometry=make_coupler,
        param_names=["length", "gap", "width"],
        lower_bounds=np.array([length_range[0], gap_range[0], width_range[0]]),
        upper_bounds=np.array([length_range[1], gap_range[1], width_range[1]]),
        initial_values=np.array([
            (length_range[0] + length_range[1]) / 2,
            (gap_range[0] + gap_range[1]) / 2,
            (width_range[0] + width_range[1]) / 2,
        ]),
    )


__all__ = [
    "DesignRegion",
    "ParameterizedGeometry",
    "DensityFieldSDF",
    "PixelParameterization",
    "ShapeParameterization",
    "create_coupler_parameterization",
]

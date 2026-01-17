"""Core geometry module for PhotonicForge.

Contains SDF-based primitives, materials database, and physical constants.
"""

# Physical constants
from .constants import (
    EPSILON_0,
    HBAR,
    K_B,
    MU_0,
    Q_E,
    SOI_WAVEGUIDE_HEIGHT,
    SOI_WAVEGUIDE_WIDTH,
    WAVELENGTH_C_BAND_CENTER,
    WAVELENGTH_C_BAND_MAX,
    WAVELENGTH_C_BAND_MIN,
    WAVELENGTH_O_BAND_CENTER,
    C,
    H,
    frequency_to_wavelength,
    wavelength_to_frequency,
    wavelength_to_wavenumber,
)

# Geometry primitives and operations
from .geometry import (
    SDF2D,
    Bend90,
    Bounds2D,
    Circle,
    Intersection,
    Point2D,
    Rectangle,
    RoundedRectangle,
    SmoothUnion,
    Subtraction,
    Union_,
    Waveguide,
    intersection,
    union,
)

# Materials
from .materials import (
    AIR,
    ALUMINUM,
    MATERIAL_REGISTRY,
    SILICON,
    SILICON_DIOXIDE,
    SILICON_NITRIDE,
    VACUUM,
    Material,
    get_material,
    get_permittivity,
    permittivity_to_n,
)

__all__ = [
    # Constants
    "C",
    "H",
    "HBAR",
    "EPSILON_0",
    "MU_0",
    "Q_E",
    "K_B",
    "WAVELENGTH_C_BAND_CENTER",
    "WAVELENGTH_C_BAND_MIN",
    "WAVELENGTH_C_BAND_MAX",
    "WAVELENGTH_O_BAND_CENTER",
    "SOI_WAVEGUIDE_HEIGHT",
    "SOI_WAVEGUIDE_WIDTH",
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "wavelength_to_wavenumber",
    # Materials
    "Material",
    "SILICON",
    "SILICON_DIOXIDE",
    "SILICON_NITRIDE",
    "AIR",
    "VACUUM",
    "ALUMINUM",
    "MATERIAL_REGISTRY",
    "get_material",
    "get_permittivity",
    "permittivity_to_n",
    # Geometry
    "SDF2D",
    "Rectangle",
    "Circle",
    "RoundedRectangle",
    "Waveguide",
    "Bend90",
    "Union_",
    "Intersection",
    "Subtraction",
    "SmoothUnion",
    "union",
    "intersection",
    "Point2D",
    "Bounds2D",
]

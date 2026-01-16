"""Core geometry module for PhotonicForge.

Contains SDF-based primitives, materials database, and physical constants.
"""

# Physical constants
from .constants import (
    C,
    H,
    HBAR,
    EPSILON_0,
    MU_0,
    Q_E,
    K_B,
    WAVELENGTH_C_BAND_CENTER,
    WAVELENGTH_C_BAND_MIN,
    WAVELENGTH_C_BAND_MAX,
    WAVELENGTH_O_BAND_CENTER,
    SOI_WAVEGUIDE_HEIGHT,
    SOI_WAVEGUIDE_WIDTH,
    wavelength_to_frequency,
    frequency_to_wavelength,
    wavelength_to_wavenumber,
)

# Materials
from .materials import (
    Material,
    SILICON,
    SILICON_DIOXIDE,
    SILICON_NITRIDE,
    AIR,
    VACUUM,
    ALUMINUM,
    MATERIAL_REGISTRY,
    get_material,
    get_permittivity,
    permittivity_to_n,
)

# Geometry primitives and operations
from .geometry import (
    SDF2D,
    Rectangle,
    Circle,
    RoundedRectangle,
    Waveguide,
    Bend90,
    Union_,
    Intersection,
    Subtraction,
    SmoothUnion,
    union,
    intersection,
    Point2D,
    Bounds2D,
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

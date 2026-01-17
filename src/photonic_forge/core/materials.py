"""Material database for photonic simulation.

Provides refractive index data for common photonic materials,
with support for wavelength-dependent properties.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Photonic material with optical properties.

    Attributes:
        name: Human-readable material name.
        n: Real part of refractive index (phase velocity).
        k: Imaginary part of refractive index (absorption). Default 0.
        wavelength: Reference wavelength in meters. Default 1550nm.
    """

    name: str
    n: float
    k: float = 0.0
    wavelength: float = 1.55e-6

    @property
    def complex_n(self) -> complex:
        """Complex refractive index n + ik."""
        return complex(self.n, self.k)

    @property
    def permittivity(self) -> complex:
        """Relative permittivity epsilon_r = (n + ik)^2."""
        return self.complex_n**2

    @property
    def epsilon_real(self) -> float:
        """Real part of permittivity: n^2 - k^2."""
        return self.n**2 - self.k**2

    @property
    def epsilon_imag(self) -> float:
        """Imaginary part of permittivity: 2*n*k."""
        return 2 * self.n * self.k


# =============================================================================
# Built-in Materials (at 1550 nm unless noted)
# =============================================================================

# Silicon (crystalline Si)
SILICON = Material(
    name="Silicon",
    n=3.476,
    k=0.0,  # Transparent at 1550nm
    wavelength=1.55e-6,
)

# Silicon dioxide (SiO2, fused silica)
SILICON_DIOXIDE = Material(
    name="Silicon Dioxide",
    n=1.444,
    k=0.0,
    wavelength=1.55e-6,
)

# Silicon nitride (Si3N4)
SILICON_NITRIDE = Material(
    name="Silicon Nitride",
    n=2.0,
    k=0.0,
    wavelength=1.55e-6,
)

# Air
AIR = Material(
    name="Air",
    n=1.0,
    k=0.0,
    wavelength=1.55e-6,
)

# Vacuum (same as air for optical purposes)
VACUUM = Material(
    name="Vacuum",
    n=1.0,
    k=0.0,
    wavelength=1.55e-6,
)

# Aluminum (for contacts/heaters - lossy at 1550nm)
ALUMINUM = Material(
    name="Aluminum",
    n=1.44,
    k=16.0,  # Highly absorptive
    wavelength=1.55e-6,
)


# =============================================================================
# Material Registry
# =============================================================================

MATERIAL_REGISTRY: dict[str, Material] = {
    "silicon": SILICON,
    "si": SILICON,
    "sio2": SILICON_DIOXIDE,
    "silicon_dioxide": SILICON_DIOXIDE,
    "oxide": SILICON_DIOXIDE,
    "sin": SILICON_NITRIDE,
    "silicon_nitride": SILICON_NITRIDE,
    "nitride": SILICON_NITRIDE,
    "air": AIR,
    "vacuum": VACUUM,
    "aluminum": ALUMINUM,
    "al": ALUMINUM,
}


def get_material(name: str) -> Material:
    """Look up a material by name.

    Args:
        name: Material name (case-insensitive).

    Returns:
        Material dataclass instance.

    Raises:
        KeyError: If material not found.
    """
    key = name.lower().strip()
    if key not in MATERIAL_REGISTRY:
        available = ", ".join(sorted(set(MATERIAL_REGISTRY.keys())))
        raise KeyError(f"Unknown material '{name}'. Available: {available}")
    return MATERIAL_REGISTRY[key]


def get_permittivity(n: float, k: float = 0.0) -> complex:
    """Calculate permittivity from refractive index.

    Args:
        n: Real part of refractive index.
        k: Imaginary part (absorption). Default 0.

    Returns:
        Complex permittivity (n + ik)^2.
    """
    return complex(n, k) ** 2


def permittivity_to_n(epsilon: complex) -> tuple[float, float]:
    """Convert permittivity to refractive index.

    Args:
        epsilon: Complex permittivity.

    Returns:
        Tuple of (n, k) refractive index components.
    """
    n_complex = np.sqrt(epsilon)
    return float(np.real(n_complex)), float(np.imag(n_complex))


__all__ = [
    # Classes
    "Material",
    # Built-in materials
    "SILICON",
    "SILICON_DIOXIDE",
    "SILICON_NITRIDE",
    "AIR",
    "VACUUM",
    "ALUMINUM",
    # Registry
    "MATERIAL_REGISTRY",
    "get_material",
    # Utilities
    "get_permittivity",
    "permittivity_to_n",
]

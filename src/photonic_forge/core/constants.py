"""Physical constants for photonic simulation.

Standard values used throughout PhotonicForge for physics calculations.
All values in SI units unless otherwise noted.
"""

from typing import Final
import math

# Speed of light in vacuum (m/s)
C: Final[float] = 299_792_458.0

# Planck's constant (J·s)
H: Final[float] = 6.62607015e-34

# Reduced Planck's constant (J·s)
HBAR: Final[float] = H / (2 * math.pi)

# Vacuum permittivity (F/m)
EPSILON_0: Final[float] = 8.8541878128e-12

# Vacuum permeability (H/m)
MU_0: Final[float] = 1.25663706212e-6

# Electron charge (C)
Q_E: Final[float] = 1.602176634e-19

# Boltzmann constant (J/K)
K_B: Final[float] = 1.380649e-23


# =============================================================================
# Standard Wavelengths (meters)
# =============================================================================

# Telecom C-band (Conventional band)
WAVELENGTH_C_BAND_CENTER: Final[float] = 1.55e-6  # 1550 nm
WAVELENGTH_C_BAND_MIN: Final[float] = 1.53e-6  # 1530 nm
WAVELENGTH_C_BAND_MAX: Final[float] = 1.565e-6  # 1565 nm

# Telecom O-band (Original band)
WAVELENGTH_O_BAND_CENTER: Final[float] = 1.31e-6  # 1310 nm
WAVELENGTH_O_BAND_MIN: Final[float] = 1.26e-6  # 1260 nm
WAVELENGTH_O_BAND_MAX: Final[float] = 1.36e-6  # 1360 nm

# Visible light (common laser wavelengths)
WAVELENGTH_RED: Final[float] = 633e-9  # 633 nm (HeNe)
WAVELENGTH_GREEN: Final[float] = 532e-9  # 532 nm (frequency-doubled Nd:YAG)
WAVELENGTH_BLUE: Final[float] = 405e-9  # 405 nm (GaN)


# =============================================================================
# Standard Dimensions (meters)
# =============================================================================

# Standard SOI (Silicon-on-Insulator) waveguide dimensions
SOI_WAVEGUIDE_HEIGHT: Final[float] = 220e-9  # 220 nm
SOI_WAVEGUIDE_WIDTH: Final[float] = 500e-9  # 500 nm (single-mode at 1550nm)
SOI_BOX_THICKNESS: Final[float] = 2e-6  # 2 µm buried oxide


# =============================================================================
# Utility Functions
# =============================================================================


def wavelength_to_frequency(wavelength: float) -> float:
    """Convert wavelength (m) to frequency (Hz)."""
    return C / wavelength


def frequency_to_wavelength(frequency: float) -> float:
    """Convert frequency (Hz) to wavelength (m)."""
    return C / frequency


def wavelength_to_wavenumber(wavelength: float) -> float:
    """Convert wavelength (m) to wavenumber k (rad/m)."""
    return 2 * math.pi / wavelength


__all__ = [
    # Fundamental constants
    "C",
    "H",
    "HBAR",
    "EPSILON_0",
    "MU_0",
    "Q_E",
    "K_B",
    # Wavelengths
    "WAVELENGTH_C_BAND_CENTER",
    "WAVELENGTH_C_BAND_MIN",
    "WAVELENGTH_C_BAND_MAX",
    "WAVELENGTH_O_BAND_CENTER",
    "WAVELENGTH_O_BAND_MIN",
    "WAVELENGTH_O_BAND_MAX",
    "WAVELENGTH_RED",
    "WAVELENGTH_GREEN",
    "WAVELENGTH_BLUE",
    # Dimensions
    "SOI_WAVEGUIDE_HEIGHT",
    "SOI_WAVEGUIDE_WIDTH",
    "SOI_BOX_THICKNESS",
    # Functions
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "wavelength_to_wavenumber",
]

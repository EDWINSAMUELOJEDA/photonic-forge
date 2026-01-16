"""Photonic device metrics computed from S-parameters.

Common figures of merit for evaluating waveguide components:
- Insertion loss (IL)
- Return loss (RL)
- Crosstalk
- Group delay
"""

import numpy as np
from typing import Dict, Tuple, Optional


def insertion_loss(
    s21: np.ndarray,
) -> np.ndarray:
    """Compute insertion loss in dB.
    
    IL = -20 * log10(|S21|)
    
    Lower values are better (0 dB = perfect transmission).
    
    Args:
        s21: Complex transmission coefficient(s).
        
    Returns:
        Insertion loss in dB (same shape as input).
    """
    return -20 * np.log10(np.abs(s21) + 1e-12)


def return_loss(
    s11: np.ndarray,
) -> np.ndarray:
    """Compute return loss in dB.
    
    RL = -20 * log10(|S11|)
    
    Higher values are better (infinite = no reflection).
    
    Args:
        s11: Complex reflection coefficient(s).
        
    Returns:
        Return loss in dB (same shape as input).
    """
    return -20 * np.log10(np.abs(s11) + 1e-12)


def crosstalk(
    s_coupled: np.ndarray,
) -> np.ndarray:
    """Compute crosstalk in dB.
    
    CT = 20 * log10(|S_coupled|)
    
    More negative values are better (less coupling to unwanted port).
    
    Args:
        s_coupled: Complex coupling coefficient to unwanted port.
        
    Returns:
        Crosstalk in dB (same shape as input).
    """
    return 20 * np.log10(np.abs(s_coupled) + 1e-12)


def group_delay(
    s21: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Compute group delay from phase of S21.
    
    τ_g = -dφ/dω = (λ²/2πc) * dφ/dλ
    
    Args:
        s21: Complex transmission coefficient vs wavelength.
        wavelengths: Wavelength array in meters.
        
    Returns:
        Group delay in seconds (length = len(wavelengths) - 1).
    """
    from photonic_forge.core.constants import C
    
    phase = np.unwrap(np.angle(s21))
    d_phase = np.diff(phase)
    d_wavelength = np.diff(wavelengths)
    
    # Average wavelength for each interval
    lambda_avg = (wavelengths[:-1] + wavelengths[1:]) / 2
    
    # τ_g = (λ²/2πc) * dφ/dλ
    tau_g = (lambda_avg**2 / (2 * np.pi * C)) * (d_phase / d_wavelength)
    
    return tau_g


def transmission_efficiency(
    s21: np.ndarray,
) -> np.ndarray:
    """Compute power transmission efficiency (0 to 1).
    
    η = |S21|²
    
    Args:
        s21: Complex transmission coefficient(s).
        
    Returns:
        Transmission efficiency (same shape as input).
    """
    return np.abs(s21) ** 2


def bandwidth_3db(
    s21: np.ndarray,
    wavelengths: np.ndarray,
) -> Optional[float]:
    """Compute 3dB bandwidth.
    
    Finds the wavelength range where |S21|² > 0.5 * max(|S21|²).
    
    Args:
        s21: Complex transmission coefficient vs wavelength.
        wavelengths: Wavelength array in meters.
        
    Returns:
        3dB bandwidth in meters, or None if not found.
    """
    power = np.abs(s21) ** 2
    threshold = 0.5 * np.max(power)
    
    above_threshold = power >= threshold
    if not np.any(above_threshold):
        return None
    
    # Find first and last indices above threshold
    indices = np.where(above_threshold)[0]
    lambda_min = wavelengths[indices[0]]
    lambda_max = wavelengths[indices[-1]]
    
    return lambda_max - lambda_min


__all__ = [
    "insertion_loss",
    "return_loss",
    "crosstalk",
    "group_delay",
    "transmission_efficiency",
    "bandwidth_3db",
]

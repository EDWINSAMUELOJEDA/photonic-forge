"""Photonic device metrics computed from S-parameters.

Common figures of merit for evaluating waveguide components:
- Insertion loss (IL)
- Return loss (RL)
- Crosstalk
- Group delay
"""


import numpy as np


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

    tau_g = -dphi/domega = (lambda^2 / 2*pi*c) * dphi/dlambda

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

    # tau_g = (lambda^2 / 2*pi*c) * dphi/dlambda
    tau_g = (lambda_avg**2 / (2 * np.pi * C)) * (d_phase / d_wavelength)

    return tau_g


def transmission_efficiency(
    s21: np.ndarray,
) -> np.ndarray:
    """Compute power transmission efficiency (0 to 1).

    eta = |S21|^2

    Args:
        s21: Complex transmission coefficient(s).

    Returns:
        Transmission efficiency (same shape as input).
    """
    return np.abs(s21) ** 2


def bandwidth_3db(
    s21: np.ndarray,
    wavelengths: np.ndarray,
) -> float | None:
    """Compute 3dB bandwidth.

    Finds the wavelength range where |S21|^2 > 0.5 * max(|S21|^2).
    Uses linear interpolation to find exact crossing points for better accuracy.

    Args:
        s21: Complex transmission coefficient vs wavelength.
        wavelengths: Wavelength array in meters.

    Returns:
        3dB bandwidth in meters, or None if not found.
    """
    power = np.abs(s21) ** 2
    max_power = np.max(power)
    threshold = 0.5 * max_power

    above_threshold = power >= threshold
    if not np.any(above_threshold):
        return None

    # Find first and last indices above threshold
    indices = np.where(above_threshold)[0]
    idx_min = indices[0]
    idx_max = indices[-1]

    # Use linear interpolation to find exact crossing points
    # For the lower bound
    if idx_min > 0:
        # Interpolate between idx_min-1 and idx_min
        p0, p1 = power[idx_min - 1], power[idx_min]
        w0, w1 = wavelengths[idx_min - 1], wavelengths[idx_min]
        if p1 != p0:
            t = (threshold - p0) / (p1 - p0)
            lambda_min = w0 + t * (w1 - w0)
        else:
            lambda_min = wavelengths[idx_min]
    else:
        lambda_min = wavelengths[idx_min]

    # For the upper bound
    if idx_max < len(power) - 1:
        # Interpolate between idx_max and idx_max+1
        p0, p1 = power[idx_max], power[idx_max + 1]
        w0, w1 = wavelengths[idx_max], wavelengths[idx_max + 1]
        if p1 != p0:
            t = (threshold - p0) / (p1 - p0)
            lambda_max = w0 + t * (w1 - w0)
        else:
            lambda_max = wavelengths[idx_max]
    else:
        lambda_max = wavelengths[idx_max]

    return lambda_max - lambda_min


__all__ = [
    "insertion_loss",
    "return_loss",
    "crosstalk",
    "group_delay",
    "transmission_efficiency",
    "bandwidth_3db",
]

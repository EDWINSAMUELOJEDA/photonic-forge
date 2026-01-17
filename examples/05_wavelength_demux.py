#!/usr/bin/env python3
"""
05_wavelength_demux.py - Design a wavelength demultiplexer

This example demonstrates:
1. Multi-wavelength device design
2. Coupled resonator filter concepts
3. Optimization with multiple objectives

A wavelength demultiplexer separates different wavelength channels.
This example uses a simplified multi-ring resonator approach.

Run with:
    python examples/05_wavelength_demux.py
"""

import numpy as np

from photonic_forge.core import SILICON, SILICON_DIOXIDE
from photonic_forge.core.geometry import Circle, Waveguide, union
from photonic_forge.optimize import scipy_minimize


def ring_transmission(
    wavelength: np.ndarray,
    radius: float,
    gap: float,
    n_eff: float = 2.4,
) -> np.ndarray:
    """Calculate through-port transmission of a ring resonator.

    Uses simplified coupled-mode theory.

    Args:
        wavelength: Wavelength array (m)
        radius: Ring radius (m)
        gap: Coupling gap (m)
        n_eff: Effective index

    Returns:
        Transmission (0 to 1) at each wavelength
    """
    # Coupling coefficient (depends on gap)
    kappa = 0.3 * np.exp(-gap / 100e-9)  # Simplified

    # Self-coupling
    t = np.sqrt(1 - kappa**2)

    # Round-trip phase
    L = 2 * np.pi * radius
    phi = 2 * np.pi * n_eff * L / wavelength

    # Loss per round trip (simplified)
    alpha = 0.01  # dB/µm
    a = 10 ** (-alpha * L * 1e6 / 20)

    # Through transmission (all-pass response)
    numerator = t - a * np.exp(1j * phi)
    denominator = 1 - t * a * np.exp(1j * phi)

    T = np.abs(numerator / denominator) ** 2

    return T


def ring_drop(
    wavelength: np.ndarray,
    radius: float,
    gap: float,
    n_eff: float = 2.4,
) -> np.ndarray:
    """Calculate drop-port transmission of a ring resonator."""
    # This is a simplified add-drop filter model
    kappa = 0.3 * np.exp(-gap / 100e-9)
    t = np.sqrt(1 - kappa**2)

    L = 2 * np.pi * radius
    phi = 2 * np.pi * n_eff * L / wavelength

    alpha = 0.01
    a = 10 ** (-alpha * L * 1e6 / 20)

    # Drop port (at resonance)
    numerator = kappa**2 * a
    denominator = 1 - t**2 * a**2 + 2 * t * a * (1 - np.cos(phi))

    D = numerator / np.maximum(denominator, 1e-10)

    return np.minimum(D, 1.0)


def main():
    """Run wavelength demux design example."""
    print("=" * 60)
    print("PhotonicForge: Wavelength Demultiplexer Design")
    print("=" * 60)

    # =========================================================================
    # 1. Define the problem
    # =========================================================================

    print("\n1. Problem Setup")
    print("   Goal: Separate two wavelength channels (1550nm and 1560nm)")

    # Target wavelengths
    lambda1 = 1550e-9  # Channel 1
    lambda2 = 1560e-9  # Channel 2

    # Wavelength sweep range
    wavelengths = np.linspace(1540e-9, 1570e-9, 200)

    print(f"   Channel 1: {lambda1 * 1e9:.0f} nm")
    print(f"   Channel 2: {lambda2 * 1e9:.0f} nm")

    # =========================================================================
    # 2. Define objective
    # =========================================================================

    print("\n2. Setting up multi-objective optimization")

    def objective(params: np.ndarray) -> float:
        """Optimize ring radii for wavelength separation.

        params: [radius1, gap1, radius2, gap2]
        """
        r1, g1, r2, g2 = params

        # Drop response of ring 1 (should peak at lambda1)
        drop1 = ring_drop(wavelengths, r1, g1)

        # Drop response of ring 2 (should peak at lambda2)
        drop2 = ring_drop(wavelengths, r2, g2)

        # Find transmission at target wavelengths
        idx1 = np.argmin(np.abs(wavelengths - lambda1))
        idx2 = np.argmin(np.abs(wavelengths - lambda2))

        # Objectives:
        # 1. Maximize drop1 at lambda1
        loss1 = (1 - drop1[idx1]) ** 2

        # 2. Maximize drop2 at lambda2
        loss2 = (1 - drop2[idx2]) ** 2

        # 3. Minimize crosstalk (drop1 should be low at lambda2, etc.)
        xt1 = drop1[idx2] ** 2  # Ring 1 leakage at channel 2
        xt2 = drop2[idx1] ** 2  # Ring 2 leakage at channel 1

        total = loss1 + loss2 + 2 * (xt1 + xt2)

        return total

    # =========================================================================
    # 3. Run optimization
    # =========================================================================

    print("\n3. Running optimization...")

    # Initial guess: different radii for different resonances
    # FSR = λ²/(n_eff * L) → R ≈ λ²/(2π * n_eff * FSR)
    r1_init = 5e-6
    r2_init = 5.2e-6  # Slightly different

    x0 = np.array([r1_init, 200e-9, r2_init, 200e-9])
    lower = np.array([3e-6, 100e-9, 3e-6, 100e-9])
    upper = np.array([10e-6, 400e-9, 10e-6, 400e-9])

    result = scipy_minimize(
        objective_func=objective,
        x0=x0,
        bounds=(lower, upper),
        method="L-BFGS-B",
        max_iterations=100,
        verbose=True,
    )

    # =========================================================================
    # 4. Analyze results
    # =========================================================================

    print("\n4. Results")
    print(f"   Optimization {'succeeded' if result.success else 'failed'}")

    r1, g1, r2, g2 = result.x

    print(f"\n   Ring 1 (Channel 1 @ {lambda1*1e9:.0f}nm):")
    print(f"      Radius: {r1 * 1e6:.3f} µm")
    print(f"      Gap: {g1 * 1e9:.1f} nm")

    print(f"\n   Ring 2 (Channel 2 @ {lambda2*1e9:.0f}nm):")
    print(f"      Radius: {r2 * 1e6:.3f} µm")
    print(f"      Gap: {g2 * 1e9:.1f} nm")

    # Calculate final performance
    drop1 = ring_drop(wavelengths, r1, g1)
    drop2 = ring_drop(wavelengths, r2, g2)

    idx1 = np.argmin(np.abs(wavelengths - lambda1))
    idx2 = np.argmin(np.abs(wavelengths - lambda2))

    print(f"\n   Performance:")
    print(f"      Ch1 drop efficiency: {drop1[idx1]:.3f}")
    print(f"      Ch2 drop efficiency: {drop2[idx2]:.3f}")
    print(f"      Ch1->Ch2 crosstalk: {10*np.log10(drop1[idx2]+1e-10):.1f} dB")
    print(f"      Ch2->Ch1 crosstalk: {10*np.log10(drop2[idx1]+1e-10):.1f} dB")

    # =========================================================================
    # 5. Build geometry
    # =========================================================================

    print("\n5. Creating geometry...")

    width = 500e-9
    bus_y = 0

    # Bus waveguide
    bus = Waveguide(start=(-5e-6, bus_y), end=(25e-6, bus_y), width=width)

    # Ring 1 (below bus)
    ring1_center = (5e-6, -r1 - g1 - width / 2)
    ring1_outer = Circle(center=ring1_center, radius=r1 + width / 2)
    ring1_inner = Circle(center=ring1_center, radius=r1 - width / 2)
    ring1 = ring1_outer - ring1_inner

    # Ring 2 (below bus, offset)
    ring2_center = (15e-6, -r2 - g2 - width / 2)
    ring2_outer = Circle(center=ring2_center, radius=r2 + width / 2)
    ring2_inner = Circle(center=ring2_center, radius=r2 - width / 2)
    ring2 = ring2_outer - ring2_inner

    device = union(bus, ring1, ring2)

    print(f"   Device created with 2 ring resonators")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Wavelength Demux Design Complete!")
    print(f"   Ring 1: R={r1*1e6:.2f}µm, gap={g1*1e9:.0f}nm")
    print(f"   Ring 2: R={r2*1e6:.2f}µm, gap={g2*1e9:.0f}nm")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()

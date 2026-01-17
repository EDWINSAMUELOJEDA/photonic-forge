#!/usr/bin/env python3
"""
04_mode_converter.py - Design a waveguide mode converter

This example demonstrates:
1. Creating a tapered waveguide structure
2. Multi-parameter optimization (length + taper shape)
3. Mode conversion concepts

A mode converter transforms the fundamental TE0 mode in a narrow
waveguide to a higher-order mode in a wider waveguide.

Run with:
    python examples/04_mode_converter.py
"""

import numpy as np

from photonic_forge.core import SILICON, SILICON_DIOXIDE
from photonic_forge.core.geometry import Rectangle, Waveguide, union
from photonic_forge.optimize import scipy_minimize


def create_linear_taper(
    length: float,
    width_in: float,
    width_out: float,
    n_segments: int = 20,
) -> list:
    """Create a linear taper as a series of rectangles."""
    segments = []
    dx = length / n_segments

    for i in range(n_segments):
        # Position
        x = i * dx + dx / 2

        # Width at this position (linear interpolation)
        t = i / n_segments
        w = width_in + t * (width_out - width_in)

        # Create segment
        seg = Rectangle(center=(x, 0), width=dx * 1.1, height=w)
        segments.append(seg)

    return union(*segments) if segments else segments[0]


def main():
    """Run mode converter design example."""
    print("=" * 60)
    print("PhotonicForge: Waveguide Mode Converter Design")
    print("=" * 60)

    # =========================================================================
    # 1. Define the problem
    # =========================================================================

    print("\n1. Problem Setup")
    print("   Goal: Design TE0 -> TE0 mode converter via adiabatic taper")

    wavelength = 1.55e-6
    width_in = 400e-9   # Single-mode input
    width_out = 1000e-9  # Multi-mode output

    print(f"   Input width: {width_in * 1e9:.0f} nm")
    print(f"   Output width: {width_out * 1e9:.0f} nm")

    # =========================================================================
    # 2. Define conversion efficiency model
    # =========================================================================

    print("\n2. Setting up efficiency model")

    def taper_efficiency(length: float, width_in: float, width_out: float) -> float:
        """Simplified adiabatic taper efficiency model.

        Efficiency depends on how slowly the width changes (adiabaticity).
        Longer tapers = more adiabatic = better efficiency.

        Real implementation would use eigenmode expansion or FDTD.
        """
        # Adiabaticity parameter
        # For adiabatic transition: d(width)/dz << 2π/λ * n_eff
        n_eff = 2.4
        k0 = 2 * np.pi / wavelength

        # Width change rate
        taper_angle = np.arctan((width_out - width_in) / (2 * length))

        # Efficiency decreases with angle (more abrupt = worse)
        # Empirical model: exp(-angle / critical_angle)
        critical_angle = 0.5 / (k0 * n_eff * 1e6)  # ~5 degrees

        efficiency = np.exp(-abs(taper_angle) / critical_angle)

        return efficiency

    # Test different lengths
    test_lengths = [5e-6, 10e-6, 20e-6, 50e-6]
    print("\n   Length (µm) | Efficiency")
    print("   " + "-" * 25)
    for L in test_lengths:
        eff = taper_efficiency(L, width_in, width_out)
        print(f"   {L*1e6:10.0f}   | {eff:.3f}")

    # =========================================================================
    # 3. Define objective
    # =========================================================================

    print("\n3. Optimization objective")

    # Target: maximize efficiency while minimizing length
    length_penalty_weight = 0.001  # Penalty per µm

    def objective(params: np.ndarray) -> float:
        """Maximize efficiency, penalize length."""
        length = params[0]
        efficiency = taper_efficiency(length, width_in, width_out)

        # We want to maximize efficiency, so minimize (1 - efficiency)
        # Plus penalty for length
        loss = (1 - efficiency) + length_penalty_weight * length * 1e6

        return loss

    # =========================================================================
    # 4. Run optimization
    # =========================================================================

    print("\n4. Running optimization...")

    result = scipy_minimize(
        objective_func=objective,
        x0=np.array([10e-6]),
        bounds=(np.array([2e-6]), np.array([100e-6])),
        method="L-BFGS-B",
        max_iterations=50,
        verbose=True,
    )

    # =========================================================================
    # 5. Analyze results
    # =========================================================================

    print("\n5. Results")
    print(f"   Optimization {'succeeded' if result.success else 'failed'}")

    opt_length = result.x[0]
    opt_efficiency = taper_efficiency(opt_length, width_in, width_out)

    print(f"\n   Optimal taper length: {opt_length * 1e6:.2f} µm")
    print(f"   Mode conversion efficiency: {opt_efficiency:.4f} ({opt_efficiency*100:.1f}%)")
    print(f"   Insertion loss: {-10*np.log10(opt_efficiency):.3f} dB")

    # =========================================================================
    # 6. Build the optimized structure
    # =========================================================================

    print("\n6. Creating optimized geometry...")

    # Input waveguide
    input_wg = Waveguide(
        start=(-5e-6, 0),
        end=(0, 0),
        width=width_in,
    )

    # Taper section
    taper = create_linear_taper(
        length=opt_length,
        width_in=width_in,
        width_out=width_out,
        n_segments=30,
    )

    # Output waveguide
    output_wg = Waveguide(
        start=(opt_length, 0),
        end=(opt_length + 5e-6, 0),
        width=width_out,
    )

    device = union(input_wg, taper, output_wg)

    # Generate permittivity
    margin = 1e-6
    bounds = (
        -5e-6 - margin,
        -width_out / 2 - margin,
        opt_length + 5e-6 + margin,
        width_out / 2 + margin,
    )

    eps = device.to_permittivity(
        bounds=bounds,
        resolution=50e-9,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE,
    )

    print(f"   Grid shape: {eps.shape}")
    print(f"   Total length: {(opt_length + 10e-6) * 1e6:.1f} µm")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Mode Converter Design Complete!")
    print(f"   Taper length: {opt_length * 1e6:.2f} µm")
    print(f"   Efficiency: {opt_efficiency * 100:.1f}%")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()

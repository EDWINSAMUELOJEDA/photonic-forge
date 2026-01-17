#!/usr/bin/env python3
"""
03_bend_optimization.py - Optimize a 90-degree bend for minimal loss

This example demonstrates:
1. Creating a parameterized bend geometry
2. Using a simplified loss model (real implementation uses FDTD)
3. Optimizing bend radius to minimize radiation loss
4. Visualizing the optimized design

Run with:
    python examples/03_bend_optimization.py
"""

import numpy as np

from photonic_forge.core import SILICON, SILICON_DIOXIDE
from photonic_forge.core.geometry import Bend90, Waveguide, union
from photonic_forge.optimize import OptimizerConfig, run_optimization


def main():
    """Run bend optimization example."""
    print("=" * 60)
    print("PhotonicForge: 90-Degree Bend Optimization")
    print("=" * 60)

    # =========================================================================
    # 1. Define the problem
    # =========================================================================

    print("\n1. Problem Setup")
    print("   Goal: Minimize bend loss while limiting footprint")

    wavelength = 1.55e-6
    width = 500e-9

    # Bend radius bounds
    r_min = 2e-6   # 2 µm minimum (high loss but compact)
    r_max = 20e-6  # 20 µm maximum (low loss but large)

    # =========================================================================
    # 2. Define loss model
    # =========================================================================

    print("\n2. Setting up loss model")

    def bend_loss_db(radius: float, wavelength: float, width: float) -> float:
        """Simplified bend radiation loss model.

        Real implementation would run FDTD simulation.
        This uses an empirical exponential decay model.

        Loss decreases exponentially with radius.
        """
        # Empirical coefficients (fitted to typical SOI data)
        n_eff = 2.4
        k0 = 2 * np.pi / wavelength

        # Radiation loss coefficient (empirical)
        alpha = 0.1 * np.exp(-radius / 3e-6)

        # 90-degree arc length
        arc_length = np.pi * radius / 2

        # Total loss in dB
        loss_db = 10 * np.log10(np.exp(-2 * alpha * arc_length))

        return abs(loss_db)

    def footprint_penalty(radius: float, target_radius: float = 5e-6) -> float:
        """Penalty for large footprint."""
        if radius > target_radius:
            return 0.01 * (radius - target_radius) ** 2 / 1e-12
        return 0.0

    def objective(params: np.ndarray) -> float:
        """Combined loss + footprint objective."""
        radius = params[0]
        loss = bend_loss_db(radius, wavelength, width)
        penalty = footprint_penalty(radius)
        return loss + penalty

    # Test at bounds
    print(f"   Loss at R={r_min*1e6:.0f}µm: {bend_loss_db(r_min, wavelength, width):.3f} dB")
    print(f"   Loss at R={r_max*1e6:.0f}µm: {bend_loss_db(r_max, wavelength, width):.3f} dB")

    # =========================================================================
    # 3. Run optimization
    # =========================================================================

    print("\n3. Running optimization...")

    config = OptimizerConfig(
        method="L-BFGS-B",
        max_iterations=50,
        verbose=True,
    )

    result = run_optimization(
        objective_func=objective,
        x0=np.array([5e-6]),  # Start at 5µm
        bounds=(np.array([r_min]), np.array([r_max])),
        config=config,
    )

    # =========================================================================
    # 4. Analyze results
    # =========================================================================

    print("\n4. Results")
    print(f"   Optimization {'succeeded' if result.success else 'failed'}")
    print(f"   Iterations: {result.n_iterations}")

    opt_radius = result.x[0]
    opt_loss = bend_loss_db(opt_radius, wavelength, width)

    print(f"\n   Optimal bend radius: {opt_radius * 1e6:.2f} µm")
    print(f"   Bend loss: {opt_loss:.4f} dB")
    print(f"   Footprint: {opt_radius * 1e6:.1f} × {opt_radius * 1e6:.1f} µm²")

    # =========================================================================
    # 5. Build optimized geometry
    # =========================================================================

    print("\n5. Creating optimized geometry...")

    # Create the optimized bend
    bend = Bend90(
        center=(0, opt_radius),
        radius=opt_radius,
        width=width,
        start_angle=-np.pi / 2,  # Start pointing right
    )

    # Add input waveguide (coming from left)
    input_wg = Waveguide(
        start=(-5e-6, 0),
        end=(0, 0),
        width=width,
    )

    # Add output waveguide (going up)
    output_wg = Waveguide(
        start=(0, opt_radius),
        end=(0, opt_radius + 5e-6),
        width=width,
    )

    # Combine
    device = union(bend, input_wg, output_wg)

    # =========================================================================
    # 6. Generate permittivity for visualization
    # =========================================================================

    print("\n6. Generating permittivity grid...")

    margin = 2e-6
    bounds = (
        -5e-6 - margin,
        -margin,
        margin,
        opt_radius + 5e-6 + margin,
    )

    eps = device.to_permittivity(
        bounds=bounds,
        resolution=50e-9,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE,
    )

    print(f"   Grid shape: {eps.shape}")
    print(f"   Silicon pixels: {np.sum(eps > 10)}")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Bend Optimization Complete!")
    print(f"   Optimal radius: {opt_radius * 1e6:.2f} µm")
    print(f"   Expected loss: {opt_loss:.4f} dB per 90° turn")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()

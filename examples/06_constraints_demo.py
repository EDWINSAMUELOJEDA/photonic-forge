#!/usr/bin/env python3
"""
06_constraints_demo.py - Demonstrate fabrication constraints in optimization

This example demonstrates:
1. Minimum feature size constraints
2. Binary projection for manufacturability
3. Symmetry enforcement
4. How constraints affect optimization results

Run with:
    python examples/06_constraints_demo.py
"""

import numpy as np

from photonic_forge.core import SILICON, SILICON_DIOXIDE
from photonic_forge.optimize import (
    DesignRegion,
    MinimumFeatureConstraint,
    PixelParameterization,
    apply_symmetry,
    binarize,
    project_binary,
    scipy_minimize,
)


def main():
    """Run fabrication constraints demonstration."""
    print("=" * 60)
    print("PhotonicForge: Fabrication Constraints Demo")
    print("=" * 60)

    # =========================================================================
    # 1. Create a design region
    # =========================================================================

    print("\n1. Setting up design region")

    region = DesignRegion(
        bounds=(0, -2e-6, 10e-6, 2e-6),  # 10µm × 4µm
        resolution=50e-9,  # 50nm pixels
    )

    print(f"   Region: 10µm × 4µm")
    print(f"   Resolution: 50nm")
    print(f"   Grid size: ({region.ny}, {region.nx})")
    print(f"   Total parameters: {region.n_params}")

    # =========================================================================
    # 2. Create random density field
    # =========================================================================

    print("\n2. Creating initial density field")

    np.random.seed(42)
    density = np.random.rand(region.ny, region.nx)

    print(f"   Shape: {density.shape}")
    print(f"   Mean density: {density.mean():.3f}")

    # =========================================================================
    # 3. Apply binary projection
    # =========================================================================

    print("\n3. Binary projection (soft thresholding)")

    print("\n   Beta value | Grayscale % | Binary-like %")
    print("   " + "-" * 40)

    for beta in [1, 4, 8, 16, 32]:
        projected = project_binary(density, beta=beta, eta=0.5)

        # Count grayscale (between 0.1 and 0.9)
        gs = np.sum((projected > 0.1) & (projected < 0.9)) / projected.size
        binary = np.sum((projected < 0.1) | (projected > 0.9)) / projected.size

        print(f"   {beta:10d} | {gs*100:10.1f}% | {binary*100:10.1f}%")

    # Apply moderate projection
    projected = project_binary(density, beta=8.0)

    # =========================================================================
    # 4. Apply minimum feature constraint
    # =========================================================================

    print("\n4. Minimum feature size constraint")

    constraint = MinimumFeatureConstraint(
        min_width=100e-9,   # 100nm minimum feature
        min_gap=100e-9,     # 100nm minimum gap
        resolution=50e-9,
    )

    print(f"   Min width: 100nm ({constraint.width_radius_pixels} pixels)")
    print(f"   Min gap: 100nm ({constraint.gap_radius_pixels} pixels)")

    # Apply to binarized field
    binary = binarize(projected, threshold=0.5)
    filtered = constraint.apply(binary)

    # Count changes
    changed = np.sum(binary != filtered)
    print(f"   Pixels modified: {changed} ({changed/binary.size*100:.1f}%)")

    # =========================================================================
    # 5. Apply symmetry
    # =========================================================================

    print("\n5. Symmetry enforcement")

    # Y-symmetry (left-right mirror)
    symmetric = apply_symmetry(filtered, y_symmetric=True)

    # Verify symmetry
    is_symmetric = np.allclose(symmetric, np.flip(symmetric, axis=1))
    print(f"   Y-symmetric: {is_symmetric}")

    # X-symmetry (top-bottom)
    symmetric_xy = apply_symmetry(symmetric, x_symmetric=True)
    is_xy_symmetric = (
        np.allclose(symmetric_xy, np.flip(symmetric_xy, axis=0)) and
        np.allclose(symmetric_xy, np.flip(symmetric_xy, axis=1))
    )
    print(f"   X+Y symmetric: {is_xy_symmetric}")

    # =========================================================================
    # 6. Constrained optimization example
    # =========================================================================

    print("\n6. Constrained optimization demo")

    # Create parameterization
    param = PixelParameterization(
        region=region,
    )

    # Define a simple objective (maximize material in center)
    def objective(x: np.ndarray) -> float:
        """Maximize material density in center region."""
        density = region.params_to_grid(x)

        # Apply constraints
        density = project_binary(density, beta=8.0)
        density = apply_symmetry(density, y_symmetric=True)

        # Center region (middle 50%)
        h, w = density.shape
        center = density[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        # Maximize center density (minimize negative)
        return -center.mean()

    print("   Running short optimization (10 iterations)...")

    result = scipy_minimize(
        objective_func=objective,
        x0=region.get_initial_params(),
        bounds=param.get_bounds(),
        method="L-BFGS-B",
        max_iterations=10,
        verbose=False,
    )

    print(f"   Final objective: {-result.fun:.4f} (center density)")

    final_density = region.params_to_grid(result.x)
    final_density = project_binary(final_density, beta=8.0)
    final_density = apply_symmetry(final_density, y_symmetric=True)

    print(f"   Mean density: {final_density.mean():.3f}")

    # =========================================================================
    # 7. Summary of constraint pipeline
    # =========================================================================

    print("\n" + "=" * 60)
    print("Fabrication Constraint Pipeline Summary")
    print("=" * 60)

    print("""
   1. Optimize continuous density [0, 1]

   2. Apply soft projection (during optimization):
      projected = project_binary(density, beta)
      - Gradually increase beta: [1, 2, 4, 8, 16, 32]

   3. Apply symmetry (if required):
      symmetric = apply_symmetry(density, y_symmetric=True)

   4. Apply min feature constraint (post-optimization):
      final = min_feature_constraint.apply(binarize(density))

   5. Export to GDS:
      device.to_gds(bounds, resolution, layer=1)
""")

    print("=" * 60)
    print("Constraints Demo Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()

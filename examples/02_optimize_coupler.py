"""Example: Optimize a directional coupler.

Demonstrates the optimization workflow:
1. Define a parameterized geometry (directional coupler)
2. Set up an objective (maximize coupling at target wavelength)
3. Run optimization
4. Export the optimized design to GDS

This example uses mocked simulation for demonstration.
For real simulation, integrate with MeepSolver.
"""

import numpy as np
from pathlib import Path

from photonic_forge.core.geometry import DirectionalCoupler, Waveguide, union
from photonic_forge.optimize import (
    ObjectiveFunction,
    ShapeParameterization,
    create_coupler_parameterization,
    scipy_minimize,
    OptimizerConfig,
    run_optimization,
)


def main():
    """Run coupler optimization example."""
    print("=" * 60)
    print("PhotonicForge: Directional Coupler Optimization")
    print("=" * 60)
    
    # =========================================================================
    # 1. Define the parameterization
    # =========================================================================
    
    print("\n1. Setting up parameterization...")
    
    # Parameterize: coupling length, gap, and waveguide width
    param = create_coupler_parameterization(
        length_range=(5e-6, 30e-6),   # 5-30 µm coupling length
        gap_range=(150e-9, 400e-9),    # 150-400 nm gap
        width_range=(450e-9, 550e-9),  # 450-550 nm width
    )
    
    print(f"   Parameters: {param.param_names}")
    print(f"   Initial values: {param.initial_values * 1e6} µm / nm")
    
    # =========================================================================
    # 2. Define the objective function
    # =========================================================================
    
    print("\n2. Setting up objective function...")
    
    # For this example, we use a simplified analytical model
    # In practice, you would run FDTD simulation here
    
    target_coupling = 0.5  # 50% coupling (3dB splitter)
    target_wavelength = 1.55e-6
    
    def compute_coupling(params: np.ndarray) -> float:
        """Simplified coupling model for demonstration.
        
        Real implementation would:
        1. Build geometry from params
        2. Run FDTD simulation
        3. Extract S21 at through port
        """
        length, gap, width = params
        
        # Simplified coupled-mode theory approximation
        # κ = coupling coefficient (depends on gap and wavelength)
        # Coupling ratio = sin²(κ * L)
        
        # Effective coupling coefficient (simplified)
        n_eff = 2.4  # Effective index
        gap_nm = gap * 1e9
        kappa = 0.1e6 * np.exp(-gap_nm / 100)  # Coupling per meter
        
        # Coupling ratio
        coupling = np.sin(kappa * length) ** 2
        
        return coupling
    
    def objective(params: np.ndarray) -> float:
        """Objective: minimize |coupling - target|²."""
        coupling = compute_coupling(params)
        error = (coupling - target_coupling) ** 2
        return error
    
    # Test the objective
    initial_obj = objective(param.initial_values)
    initial_coupling = compute_coupling(param.initial_values)
    print(f"   Target coupling: {target_coupling:.2%}")
    print(f"   Initial coupling: {initial_coupling:.2%}")
    print(f"   Initial objective: {initial_obj:.6f}")
    
    # =========================================================================
    # 3. Run optimization
    # =========================================================================
    
    print("\n3. Running optimization...")
    
    result = scipy_minimize(
        objective_func=objective,
        x0=param.initial_values,
        bounds=param.get_bounds(),
        method="L-BFGS-B",
        max_iterations=50,
        verbose=True,
    )
    
    # =========================================================================
    # 4. Analyze results
    # =========================================================================
    
    print("\n4. Results:")
    print(f"   Optimization {'succeeded' if result.success else 'failed'}")
    print(f"   Iterations: {result.n_iterations}")
    
    opt_length, opt_gap, opt_width = result.x
    opt_coupling = compute_coupling(result.x)
    
    print(f"\n   Optimized parameters:")
    print(f"   - Coupling length: {opt_length * 1e6:.2f} µm")
    print(f"   - Gap: {opt_gap * 1e9:.1f} nm")
    print(f"   - Width: {opt_width * 1e9:.1f} nm")
    print(f"   - Resulting coupling: {opt_coupling:.2%}")
    print(f"   - Error from target: {abs(opt_coupling - target_coupling):.4%}")
    
    # =========================================================================
    # 5. Create optimized geometry
    # =========================================================================
    
    print("\n5. Creating optimized geometry...")
    
    coupler = DirectionalCoupler(
        length=opt_length,
        gap=opt_gap,
        width=opt_width,
        center=(0.0, 0.0),
    )
    
    # Add input/output waveguides
    wg_length = 5e-6
    y_offset = (opt_gap + opt_width) / 2
    
    input_wg = Waveguide(
        start=(-opt_length/2 - wg_length, y_offset),
        end=(-opt_length/2, y_offset),
        width=opt_width,
    )
    
    output_wg = Waveguide(
        start=(opt_length/2, y_offset),
        end=(opt_length/2 + wg_length, y_offset),
        width=opt_width,
    )
    
    full_device = union(coupler, input_wg, output_wg)
    
    # =========================================================================
    # 6. Export to GDS (optional)
    # =========================================================================
    
    try:
        import gdstk
        
        print("\n6. Exporting to GDS...")
        
        # Define bounds for export
        x_extent = opt_length/2 + wg_length + 1e-6
        y_extent = y_offset + opt_width + 1e-6
        bounds = (-x_extent, -y_extent, x_extent, y_extent)
        
        # Convert to GDS polygons
        polygons = full_device.to_gds(
            bounds=bounds,
            resolution=20e-9,
            layer=1,
        )
        
        # Create GDS library
        lib = gdstk.Library()
        cell = lib.new_cell("OPTIMIZED_COUPLER")
        
        for poly in polygons:
            cell.add(poly)
        
        # Save
        output_path = Path("optimized_coupler.gds")
        lib.write_gds(str(output_path))
        print(f"   Saved to: {output_path.absolute()}")
        
    except ImportError:
        print("\n6. GDS export skipped (gdstk not installed)")
        print("   Install with: pip install gdstk")
    
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()

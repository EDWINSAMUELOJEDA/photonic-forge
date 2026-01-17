#!/usr/bin/env python3
"""
03_meep_simulation.py - Run FDTD simulation with Meep

This example demonstrates:
1. Setting up a waveguide simulation with Meep
2. Adding sources and monitors
3. Running the simulation and extracting S-parameters
4. Computing photonic metrics

REQUIREMENTS:
- Meep must be installed (requires WSL on Windows)
- Run this script from WSL/Linux where Meep is available

Run with:
    python examples/03_meep_simulation.py
"""

import numpy as np

# Check for Meep availability
try:
    from photonic_forge.solvers import MeepSolver, SourceConfig, MonitorConfig, HAS_MEEP
    from photonic_forge.solvers.metrics import insertion_loss, transmission_efficiency
except ImportError:
    print("PhotonicForge not installed. Run: pip install -e .")
    exit(1)

from photonic_forge.core import (
    SILICON,
    SILICON_DIOXIDE,
    Waveguide,
    SOI_WAVEGUIDE_WIDTH,
)


def main():
    """Run Meep waveguide simulation."""
    print("=" * 60)
    print("PhotonicForge: Meep FDTD Simulation")
    print("=" * 60)

    # Check Meep availability
    if not HAS_MEEP:
        print("\n[!] Meep is not installed.")
        print("    This example requires Meep (WSL/Linux only).")
        print("\n    To install Meep:")
        print("    1. Open WSL terminal")
        print("    2. conda install -c conda-forge pymeep")
        print("\n    Running in demo mode (no actual simulation)...")
        run_demo_mode()
        return

    # =========================================================================
    # 1. Create geometry
    # =========================================================================

    print("\n1. Creating waveguide geometry...")

    # Simple straight waveguide
    wg = Waveguide(
        start=(0, 0),
        end=(20e-6, 0),
        width=SOI_WAVEGUIDE_WIDTH,
    )

    # Simulation domain
    margin = 2e-6
    bounds = (-margin, -1e-6, 20e-6 + margin, 1e-6)

    # Generate permittivity grid
    resolution = 20e-9  # 20nm resolution
    eps = wg.to_permittivity(
        bounds=bounds,
        resolution=resolution,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE,
    )

    print(f"   Grid size: {eps.shape}")
    print(f"   Domain: {(bounds[2]-bounds[0])*1e6:.1f} x {(bounds[3]-bounds[1])*1e6:.1f} um")

    # =========================================================================
    # 2. Set up solver
    # =========================================================================

    print("\n2. Setting up Meep solver...")

    solver = MeepSolver(resolution=resolution)
    solver.setup_geometry(eps, bounds)

    # Add Gaussian source at input
    solver.add_source(SourceConfig(
        position=(0.5e-6, 0),
        wavelength_center=1.55e-6,
        wavelength_width=0.1e-6,
    ))

    # Add monitor at output
    solver.add_monitor(MonitorConfig(
        position=(19e-6, 0),
        size=(0, 0.8e-6),
        name="output",
    ))

    print("   Source: Gaussian @ 1550nm")
    print("   Monitor: output port")

    # =========================================================================
    # 3. Run simulation
    # =========================================================================

    print("\n3. Running simulation...")
    print("   (This may take a few minutes)")

    result = solver.run(
        metadata={
            "design_intent": "straight_waveguide_characterization",
            "target_wavelength": 1.55e-6,
        }
    )

    print(f"   Completed! {len(result.wavelengths)} wavelength points")

    # =========================================================================
    # 4. Compute metrics
    # =========================================================================

    print("\n4. Computing metrics...")

    s21 = result.s_parameters.get(('in', 'output'))
    if s21 is not None:
        il = insertion_loss(s21)
        eff = transmission_efficiency(s21)

        print(f"   Wavelength range: {result.wavelengths[0]*1e9:.1f} - {result.wavelengths[-1]*1e9:.1f} nm")
        print(f"   Avg insertion loss: {np.mean(il):.3f} dB")
        print(f"   Avg transmission: {np.mean(eff)*100:.1f}%")
        print(f"   Min transmission: {np.min(eff)*100:.1f}%")

    # =========================================================================
    # 5. Data Moat logging
    # =========================================================================

    print("\n5. Data Moat:")
    print(f"   Design ID: {result.design_id}")
    print(f"   Geometry hash: {result.geometry_hash[:16]}...")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


def run_demo_mode():
    """Demo mode when Meep is not available."""
    print("\n--- DEMO MODE ---")
    print("This shows what the output would look like:\n")

    # Create mock geometry
    from photonic_forge.core import Waveguide, SILICON, SILICON_DIOXIDE

    wg = Waveguide(start=(0, 0), end=(20e-6, 0), width=500e-9)
    bounds = (-2e-6, -1e-6, 22e-6, 1e-6)

    eps = wg.to_permittivity(
        bounds=bounds,
        resolution=50e-9,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE,
    )

    print(f"Geometry created: {eps.shape}")
    print(f"Silicon pixels: {np.sum(eps > 10)}")

    # Mock S-parameters
    wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
    s21 = 0.95 * np.exp(-((wavelengths - 1.55e-6) / 0.02e-6) ** 2) + 0j

    from photonic_forge.solvers.metrics import insertion_loss, transmission_efficiency

    il = insertion_loss(s21)
    eff = transmission_efficiency(s21)

    print(f"\nMock S-parameter results:")
    print(f"   Avg insertion loss: {np.mean(il):.3f} dB")
    print(f"   Avg transmission: {np.mean(eff)*100:.1f}%")

    print("\nTo run real simulations, install Meep in WSL/Linux.")


if __name__ == "__main__":
    main()

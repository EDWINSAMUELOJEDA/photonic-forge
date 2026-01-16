#!/usr/bin/env python3
"""
01_hello_waveguide.py - First PhotonicForge Demo

This example demonstrates:
1. Importing PhotonicForge SDF geometry primitives
2. Creating a straight waveguide using SDF representation
3. Evaluating signed distance at various points
4. Generating a permittivity grid for FDTD simulation

Run with:
    python examples/01_hello_waveguide.py
"""

import numpy as np
import photonic_forge
from photonic_forge.core import (
    Waveguide,
    Rectangle,
    Circle,
    union,
    SILICON,
    SILICON_DIOXIDE,
    SOI_WAVEGUIDE_WIDTH,
    SOI_WAVEGUIDE_HEIGHT,
    WAVELENGTH_C_BAND_CENTER,
)

print(f"PhotonicForge v{photonic_forge.__version__}")
print("=" * 60)

# =============================================================================
# Create a straight waveguide
# =============================================================================

# Standard SOI waveguide: 20 um long, 500 nm wide
waveguide = Waveguide(
    start=(0, 0),
    end=(20e-6, 0),  # 20 um length
    width=SOI_WAVEGUIDE_WIDTH,  # 500 nm
)

print("\n[Waveguide Created]")
print(f"   Start:      (0, 0)")
print(f"   End:        (20 um, 0)")
print(f"   Width:      {SOI_WAVEGUIDE_WIDTH * 1e9:.0f} nm")

# =============================================================================
# Evaluate SDF at sample points
# =============================================================================

print("\n[SDF Evaluation]")
print("   Point              | Distance (um)  | Location")
print("   " + "-" * 50)

test_points = [
    ("Center of WG", 10e-6, 0),
    ("Edge of WG", 10e-6, 250e-9),
    ("Just outside", 10e-6, 300e-9),
    ("Far outside", 10e-6, 1e-6),
]

for name, x, y in test_points:
    dist = waveguide.distance(np.array([x]), np.array([y]))[0]
    location = "INSIDE" if dist < 0 else ("BOUNDARY" if abs(dist) < 1e-9 else "OUTSIDE")
    print(f"   {name:17} | {dist*1e6:+13.4f} | {location}")

# =============================================================================
# Create a more complex structure: waveguide with taper
# =============================================================================

print("\n[Complex Structure: Waveguide + Coupling Region]")

# Main waveguide
main_wg = Waveguide(start=(0, 0), end=(15e-6, 0), width=500e-9)

# Wider coupling region
coupling = Rectangle(center=(17.5e-6, 0), width=5e-6, height=1e-6)

# Combine with union
device = union(main_wg, coupling)

# Test the combined structure
test_x = np.array([5e-6, 17.5e-6])
test_y = np.array([0, 0])
distances = device.distance(test_x, test_y)
print(f"   Main WG center (5 um):     {distances[0]*1e9:+.1f} nm")
print(f"   Coupling center (17.5 um): {distances[1]*1e9:+.1f} nm")

# =============================================================================
# Generate permittivity grid for FDTD
# =============================================================================

print("\n[Permittivity Grid]")

# Small region around waveguide
bounds = (-1e-6, -1e-6, 21e-6, 1e-6)  # (x_min, y_min, x_max, y_max)
resolution = 50e-9  # 50 nm grid

eps_grid = waveguide.to_permittivity(
    bounds=bounds,
    resolution=resolution,
    material_inside=SILICON,
    material_outside=SILICON_DIOXIDE,
)

print(f"   Grid shape: {eps_grid.shape}")
print(f"   Resolution: {resolution*1e9:.0f} nm")
print(f"   Silicon permittivity: {SILICON.epsilon_real:.3f}")
print(f"   SiO2 permittivity: {SILICON_DIOXIDE.epsilon_real:.3f}")

# Count pixels
n_silicon = np.sum(eps_grid > 10)  # Silicon has eps ~12
n_oxide = np.sum(eps_grid < 3)  # SiO2 has eps ~2
print(f"   Silicon pixels: {n_silicon}")
print(f"   Oxide pixels: {n_oxide}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("[OK] Week 1 SDF Geometry Demo Complete!")
print("   - SDF primitives: Rectangle, Circle, Waveguide, Bend90")
print("   - Boolean ops: union, intersection, subtraction")
print("   - Grid evaluation for FDTD simulation")
print("\n   Next: Week 2 - GDS export (open in KLayout)")

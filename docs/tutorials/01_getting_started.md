# Tutorial: Getting Started with PhotonicForge

This tutorial walks you through setting up PhotonicForge and creating your first waveguide design.

## Prerequisites

- Python 3.11+
- Basic knowledge of photonics concepts

## Installation

```bash
# Clone the repository
git clone https://github.com/edwinsamuelojeda/photonic-forge.git
cd photonic-forge

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install with development dependencies
pip install -e ".[dev,viz,gds]"
```

## Your First Waveguide

Let's create a simple silicon-on-insulator (SOI) waveguide.

### Step 1: Import the basics

```python
import numpy as np
from photonic_forge.core import (
    Waveguide,
    SILICON,
    SILICON_DIOXIDE,
    SOI_WAVEGUIDE_WIDTH,
)
```

### Step 2: Create geometry

```python
# Create a 20µm straight waveguide
wg = Waveguide(
    start=(0, 0),
    end=(20e-6, 0),
    width=SOI_WAVEGUIDE_WIDTH,  # 500nm
)

print(f"Waveguide width: {SOI_WAVEGUIDE_WIDTH * 1e9:.0f} nm")
```

### Step 3: Evaluate the SDF

The SDF (Signed Distance Field) tells us if points are inside or outside:

```python
# Check distance at various points
test_points = [
    (10e-6, 0),       # Center of waveguide
    (10e-6, 250e-9),  # Edge of waveguide
    (10e-6, 500e-9),  # Just outside
]

for x, y in test_points:
    dist = wg.distance(np.array([x]), np.array([y]))[0]
    location = "inside" if dist < 0 else "outside"
    print(f"Point ({x*1e6:.1f}µm, {y*1e9:.0f}nm): {dist*1e9:.1f}nm ({location})")
```

Output:
```
Point (10.0µm, 0nm): -250.0nm (inside)
Point (10.0µm, 250nm): 0.0nm (inside)
Point (10.0µm, 500nm): 250.0nm (outside)
```

### Step 4: Generate permittivity grid

Convert the SDF to a permittivity array for FDTD simulation:

```python
bounds = (-1e-6, -1e-6, 21e-6, 1e-6)  # Simulation domain
resolution = 50e-9  # 50nm grid

eps = wg.to_permittivity(
    bounds=bounds,
    resolution=resolution,
    material_inside=SILICON,       # ε ≈ 12.1
    material_outside=SILICON_DIOXIDE,  # ε ≈ 2.1
)

print(f"Grid shape: {eps.shape}")
print(f"Silicon pixels: {np.sum(eps > 10)}")
```

## Combining Shapes

Use boolean operations to build complex structures:

```python
from photonic_forge.core import Rectangle, Circle, union

# Main waveguide
wg = Waveguide(start=(0, 0), end=(15e-6, 0), width=500e-9)

# Add a coupling region
coupler = Rectangle(center=(17.5e-6, 0), width=5e-6, height=1e-6)

# Combine with union
device = union(wg, coupler)

# Or use operator syntax
device = wg | coupler
```

## Exporting to GDS

Export your design for fabrication (requires `gdstk`):

```python
try:
    import gdstk
    
    # Convert to GDS polygons
    polygons = device.to_gds(
        bounds=bounds,
        resolution=20e-9,
        layer=1,
    )
    
    # Create and save GDS file
    lib = gdstk.Library()
    cell = lib.new_cell("MY_DEVICE")
    for poly in polygons:
        cell.add(poly)
    lib.write_gds("my_device.gds")
    
except ImportError:
    print("Install gdstk: pip install gdstk")
```

## Next Steps

- [Tutorial 2: Inverse Design](02_inverse_design.md) - Optimize a coupler
- See `examples/01_hello_waveguide.py` for the complete script
- Read the [Core API Reference](../api/core.md) for all geometry options

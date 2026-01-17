# Core Module API

The `photonic_forge.core` module provides geometry primitives, materials, and physical constants.

## Geometry Primitives

All geometry classes inherit from `SDF2D` and implement the signed distance field interface.

### Rectangle

```python
from photonic_forge.core import Rectangle

rect = Rectangle(center=(0, 0), width=2.0, height=1.0)
```

**Attributes:**
- `center: tuple[float, float]` - Center position (x, y)
- `width: float` - Width in meters
- `height: float` - Height in meters

### Circle

```python
from photonic_forge.core import Circle

circle = Circle(center=(0, 0), radius=1.0)
```

**Attributes:**
- `center: tuple[float, float]` - Center position
- `radius: float` - Radius in meters

### RoundedRectangle

```python
from photonic_forge.core import RoundedRectangle

rr = RoundedRectangle(center=(0, 0), width=2.0, height=1.0, corner_radius=0.2)
```

**Attributes:**
- `corner_radius: float` - Must be ≤ min(width, height) / 2

### Waveguide

Straight waveguide segment with rounded endcaps.

```python
from photonic_forge.core import Waveguide

wg = Waveguide(start=(0, 0), end=(10e-6, 0), width=500e-9)
```

**Attributes:**
- `start: tuple[float, float]` - Start point
- `end: tuple[float, float]` - End point  
- `width: float` - Waveguide width

### Bend90

90-degree waveguide bend as an arc sector.

```python
from photonic_forge.core import Bend90

bend = Bend90(center=(0, 0), radius=5e-6, width=500e-9, start_angle=0)
```

**Attributes:**
- `center: tuple[float, float]` - Arc center
- `radius: float` - Bend radius (to centerline)
- `width: float` - Waveguide width
- `start_angle: float` - Starting angle in radians

### DirectionalCoupler

Two parallel waveguides for evanescent coupling.

```python
from photonic_forge.core import DirectionalCoupler

coupler = DirectionalCoupler(
    length=10e-6,
    gap=200e-9,
    width=500e-9,
    center=(0, 0),
)
```

## Boolean Operations

### Union

```python
combined = shape1 | shape2
# or
from photonic_forge.core import union
combined = union(shape1, shape2, shape3)
```

### Intersection

```python
overlap = shape1 & shape2
# or
from photonic_forge.core import intersection
overlap = intersection(shape1, shape2)
```

### Subtraction

```python
diff = shape1 - shape2
```

### SmoothUnion

Blends two shapes with smooth fillets.

```python
from photonic_forge.core.geometry import SmoothUnion
smooth = SmoothUnion(shape1, shape2, k=0.5)
```

## SDF Methods

All SDF objects support:

### distance(x, y)
Compute signed distance at points.

```python
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 0.0, 0.0])
distances = shape.distance(x, y)  # Negative = inside
```

### to_array(bounds, resolution)
Evaluate SDF on a grid.

```python
arr = shape.to_array(
    bounds=(x_min, y_min, x_max, y_max),
    resolution=50e-9
)
```

### to_permittivity(bounds, resolution, material_inside, material_outside)
Generate permittivity array for FDTD.

```python
eps = shape.to_permittivity(
    bounds=(-1e-6, -1e-6, 21e-6, 1e-6),
    resolution=50e-9,
    material_inside=SILICON,
    material_outside=SILICON_DIOXIDE,
)
```

## Materials

Pre-defined materials at 1550nm:

| Material | Variable | ε_r | n |
|----------|----------|-----|---|
| Silicon | `SILICON` | 12.1 | 3.48 |
| Silicon Dioxide | `SILICON_DIOXIDE` | 2.1 | 1.44 |
| Silicon Nitride | `SILICON_NITRIDE` | 4.0 | 2.0 |
| Air | `AIR` | 1.0 | 1.0 |

```python
from photonic_forge.core import SILICON, SILICON_DIOXIDE
```

## Constants

```python
from photonic_forge.core import (
    C,                      # Speed of light (m/s)
    WAVELENGTH_C_BAND_CENTER,  # 1550nm
    SOI_WAVEGUIDE_WIDTH,    # 500nm
    SOI_WAVEGUIDE_HEIGHT,   # 220nm
)
```

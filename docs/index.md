# PhotonicForge Documentation

Welcome to **PhotonicForge** — an open-source photonic integrated circuit (PIC) design platform with differentiable geometry and optimization capabilities.

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/edwinsamuelojeda/photonic-forge.git
cd photonic-forge
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e ".[dev,viz]"
```

### First Example

```python
from photonic_forge.core import Waveguide, SILICON, SILICON_DIOXIDE

# Create a 20µm waveguide
wg = Waveguide(start=(0, 0), end=(20e-6, 0), width=500e-9)

# Generate permittivity grid for simulation
eps = wg.to_permittivity(
    bounds=(-1e-6, -1e-6, 21e-6, 1e-6),
    resolution=50e-9,
    material_inside=SILICON,
    material_outside=SILICON_DIOXIDE,
)
```

## Core Concepts

### Signed Distance Fields (SDFs)
PhotonicForge uses SDFs to represent geometry. SDFs are functions that return:
- **Negative values** for points inside the shape
- **Zero** on the boundary
- **Positive values** for points outside

This representation enables smooth boolean operations and easy gradient computation.

### Geometry Primitives
- `Rectangle`, `Circle`, `RoundedRectangle` - Basic shapes
- `Waveguide` - Straight waveguide segment
- `Bend90` - 90-degree waveguide bend
- `DirectionalCoupler` - Two coupled waveguides

### Boolean Operations
```python
from photonic_forge.core import Circle, union, intersection

c1 = Circle(center=(0, 0), radius=1)
c2 = Circle(center=(1, 0), radius=1)

combined = c1 | c2        # Union (OR)
overlap = c1 & c2         # Intersection (AND)
diff = c1 - c2            # Subtraction
```

## Module Reference

| Module | Description |
|--------|-------------|
| [core](api/core.md) | Geometry primitives, materials, constants |
| [solvers](api/solvers.md) | FDTD simulation (Meep, metrics) |
| [optimize](api/optimize.md) | Optimization algorithms, objectives, constraints |
| [pdk](api/pdk.md) | Process Design Kit definitions |

## Tutorials

1. [Getting Started](tutorials/01_getting_started.md) - Build your first waveguide
2. [Inverse Design](tutorials/02_inverse_design.md) - Optimize a directional coupler

## Examples

Located in `examples/`:
- `01_hello_waveguide.py` - Basic geometry creation
- `02_optimize_coupler.py` - Optimization workflow
- `03_bend_optimization.py` - Bend loss minimization
- `06_constraints_demo.py` - Fabrication constraints

## Development

```bash
pytest tests/ -v          # Run tests
ruff check --fix src/     # Lint
black src/ tests/         # Format
```

## License

Apache 2.0 - See [LICENSE](../LICENSE)

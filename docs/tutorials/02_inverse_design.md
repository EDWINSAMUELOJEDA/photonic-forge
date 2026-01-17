# Tutorial: Inverse Design with PhotonicForge

Learn how to optimize photonic devices using PhotonicForge's optimization tools.

## Overview

Inverse design starts with a target performance and works backward to find the geometry that achieves it. This tutorial covers:

1. Defining a parameterized geometry
2. Setting optimization objectives
3. Running the optimizer
4. Analyzing and exporting results

## Example: Directional Coupler

We'll design a 50/50 directional coupler (3dB splitter) at 1550nm.

### Step 1: Define Parameters

```python
import numpy as np
from photonic_forge.optimize import create_coupler_parameterization

# Parameterize: length, gap, width
param = create_coupler_parameterization(
    length_range=(5e-6, 30e-6),   # 5-30 µm
    gap_range=(150e-9, 400e-9),   # 150-400 nm
    width_range=(450e-9, 550e-9), # 450-550 nm
)

print(f"Parameters: {param.param_names}")
print(f"Initial: {param.initial_values * 1e6}")  # in µm
```

### Step 2: Define Objective Function

For real optimization, you'd run FDTD simulation here. We'll use a simplified analytical model:

```python
def compute_coupling(params):
    """Simplified coupled-mode theory model."""
    length, gap, width = params
    
    # Coupling coefficient (simplified)
    kappa = 0.1e6 * np.exp(-gap * 1e9 / 100)
    
    # Power coupling ratio
    return np.sin(kappa * length) ** 2

def objective(params):
    """Minimize deviation from 50% coupling."""
    target = 0.5
    coupling = compute_coupling(params)
    return (coupling - target) ** 2

# Test initial design
print(f"Initial coupling: {compute_coupling(param.initial_values):.2%}")
```

### Step 3: Run Optimization

```python
from photonic_forge.optimize import scipy_minimize

result = scipy_minimize(
    objective_func=objective,
    x0=param.initial_values,
    bounds=param.get_bounds(),
    method="L-BFGS-B",
    max_iterations=50,
    verbose=True,
)

print(f"\nOptimization {'succeeded' if result.success else 'failed'}")
print(f"Iterations: {result.n_iterations}")
```

### Step 4: Analyze Results

```python
opt_length, opt_gap, opt_width = result.x
opt_coupling = compute_coupling(result.x)

print(f"\nOptimized parameters:")
print(f"  Length: {opt_length * 1e6:.2f} µm")
print(f"  Gap: {opt_gap * 1e9:.1f} nm")
print(f"  Width: {opt_width * 1e9:.1f} nm")
print(f"  Coupling: {opt_coupling:.2%}")
print(f"  Error: {abs(opt_coupling - 0.5):.4%}")
```

### Step 5: Build Optimized Geometry

```python
from photonic_forge.core.geometry import DirectionalCoupler, Waveguide, union

coupler = DirectionalCoupler(
    length=opt_length,
    gap=opt_gap,
    width=opt_width,
    center=(0, 0),
)

# Add input/output waveguides
y_offset = (opt_gap + opt_width) / 2
input_wg = Waveguide(
    start=(-opt_length/2 - 5e-6, y_offset),
    end=(-opt_length/2, y_offset),
    width=opt_width,
)

device = union(coupler, input_wg)
```

## Using Different Optimizers

### Pattern Search (Gradient-Free)

```python
from photonic_forge.optimize import pattern_search

result = pattern_search(
    objective_func=objective,
    x0=param.initial_values,
    bounds=param.get_bounds(),
    step_size=0.1,
    min_step=1e-4,
)
```

### Continuation (Topology Optimization)

For free-form pixel-based optimization:

```python
from photonic_forge.optimize import continuation_optimization

result = continuation_optimization(
    objective_func=objective,
    x0=x0,
    bounds=(lower, upper),
    beta_schedule=[1, 2, 4, 8, 16],  # Gradually binarize
    iterations_per_beta=20,
)
```

## Adding Constraints

### Fabrication Constraints

```python
from photonic_forge.optimize import (
    MinimumFeatureConstraint,
    project_binary,
    apply_symmetry,
)

# Enforce 100nm minimum features
constraint = MinimumFeatureConstraint(
    min_width=100e-9,
    min_gap=100e-9,
    resolution=20e-9,
)

# In objective function:
def constrained_objective(params):
    density = param.to_density(params)
    density = constraint.apply(density)
    density = apply_symmetry(density, y_symmetric=True)
    # ... run simulation ...
```

### Binarization

```python
# Soft projection during optimization
projected = project_binary(density, beta=8.0)

# Hard threshold for final design
from photonic_forge.optimize import binarize
final = binarize(density, threshold=0.5)
```

## Complete Example

See `examples/02_optimize_coupler.py` for a full working script.

## Next Steps

- Read about available [optimization algorithms](../api/optimize.md)
- Learn about [fabrication constraints](../api/optimize.md#constraints)
- Try `examples/03_bend_optimization.py` for bend loss minimization

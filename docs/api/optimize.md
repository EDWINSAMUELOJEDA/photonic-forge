# Optimization Module API

The `photonic_forge.optimize` module provides inverse design capabilities.

## Parameterization

### DesignRegion

Defines the optimization domain.

```python
from photonic_forge.optimize import DesignRegion

region = DesignRegion(
    bounds=(0, 0, 10e-6, 5e-6),  # (x_min, y_min, x_max, y_max)
    resolution=50e-9,            # Grid resolution
)

print(region.n_params)  # Number of optimizable pixels
```

### PixelParameterization

Free-form topology optimization with per-pixel density.

```python
from photonic_forge.optimize import PixelParameterization

param = PixelParameterization(
    region=region,
    eps_min=1.0,    # Background permittivity
    eps_max=12.1,   # Material permittivity
)

# Get initial (random or uniform) parameters
x0 = param.initial_values

# Convert parameters to SDF
sdf = param.to_sdf(x0)

# Convert parameters to permittivity
eps = param.to_permittivity(x0)
```

### ShapeParameterization

Higher-level parameterization of geometric features.

```python
from photonic_forge.optimize import ShapeParameterization

def my_geometry_factory(params):
    length, width = params
    return Waveguide(start=(0,0), end=(length, 0), width=width)

param = ShapeParameterization(
    param_names=["length", "width"],
    initial_values=np.array([10e-6, 500e-9]),
    bounds=(np.array([5e-6, 400e-9]), np.array([20e-6, 600e-9])),
    geometry_factory=my_geometry_factory,
)
```

### create_coupler_parameterization

Convenience factory for directional couplers.

```python
from photonic_forge.optimize import create_coupler_parameterization

param = create_coupler_parameterization(
    length_range=(5e-6, 30e-6),
    gap_range=(150e-9, 400e-9),
    width_range=(450e-9, 550e-9),
)
```

## Objective Functions

### ObjectiveFunction

Base class for optimization objectives.

```python
from photonic_forge.optimize import ObjectiveFunction

obj = ObjectiveFunction(
    name="insertion_loss",
    compute_fn=lambda result: insertion_loss(result.s_parameters[('in', 'out')]).mean(),
    direction="minimize",
    weight=1.0,
)

value = obj(simulation_result)
```

### CompositeObjective

Combine multiple objectives.

```python
from photonic_forge.optimize import CompositeObjective

composite = CompositeObjective()
composite.add(obj1, weight=1.0)
composite.add(obj2, weight=0.5)

total = composite(simulation_result)
breakdown = composite.breakdown(simulation_result)
```

### Preset Objectives

```python
from photonic_forge.optimize import (
    minimize_insertion_loss,
    maximize_transmission,
    maximize_bandwidth,
    target_transmission_curve,
    minimize_reflection,
)
```

## Algorithms

### run_optimization

General-purpose optimization interface.

```python
from photonic_forge.optimize import run_optimization, OptimizerConfig

config = OptimizerConfig(
    method="L-BFGS-B",
    max_iterations=100,
    tolerance=1e-6,
    verbose=True,
)

result = run_optimization(
    objective_func=my_objective,
    x0=initial_params,
    bounds=(lower, upper),
    config=config,
)

print(result.x)       # Optimal parameters
print(result.fun)     # Final objective value
print(result.success) # Convergence status
```

### scipy_minimize

Convenience wrapper for scipy optimizers.

```python
from photonic_forge.optimize import scipy_minimize

result = scipy_minimize(
    objective_func=objective,
    x0=x0,
    bounds=(lower, upper),
    method="L-BFGS-B",  # or "Powell", "Nelder-Mead", etc.
    max_iterations=50,
)
```

### pattern_search

Gradient-free coordinate search.

```python
from photonic_forge.optimize import pattern_search

result = pattern_search(
    objective_func=objective,
    x0=x0,
    bounds=(lower, upper),
    step_size=0.1,
    min_step=1e-4,
)
```

### continuation_optimization

Beta-scheduled optimization for topology problems.

```python
from photonic_forge.optimize import continuation_optimization

result = continuation_optimization(
    objective_func=objective,
    x0=x0,
    bounds=(lower, upper),
    beta_schedule=[1, 2, 4, 8, 16],
    iterations_per_beta=20,
)
```

## Constraints

### MinimumFeatureConstraint

Enforce minimum feature and gap sizes.

```python
from photonic_forge.optimize import MinimumFeatureConstraint

constraint = MinimumFeatureConstraint(
    min_width=100e-9,
    min_gap=100e-9,
    resolution=20e-9,
)

filtered = constraint.apply(density_field)
```

### project_binary

Smooth projection toward binary values.

```python
from photonic_forge.optimize import project_binary

projected = project_binary(density, beta=8.0, eta=0.5)
```

### apply_symmetry

Enforce design symmetry.

```python
from photonic_forge.optimize import apply_symmetry

symmetric = apply_symmetry(density, x_symmetric=True, y_symmetric=True)
```

### CurvatureConstraint

Penalize sharp corners.

```python
from photonic_forge.optimize import CurvatureConstraint

curv = CurvatureConstraint(
    max_curvature=1e6,  # 1/m
    resolution=20e-9,
    weight=1.0,
)

penalty = curv.penalty(density)
```

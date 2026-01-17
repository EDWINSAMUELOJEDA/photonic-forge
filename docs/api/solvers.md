# Solvers Module API

The `photonic_forge.solvers` module provides FDTD simulation interfaces and photonic metrics.

## MeepSolver

FDTD solver using the Meep backend (requires Linux/WSL).

```python
from photonic_forge.solvers import MeepSolver, SourceConfig, MonitorConfig

solver = MeepSolver(resolution=20e-9)

# Set up geometry from permittivity array
solver.setup_geometry(eps_array, bounds=(x_min, y_min, x_max, y_max))

# Add source
solver.add_source(SourceConfig(
    position=(0.5e-6, 0),
    wavelength_center=1.55e-6,
    wavelength_width=0.1e-6,
))

# Add monitor
solver.add_monitor(MonitorConfig(
    position=(19e-6, 0),
    size=(0, 1e-6),
    name="output",
))

# Run simulation
result = solver.run()
```

### SourceConfig

```python
@dataclass
class SourceConfig:
    position: tuple[float, float]  # (x, y) in meters
    wavelength_center: float = 1.55e-6
    wavelength_width: float = 0.1e-6
```

### MonitorConfig

```python
@dataclass  
class MonitorConfig:
    position: tuple[float, float]  # Monitor center
    size: tuple[float, float]      # Monitor dimensions
    name: str                      # Port identifier
```

### SimulationResult

Returned by `solver.run()`:

```python
@dataclass
class SimulationResult:
    s_parameters: dict[tuple[str, str], np.ndarray]  # Port pairs to S-params
    wavelengths: np.ndarray                          # Wavelength array (m)
    design_id: str | None                            # Unique design identifier
    geometry_hash: str | None                        # SHA-256 of geometry
    predicted_metrics: dict[str, float]              # Pre-computed metrics
    metadata: dict[str, Any]                         # User metadata
```

## Metrics

Functions to compute photonic figures of merit from S-parameters.

### insertion_loss(s21)

Insertion loss in dB: `IL = -20 * log10(|S21|)`

```python
from photonic_forge.solvers.metrics import insertion_loss

il = insertion_loss(result.s_parameters[('in', 'out')])
# Returns: array of IL values in dB (lower is better)
```

### return_loss(s11)

Return loss in dB: `RL = -20 * log10(|S11|)`

```python
from photonic_forge.solvers.metrics import return_loss

rl = return_loss(s11_data)
# Returns: array of RL values in dB (higher is better)
```

### transmission_efficiency(s21)

Power transmission: `η = |S21|²`

```python
from photonic_forge.solvers.metrics import transmission_efficiency

eta = transmission_efficiency(s21_data)
# Returns: array of values in [0, 1]
```

### bandwidth_3db(s21, wavelengths)

3dB bandwidth using linear interpolation.

```python
from photonic_forge.solvers.metrics import bandwidth_3db

bw = bandwidth_3db(s21_data, wavelengths)
# Returns: bandwidth in meters, or None
```

### crosstalk(s_coupled)

Crosstalk in dB: `CT = 20 * log10(|S_coupled|)`

```python
from photonic_forge.solvers.metrics import crosstalk

ct = crosstalk(s_coupled_data)
# Returns: array in dB (more negative is better)
```

### group_delay(s21, wavelengths)

Group delay from phase derivative.

```python
from photonic_forge.solvers.metrics import group_delay

tau = group_delay(s21_data, wavelengths)
# Returns: array of delays in seconds
```

## Data Moat

The solver automatically logs simulation results for training data collection:

- **Geometry hash**: SHA-256 fingerprint of the permittivity array
- **Design ID**: UUID for tracking design iterations
- **Metrics**: Pre-computed figures of merit
- **Metadata**: User-provided intent and context

Logs are written to `data/simulation_logs/simulations.jsonl`.

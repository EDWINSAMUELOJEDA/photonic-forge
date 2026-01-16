# PhotonicForge

**Open-source photonic integrated circuit (PIC) design platform**

[![CI Status](https://github.com/edwinsamuelojeda/photonic-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/edwinsamuelojeda/photonic-forge/actions)
[![codecov](https://codecov.io/gh/edwinsamuelojeda/photonic-forge/branch/main/graph/badge.svg)](https://codecov.io/gh/edwinsamuelojeda/photonic-forge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## What is PhotonicForge?

PhotonicForge provides a **differentiable geometry engine** for photonic layout design and a **GDS export pipeline**. Current features include:

- SDF-based 2D geometry primitives (waveguides, bends, couplers)
- Boolean operations for layout composition
- GDS export via marching-squares contour extraction
- Basic process design kit (PDK) definitions

---

## Quick Start

### Prerequisites
- Python 3.11 or newer
- `pip` and `venv`

### Installation
```bash
# Clone the repo
git clone https://github.com/edwinsamuelojeda/photonic-forge.git
cd photonic-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,viz]"

# Verify setup
pytest tests/ -v
```

### First Demo
```bash
python examples/01_hello_waveguide.py
```
This script:
1. Creates a simple straight silicon waveguide
2. Exports the layout to GDS (openable with KLayout)
3. Prints basic geometry information

---

## Architecture
```
PhotonicForge
├── core/               # SDF-based geometry engine
├── solvers/            # FDTD wrappers
├── pdk/                # Process design kit definitions
├── optimize/           # Optimization utilities
└── ui/                 # Web interface
```

---

## Development Workflow
```bash
# Run all tests
pytest tests/ -v

# Code formatting
black src/ tests/

# Linting
ruff check --fix src/ tests/

# Type checking
mypy src/

# Pre-commit checks
pre-commit run --all-files
```

### Optional Dependencies
```bash
# Install visualization and solver extras
pip install -e ".[dev,viz,solvers]"
```

---

## Project Status
- **Current:** GDS export and PDK are functional
- **Next:** Integrate FDTD solver (Meep) and expose photonic metrics

---

## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Push and open a PR

---

## License
PhotonicForge is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).

---

## Contact
- **GitHub:** [edwinsamuelojeda/photonic-forge](https://github.com/edwinsamuelojeda/photonic-forge)
- **Email:** team@photonic-forge.io

---

*Building the future of photonics, one design at a time.*

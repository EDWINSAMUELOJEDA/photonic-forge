# PhotonicForge

**Democratizing photonic chip design through AI-powered, differentiable geometry.**

[![CI Status](https://github.com/yourusername/photonic-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/photonic-forge/actions)
[![codecov](https://codecov.io/gh/yourusername/photonic-forge/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/photonic-forge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## What is PhotonicForge?

PhotonicForge is an **open-source, AI-powered photonic integrated circuit (PIC) design platform** that brings EDA to the masses. We're building a **"Netflix of photonics"**—affordable, accessible, and powered by differentiable geometry + adjoint optimization.

**Phase 1 (2026-2027):**
- SDF-based differentiable design geometry
- Fast local FDTD simulation with JAX autodiff
- Claude Sonnet as your AI design partner
- $20/month web interface (zero server cost via WebGPU)
- Target: 50K+ community users

**Phase 2 (2027-2029):**
- Integrated foundry with AI-driven MPW packing
- Closed-loop metrology data collection
- Digital twin of fab physics

**Phase 3 (2030+):**
- Master generative model for born-to-yield chips
- Photonic AI accelerators and quantum processors

---

## Quick Start

### Prerequisites
- Python 3.11 or 3.12
- `pip` and `venv`

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/photonic-forge.git
cd photonic-forge

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,viz]"

# Verify setup
pytest tests/ -v
```

### First Demo (Week 1-2)

```bash
python examples/01_hello_waveguide.py
```

This will:
1. Create a simple 20 µm straight silicon waveguide
2. Export to GDS (open in [KLayout](https://www.klayout.de/))
3. Print geometry info

---

## Architecture

```
PhotonicForge
├── core/               SDF-based differentiable geometry
├── solvers/            FDTD simulation (Meep wrapper initially, JAX later)
├── optimize/           Adjoint-method gradient-based optimization
├── neural_models/      ML surrogates and compilers (Week 6+)
├── agent/              Claude Sonnet design assistant (Week 7+)
├── pdk/                Process design kits (passive, then foundry-specific)
└── ui/                 Streamlit web interface + FastAPI backend
```

---

## Learning Path

### Week 0: Environment Setup ✅
- Repository structure + CI/CD pipeline
- `pytest`, `black`, `ruff`, `mypy` configured
- Pre-commit hooks ready

### Week 1: Geometry Core (SDF)
- Implement 2D SDF-based waveguides, bends, couplers
- Unit tests for sign convention and distance correctness
- Export to numpy arrays for downstream solvers

### Week 2: GDS Export Loop
- Export to GDS format
- Validate against golden files
- Integrate KLayout for visual verification

### Week 3-4: "Truth" Simulation (Meep FDTD)
- Wrap Meep Python interface
- Compute S-parameters, insertion loss, crosstalk
- Benchmark speed (~2 seconds per design on CPU)

### Week 5: Gradient-Free Optimization
- Random search + CMA-ES baseline
- Prove objective functions work end-to-end

### Week 6: Differentiable Pipeline (JAX)
- Autodiff through permittivity → metrics
- First true "layout as tensor" workflow
- Adjoint method gradient computation

### Week 7: LLM Agent Skeleton
- Rule-based design assistant (before LLM integration)
- Validate product interaction loop

### Week 8: WebGPU Compute Prototype
- Tiny GPU compute kernel for browser
- Performance benchmarking

### Week 9-10: MVP Hardening
- Polish, docs, hero demo (coupler or splitter)
- End-to-end workflow validated

---

## Free Software Stack

All tools are **open-source and free to use locally**:

| Tool | Purpose | License |
|------|---------|---------|
| [Meep](https://meep.readthedocs.io/) | FDTD simulation | GNU GPL |
| [gdsfactory](https://gdsfactory.github.io/) | PIC layout design | MIT |
| [KLayout](https://www.klayout.de/) | GDS viewer/editor | GPL/Commercial |
| [JAX](https://github.com/google/jax) | Autodiff + GPU compute | Apache 2.0 |
| [Streamlit](https://streamlit.io/) | Web UI | Apache 2.0 |
| [FastAPI](https://fastapi.tiangolo.com/) | Backend API | MIT |

---

## Learning Resources

### MIT OpenCourseWare
- [Photonic Materials and Devices (3.46)](https://ocw.mit.edu/courses/3-46-photonic-materials-and-devices-spring-2006/)
- [Fundamentals of Photonics: Quantum Electronics (6.974)](https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/)

### Tutorials & Docs
- [Meep documentation](https://meep.readthedocs.io/) – FDTD theory + Python usage
- [gdsfactory photonics training](https://gdsfactory.github.io/gdsfactory-photonics-training/) – PIC layout patterns
- [KLayout intro](https://www.klayout.de/intro.html) – GDS viewing

### Papers & Articles
- **SDFDiff** (CVPR 2020): Differentiable rendering via signed distance fields
- **Adjoint Method** (Fan et al., ACS Photonics 2018): Fast inverse design gradients
- **Inverse Design Lectures** (Flexcompute): [Lecture 2 (Adjoint Method)](https://www.flexcompute.com/tidy3d/learning-center/inverse-design/), [Lecture 3 (Optimization)](https://www.flexcompute.com/tidy3d/learning-center/inverse-design/Inverse-Design-in-Photonics-Lecture-3-Adjoint-Optimization/)

### Videos
- [FDTD Simulations with Meep](https://www.youtube.com/watch?v=ghdZDQf9QsM) (YouTube)
- [Inverse Design & Adjoint Optimization](https://www.youtube.com/watch?v=fKRHGoLuNjQ) (YouTube)
- [gdsfactory open-source photonics flow](https://www.youtube.com/watch?v=J91w2tyeeXc) (YouTube)

---

## Development Workflow

### Run Tests
```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -v

# With coverage
pytest tests/ --cov=photonic_forge --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check --fix src/ tests/

# Type check
mypy src/

# All at once (pre-commit)
pre-commit run --all-files
```

### Install Optional Dependencies
```bash
# Dev + visualization + solvers
pip install -e ".[dev,viz,solvers]"

# Everything (for full testing)
pip install -e ".[all]"
```

---

## Project Status

- **Current:** Week 0 setup (repo scaffolding, CI/CD)
- **Next:** Week 1 geometry core (SDF)
- **Roadmap:** [See ROADMAP.md](ROADMAP.md)

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

**Quick start:**
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run `pytest tests/` + `pre-commit run --all-files`
5. Push and open a PR

---

## License

PhotonicForge is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Citation

If you use PhotonicForge in your research, please cite:

```bibtex
@software{photonic_forge_2026,
  title = {PhotonicForge: Democratizing Photonic Chip Design},
  author = {{PhotonicForge Team}},
  year = {2026},
  url = {https://github.com/yourusername/photonic-forge},
}
```

---

## Contact

- **GitHub:** [photonic-forge](https://github.com/yourusername/photonic-forge)
- **Email:** [team@photonic-forge.io](mailto:team@photonic-forge.io)
- **Twitter:** [@photonic_forge](https://twitter.com/photonic_forge)

---

**Building the future of photonics, one design at a time.**

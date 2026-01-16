"""Optimize module for PhotonicForge.

Contains objective functions, parameterization, constraints, and
optimization algorithms for photonic inverse design.
"""

from photonic_forge.optimize.objective import (
    ObjectiveFunction,
    CompositeObjective,
    minimize_insertion_loss,
    maximize_transmission,
    maximize_bandwidth,
    target_transmission_curve,
    minimize_reflection,
)

from photonic_forge.optimize.parameterization import (
    DesignRegion,
    ParameterizedGeometry,
    DensityFieldSDF,
    PixelParameterization,
    ShapeParameterization,
    create_coupler_parameterization,
)

from photonic_forge.optimize.constraints import (
    MinimumFeatureConstraint,
    penalty_minimum_feature,
    project_binary,
    binarize,
    CurvatureConstraint,
    apply_symmetry,
)

from photonic_forge.optimize.algorithms import (
    OptimizerConfig,
    OptimizationResult,
    run_optimization,
    scipy_minimize,
    pattern_search,
    continuation_optimization,
    HAS_JAX,
    HAS_JAXOPT,
)


__all__ = [
    # Objective functions
    "ObjectiveFunction",
    "CompositeObjective",
    "minimize_insertion_loss",
    "maximize_transmission",
    "maximize_bandwidth",
    "target_transmission_curve",
    "minimize_reflection",
    # Parameterization
    "DesignRegion",
    "ParameterizedGeometry",
    "DensityFieldSDF",
    "PixelParameterization",
    "ShapeParameterization",
    "create_coupler_parameterization",
    # Constraints
    "MinimumFeatureConstraint",
    "penalty_minimum_feature",
    "project_binary",
    "binarize",
    "CurvatureConstraint",
    "apply_symmetry",
    # Algorithms
    "OptimizerConfig",
    "OptimizationResult",
    "run_optimization",
    "scipy_minimize",
    "pattern_search",
    "continuation_optimization",
    "HAS_JAX",
    "HAS_JAXOPT",
]

"""Objective function utilities for optimization.

Provides wrappers and presets for defining optimization objectives
from photonic device metrics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from photonic_forge.solvers.metrics import (
    bandwidth_3db,
    insertion_loss,
    transmission_efficiency,
)


@dataclass
class ObjectiveFunction:
    """Wrapper for an objective function with metadata.

    Attributes:
        func: Callable that takes simulation result and returns scalar.
        name: Human-readable name for logging.
        direction: 'minimize' or 'maximize'.
        weight: Weight when combining with other objectives.
    """
    func: Callable[[np.ndarray, np.ndarray], float]
    name: str = "objective"
    direction: Literal["minimize", "maximize"] = "minimize"
    weight: float = 1.0

    def __call__(self, s_params: np.ndarray, wavelengths: np.ndarray) -> float:
        """Evaluate the objective.

        Args:
            s_params: S-parameter data (e.g., S21 transmission).
            wavelengths: Wavelength array in meters.

        Returns:
            Scalar objective value.
        """
        value = self.func(s_params, wavelengths)
        # Flip sign for maximization (optimizer always minimizes)
        if self.direction == "maximize":
            return -value
        return value

    def raw_value(self, s_params: np.ndarray, wavelengths: np.ndarray) -> float:
        """Get the objective value without sign flip."""
        return self.func(s_params, wavelengths)


@dataclass
class CompositeObjective:
    """Combines multiple objectives with weights.

    The total objective is: sum(weight_i * objective_i)

    Attributes:
        objectives: List of ObjectiveFunction instances.
    """
    objectives: list[ObjectiveFunction] = field(default_factory=list)

    def add(self, objective: ObjectiveFunction) -> "CompositeObjective":
        """Add an objective to the composite."""
        self.objectives.append(objective)
        return self

    def __call__(self, s_params: np.ndarray, wavelengths: np.ndarray) -> float:
        """Evaluate all objectives and return weighted sum."""
        if not self.objectives:
            raise ValueError("No objectives defined")

        total = 0.0
        for obj in self.objectives:
            total += obj.weight * obj(s_params, wavelengths)
        return total

    def breakdown(
        self, s_params: np.ndarray, wavelengths: np.ndarray
    ) -> dict[str, float]:
        """Get individual objective values (raw, without sign flip)."""
        return {
            obj.name: obj.raw_value(s_params, wavelengths)
            for obj in self.objectives
        }


# =============================================================================
# Preset Objectives
# =============================================================================


def minimize_insertion_loss(
    wavelength_range: tuple[float, float] | None = None,
    weight: float = 1.0,
) -> ObjectiveFunction:
    """Create objective to minimize insertion loss.

    Args:
        wavelength_range: Optional (min, max) wavelength to average over.
                          If None, uses all wavelengths.
        weight: Weight for multi-objective optimization.

    Returns:
        ObjectiveFunction that computes mean insertion loss in dB.
    """
    def func(s21: np.ndarray, wavelengths: np.ndarray) -> float:
        if wavelength_range is not None:
            mask = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])
            s21 = s21[mask]
        il = insertion_loss(s21)
        return float(np.mean(il))

    return ObjectiveFunction(
        func=func,
        name="insertion_loss",
        direction="minimize",
        weight=weight,
    )


def maximize_transmission(
    target_wavelength: float = 1.55e-6,
    bandwidth: float = 0.01e-6,
    weight: float = 1.0,
) -> ObjectiveFunction:
    """Create objective to maximize transmission at target wavelength.

    Args:
        target_wavelength: Center wavelength in meters.
        bandwidth: Wavelength range around center to average.
        weight: Weight for multi-objective optimization.

    Returns:
        ObjectiveFunction that maximizes transmission efficiency.
    """
    def func(s21: np.ndarray, wavelengths: np.ndarray) -> float:
        mask = np.abs(wavelengths - target_wavelength) <= bandwidth / 2
        if not np.any(mask):
            # Find closest wavelength
            idx = np.argmin(np.abs(wavelengths - target_wavelength))
            return float(transmission_efficiency(s21[idx:idx+1])[0])
        eta = transmission_efficiency(s21[mask])
        return float(np.mean(eta))

    return ObjectiveFunction(
        func=func,
        name="transmission",
        direction="maximize",
        weight=weight,
    )


def maximize_bandwidth(
    weight: float = 1.0,
) -> ObjectiveFunction:
    """Create objective to maximize 3dB bandwidth.

    Args:
        weight: Weight for multi-objective optimization.

    Returns:
        ObjectiveFunction that maximizes bandwidth.
    """
    def func(s21: np.ndarray, wavelengths: np.ndarray) -> float:
        bw = bandwidth_3db(s21, wavelengths)
        if bw is None:
            return 0.0
        # Convert to nm for nicer numbers
        return float(bw * 1e9)

    return ObjectiveFunction(
        func=func,
        name="bandwidth_3db",
        direction="maximize",
        weight=weight,
    )


def target_transmission_curve(
    target_s21: np.ndarray,
    target_wavelengths: np.ndarray,
    weight: float = 1.0,
) -> ObjectiveFunction:
    """Create objective to match a target transmission curve.

    Uses mean squared error between |S21|² curves.

    Args:
        target_s21: Target S21 values (complex).
        target_wavelengths: Wavelengths for target curve.
        weight: Weight for multi-objective optimization.

    Returns:
        ObjectiveFunction that minimizes MSE to target.
    """
    target_power = np.abs(target_s21) ** 2

    def func(s21: np.ndarray, wavelengths: np.ndarray) -> float:
        # Interpolate to match wavelength grids
        actual_power = np.abs(s21) ** 2
        interp_power = np.interp(target_wavelengths, wavelengths, actual_power)
        mse = np.mean((interp_power - target_power) ** 2)
        return float(mse)

    return ObjectiveFunction(
        func=func,
        name="target_curve_mse",
        direction="minimize",
        weight=weight,
    )


def minimize_reflection(
    wavelength_range: tuple[float, float] | None = None,
    weight: float = 1.0,
) -> ObjectiveFunction:
    """Create objective to minimize reflection (maximize return loss).

    Args:
        wavelength_range: Optional (min, max) wavelength to average over.
        weight: Weight for multi-objective optimization.

    Returns:
        ObjectiveFunction that minimizes |S11|².
    """
    def func(s11: np.ndarray, wavelengths: np.ndarray) -> float:
        if wavelength_range is not None:
            mask = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])
            s11 = s11[mask]
        reflection_power = np.abs(s11) ** 2
        return float(np.mean(reflection_power))

    return ObjectiveFunction(
        func=func,
        name="reflection",
        direction="minimize",
        weight=weight,
    )


__all__ = [
    "ObjectiveFunction",
    "CompositeObjective",
    "minimize_insertion_loss",
    "maximize_transmission",
    "maximize_bandwidth",
    "target_transmission_curve",
    "minimize_reflection",
]

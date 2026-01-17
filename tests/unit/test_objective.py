"""Unit tests for objective function module."""

import numpy as np
import pytest

from photonic_forge.optimize.objective import (
    CompositeObjective,
    ObjectiveFunction,
    maximize_bandwidth,
    maximize_transmission,
    minimize_insertion_loss,
    minimize_reflection,
    target_transmission_curve,
)


class TestObjectiveFunction:
    """Tests for ObjectiveFunction wrapper."""

    def test_minimize_direction(self):
        """Minimize objective returns value unchanged."""
        def simple_func(s, w):
            return 1.5

        obj = ObjectiveFunction(func=simple_func, direction="minimize")
        result = obj(np.array([1.0]), np.array([1.55e-6]))

        assert result == 1.5

    def test_maximize_direction(self):
        """Maximize objective returns negated value."""
        def simple_func(s, w):
            return 1.5

        obj = ObjectiveFunction(func=simple_func, direction="maximize")
        result = obj(np.array([1.0]), np.array([1.55e-6]))

        assert result == -1.5

    def test_raw_value_ignores_direction(self):
        """raw_value returns original regardless of direction."""
        def simple_func(s, w):
            return 2.0

        obj = ObjectiveFunction(func=simple_func, direction="maximize")
        result = obj.raw_value(np.array([1.0]), np.array([1.55e-6]))

        assert result == 2.0

    def test_weight_attribute(self):
        """Weight is stored correctly."""
        obj = ObjectiveFunction(func=lambda s, w: 1, weight=0.5)
        assert obj.weight == 0.5


class TestCompositeObjective:
    """Tests for CompositeObjective."""

    def test_empty_raises(self):
        """Empty composite raises ValueError."""
        composite = CompositeObjective()

        with pytest.raises(ValueError, match="No objectives"):
            composite(np.array([1.0]), np.array([1.55e-6]))

    def test_single_objective(self):
        """Single objective returns its value."""
        obj = ObjectiveFunction(func=lambda s, w: 3.0, weight=1.0)
        composite = CompositeObjective(objectives=[obj])

        result = composite(np.array([1.0]), np.array([1.55e-6]))
        assert result == 3.0

    def test_weighted_sum(self):
        """Multiple objectives combine with weights."""
        obj1 = ObjectiveFunction(func=lambda s, w: 1.0, weight=2.0)
        obj2 = ObjectiveFunction(func=lambda s, w: 3.0, weight=0.5)
        composite = CompositeObjective(objectives=[obj1, obj2])

        # 2.0 * 1.0 + 0.5 * 3.0 = 2.0 + 1.5 = 3.5
        result = composite(np.array([1.0]), np.array([1.55e-6]))
        assert result == pytest.approx(3.5)

    def test_add_fluent(self):
        """add() method is fluent."""
        obj = ObjectiveFunction(func=lambda s, w: 1.0)
        composite = CompositeObjective()

        result = composite.add(obj)
        assert result is composite
        assert len(composite.objectives) == 1

    def test_breakdown_returns_raw_values(self):
        """breakdown returns dict of raw values."""
        obj1 = ObjectiveFunction(func=lambda s, w: 1.0, name="a", direction="minimize")
        obj2 = ObjectiveFunction(func=lambda s, w: 2.0, name="b", direction="maximize")
        composite = CompositeObjective(objectives=[obj1, obj2])

        breakdown = composite.breakdown(np.array([1.0]), np.array([1.55e-6]))

        assert breakdown["a"] == 1.0
        assert breakdown["b"] == 2.0  # Raw value, not negated


class TestPresetObjectives:
    """Tests for preset objective functions."""

    def test_minimize_insertion_loss_perfect(self):
        """Perfect transmission has ~0 dB insertion loss."""
        obj = minimize_insertion_loss()
        s21 = np.array([1.0 + 0j])
        wavelengths = np.array([1.55e-6])

        value = obj.raw_value(s21, wavelengths)
        assert value == pytest.approx(0.0, abs=0.01)

    def test_minimize_insertion_loss_half_power(self):
        """Half amplitude gives ~6 dB loss."""
        obj = minimize_insertion_loss()
        s21 = np.array([0.5 + 0j])
        wavelengths = np.array([1.55e-6])

        value = obj.raw_value(s21, wavelengths)
        assert value == pytest.approx(6.02, abs=0.1)

    def test_maximize_transmission_perfect(self):
        """Perfect transmission gives efficiency 1."""
        obj = maximize_transmission(target_wavelength=1.55e-6)
        s21 = np.array([1.0 + 0j])
        wavelengths = np.array([1.55e-6])

        value = obj.raw_value(s21, wavelengths)
        assert value == pytest.approx(1.0, abs=0.01)

    def test_maximize_transmission_direction(self):
        """maximize_transmission has direction='maximize'."""
        obj = maximize_transmission()
        assert obj.direction == "maximize"

    def test_maximize_bandwidth_flat(self):
        """Flat response gives full bandwidth."""
        obj = maximize_bandwidth()
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 100)
        s21 = np.ones_like(wavelengths) + 0j

        value = obj.raw_value(s21, wavelengths)
        # Bandwidth in nm: 100 nm
        assert value == pytest.approx(100, rel=0.1)

    def test_target_transmission_curve_perfect_match(self):
        """Matching curve gives zero MSE."""
        target = np.array([0.8, 0.9, 1.0, 0.9, 0.8]) + 0j
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 5)

        obj = target_transmission_curve(target, wavelengths)

        # Same curve should give 0
        value = obj.raw_value(target, wavelengths)
        assert value == pytest.approx(0.0, abs=1e-10)

    def test_minimize_reflection_no_reflection(self):
        """Zero reflection gives near-zero penalty."""
        obj = minimize_reflection()
        s11 = np.array([1e-6 + 0j])
        wavelengths = np.array([1.55e-6])

        value = obj.raw_value(s11, wavelengths)
        assert value < 1e-10


class TestObjectiveWavelengthRange:
    """Tests for wavelength range filtering."""

    def test_insertion_loss_with_range(self):
        """Wavelength range filters the data."""
        obj = minimize_insertion_loss(wavelength_range=(1.54e-6, 1.56e-6))

        wavelengths = np.array([1.50e-6, 1.55e-6, 1.60e-6])
        # Only middle wavelength should be used
        s21 = np.array([0.1 + 0j, 1.0 + 0j, 0.1 + 0j])

        value = obj.raw_value(s21, wavelengths)
        # Should be ~0 dB (only the 1.0 value is in range)
        assert value == pytest.approx(0.0, abs=0.1)

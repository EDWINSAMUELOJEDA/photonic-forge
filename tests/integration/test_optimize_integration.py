"""Integration tests for optimization module."""

import pytest
import numpy as np

from photonic_forge.optimize import (
    ObjectiveFunction,
    run_optimization,
    scipy_minimize,
    pattern_search,
    OptimizerConfig,
    OptimizationResult,
)


class TestRunOptimization:
    """Integration tests for run_optimization."""
    
    def test_minimize_quadratic(self):
        """Optimizer finds minimum of quadratic."""
        def quadratic(x):
            return (x[0] - 2) ** 2 + (x[1] + 1) ** 2
        
        result = run_optimization(
            objective_func=quadratic,
            x0=np.array([0.0, 0.0]),
            config=OptimizerConfig(max_iterations=50, verbose=False),
        )
        
        assert result.success
        assert result.x[0] == pytest.approx(2.0, abs=0.01)
        assert result.x[1] == pytest.approx(-1.0, abs=0.01)
        assert result.fun == pytest.approx(0.0, abs=1e-4)
    
    def test_bounded_optimization(self):
        """Optimizer respects bounds."""
        def objective(x):
            return -x[0]  # Minimize -x means maximize x
        
        result = run_optimization(
            objective_func=objective,
            x0=np.array([0.5]),
            bounds=(np.array([0.0]), np.array([1.0])),
            config=OptimizerConfig(max_iterations=50, verbose=False),
        )
        
        # Should hit upper bound
        assert result.x[0] == pytest.approx(1.0, abs=0.01)
    
    def test_history_tracked(self):
        """History contains optimization trajectory."""
        def objective(x):
            return x[0] ** 2
        
        result = run_optimization(
            objective_func=objective,
            x0=np.array([5.0]),
            config=OptimizerConfig(max_iterations=20, verbose=False),
        )
        
        assert len(result.history) > 0
        # First should be worse than last
        assert result.history[0][1] > result.history[-1][1]
    
    def test_result_structure(self):
        """OptimizationResult has expected attributes."""
        def objective(x):
            return x[0] ** 2
        
        result = run_optimization(
            objective_func=objective,
            x0=np.array([1.0]),
            config=OptimizerConfig(max_iterations=5, verbose=False),
        )
        
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        assert hasattr(result, 'n_iterations')


class TestScipyMinimize:
    """Tests for scipy_minimize wrapper."""
    
    def test_rosenbrock(self):
        """Optimize Rosenbrock function."""
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        
        result = scipy_minimize(
            objective_func=rosenbrock,
            x0=np.array([0.0, 0.0]),
            max_iterations=100,
            verbose=False,
        )
        
        # Minimum is at (1, 1)
        assert result.x[0] == pytest.approx(1.0, abs=0.1)
        assert result.x[1] == pytest.approx(1.0, abs=0.1)
    
    def test_different_methods(self):
        """Test with different scipy methods."""
        def simple(x):
            return x[0] ** 2 + x[1] ** 2
        
        for method in ["L-BFGS-B", "Powell", "Nelder-Mead"]:
            result = scipy_minimize(
                objective_func=simple,
                x0=np.array([1.0, 1.0]),
                method=method,
                max_iterations=50,
                verbose=False,
            )
            
            assert result.fun < 0.01, f"Method {method} failed"


class TestPatternSearch:
    """Tests for pattern_search optimizer."""
    
    def test_simple_minimum(self):
        """Pattern search finds simple minimum."""
        def objective(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        
        result = pattern_search(
            objective_func=objective,
            x0=np.array([0.0, 0.0]),
            max_iterations=500,
            verbose=False,
        )
        
        assert result.x[0] == pytest.approx(1.0, abs=0.1)
        assert result.x[1] == pytest.approx(2.0, abs=0.1)
    
    def test_respects_bounds(self):
        """Pattern search respects bounds."""
        def objective(x):
            return -x[0]  # Want to maximize x
        
        result = pattern_search(
            objective_func=objective,
            x0=np.array([0.5]),
            bounds=(np.array([0.0]), np.array([0.8])),
            max_iterations=100,
            verbose=False,
        )
        
        # Should be at or near upper bound
        assert result.x[0] >= 0.7


class TestOptimizationWithMockedSimulation:
    """Test full optimization loop with mocked photonic simulation."""
    
    def test_coupler_optimization_mock(self):
        """Optimize coupler parameters with mock simulation."""
        # Mock simulation: coupling depends on gap and length
        def mock_simulation(params):
            length, gap, width = params
            
            # Simplified model: closer gap = more coupling
            # Longer length = more coupling (up to a point)
            coupling = 0.5 * np.exp(-gap / 200e-9) * np.sin(length / 5e-6) ** 2
            return coupling
        
        target_coupling = 0.3
        
        def objective(params):
            coupling = mock_simulation(params)
            return (coupling - target_coupling) ** 2
        
        # Initial guess
        x0 = np.array([10e-6, 300e-9, 500e-9])  # length, gap, width
        
        # Bounds
        lower = np.array([5e-6, 150e-9, 400e-9])
        upper = np.array([30e-6, 500e-9, 600e-9])
        
        result = scipy_minimize(
            objective_func=objective,
            x0=x0,
            bounds=(lower, upper),
            max_iterations=50,
            verbose=False,
        )
        
        # Check that optimization found a solution (may already be at optimum)
        initial_error = objective(x0)
        final_error = result.fun
        
        assert final_error <= initial_error
        
        # Check final coupling is closer to target
        final_coupling = mock_simulation(result.x)
        assert abs(final_coupling - target_coupling) < 0.1

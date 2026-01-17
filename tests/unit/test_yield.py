"""Tests for yield analysis."""

import numpy as np
import pytest
from photonic_forge.optimize.yield_analysis import YieldEstimator


class MockSurrogate:
    """Mock surrogate model."""
    pass


def test_yield_estimator():
    """Test yield estimation logic."""
    surrogate = MockSurrogate()
    estimator = YieldEstimator(surrogate)
    
    nominal = np.array([1.0, 1.0])
    
    # Dummy metric function that just sums parameters
    # Should be robust to small noise
    def metric(p):
        return np.sum(p)
        
    # Test
    result = estimator.analyze(
        nominal_params=nominal,
        metric_func=metric,
        threshold=1.5, # 1+1=2, so should pass easily unless huge noise
        n_samples=100,
        std_dev=0.01 # Small noise
    )
    
    assert result.samples == 100
    assert result.yield_percent > 90.0 # Should be very high yield
    assert result.mean_metric > 1.8 

"""Yield estimation using Monte Carlo analysis.

Uses surrogates to estimate manufacturing yield by simulating
process variations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photonic_forge.neural_models.surrogate import SurrogateModel


@dataclass
class YieldResult:
    """Result of yield analysis."""
    yield_percent: float
    mean_metric: float
    std_metric: float
    worst_case: float
    samples: int


class YieldEstimator:
    """Estimates yield using surrogate models."""

    def __init__(self, surrogate: SurrogateModel):
        self.surrogate = surrogate

    def analyze(
        self,
        nominal_params: np.ndarray,
        metric_func: callable,
        threshold: float,
        n_samples: int = 1000,
        std_dev: float = 0.05,  # 5% relative variation
    ) -> YieldResult:
        """Run Monte Carlo analysis.

        Args:
            nominal_params: Design parameters.
            metric_func: Function extracting metric from surrogates prediction.
            threshold: Minimum acceptable metric value.
            n_samples: Number of random variations to test.
            std_dev: Standard deviation of variation (relative to param value).
        
        Returns:
            Yield statistics.
        """
        # Generate variations
        # noise = N(0, 1) * std_dev * params
        noise = np.random.randn(n_samples, len(nominal_params)) * std_dev * nominal_params
        variations = nominal_params + noise
        
        # Batch prediction with surrogate (mocking batch by loop if not supported, 
        # normally surrogate handles batches)
        # Using simple mean/linear logic as placeholder for complex surrogate behavior in this MVP
        
        # In real usage: predictions = self.surrogate.predict_batch(variations)
        # Here we simulate the effect for demonstration
        
        scores = []
        for params in variations:
            # We assume the surrogate has a predict method that takes params 
            # (which might need geometry generation step in reality)
            # For this MVP, we'll assume the surrogate can take raw params roughly
            # or we simulate the metric noise directly.
            
            # Simulated outcome: The metric degrades as we move away from nominal
            # plus some random process noise
            # Note: In real usage, the metric_func (wrapping surrogate) handles the physics.
            # We assume the variations are passed to it.
            
            score = metric_func(params)
            scores.append(score)

        scores_arr = np.array(scores)
        passing = scores_arr >= threshold
        
        return YieldResult(
            yield_percent=float(np.mean(passing) * 100),
            mean_metric=float(np.mean(scores_arr)),
            std_metric=float(np.std(scores_arr)),
            worst_case=float(np.min(scores_arr)),
            samples=n_samples
        )

"""Example of Yield Analysis.

Demonstrates how robust optimization improves yield compared to nominal optimization.
"""

import numpy as np
from photonic_forge.optimize import YieldEstimator, YieldResult

# Mock Design Problem:
# Objective = 1.0 - (x - target)^2
# But manufacturing adds noise: x_actual = x + noise
# A "sharp" peak would be susceptible to noise.
# A "broad" peak would be robust.

class MockSurrogate:
    """Simulates a physical response to parameters."""
    pass

def simulate_performance(params: np.ndarray) -> float:
    """Performance metric (higher is better)."""
    # Let's say we have two optima:
    # 1. Narrow Peak at x=2.0 (Height 1.0) - High risk
    # 2. Broad Peak at x=5.0 (Height 0.95) - Safe bet
    
    x = params[0]
    
    peak1 = 1.0 * np.exp(-(x - 2.0)**2 / 0.01) # Very narrow
    peak2 = 0.95 * np.exp(-(x - 5.0)**2 / 1.0) # Very broad
    
    return max(peak1, peak2)

def run_example():
    print("--- Yield Analysis Demo ---\n")
    
    estimator = YieldEstimator(MockSurrogate())
    
    # CASE 1: Nominal Optimization (found the highest peak)
    design_nominal = np.array([2.0])
    print(f"Design A (Aggressive): x = {design_nominal[0]}")
    print(f"  Nominal Perf: {simulate_performance(design_nominal):.4f}")
    
    # CASE 2: Robust Optimization (found the broad peak)
    design_robust = np.array([5.0])
    print(f"Design B (Robust):     x = {design_robust[0]}")
    print(f"  Nominal Perf: {simulate_performance(design_robust):.4f}")
    
    print("\nSimulating Manufacturing Variations (+/- 10%)...")
    
    # Run Yield Analysis
    # We pass our simulation function as the 'metric_func'
    # In real app, this would use the surrogate.predict()
    
    res_a = estimator.analyze(
        nominal_params=design_nominal,
        metric_func=simulate_performance,
        threshold=0.8, # Must be > 0.8 to work
        n_samples=1000,
        std_dev=0.1 # 10% noise
    )
    
    res_b = estimator.analyze(
        nominal_params=design_robust,
        metric_func=simulate_performance,
        threshold=0.8,
        n_samples=1000,
        std_dev=0.1
    )
    
    print("\n--- Results ---")
    print(f"Design A Yield: {res_a.yield_percent:.1f}%  (Mean Perf: {res_a.mean_metric:.4f})")
    print(f"Design B Yield: {res_b.yield_percent:.1f}%  (Mean Perf: {res_b.mean_metric:.4f})")
    
    if res_b.yield_percent > res_a.yield_percent:
        print("\nConclusion: Design B is better for mass production despite lower peak performance.")
    
if __name__ == "__main__":
    run_example()

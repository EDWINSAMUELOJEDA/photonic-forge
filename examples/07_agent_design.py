#!/usr/bin/env python3
"""
07_agent_design.py - Automated design exploration with agent

This example demonstrates:
1. Setting up a design agent with bounds
2. Running automated exploration
3. Analyzing discovered designs
4. Using different exploration strategies

Run with:
    python examples/07_agent_design.py
"""

import numpy as np

from photonic_forge.agent import (
    DesignAgent,
    AgentConfig,
    run_agent_exploration,
    LatinHypercubeStrategy,
)
from photonic_forge.core import SILICON, SILICON_DIOXIDE
from photonic_forge.core.geometry import DirectionalCoupler


def main():
    """Run agent-based design exploration."""
    print("=" * 60)
    print("PhotonicForge: Agent-Based Design Exploration")
    print("=" * 60)

    # =========================================================================
    # 1. Define the design problem
    # =========================================================================

    print("\n1. Problem Setup")
    print("   Goal: Find optimal directional coupler parameters")

    # Parameters: [length, gap, width]
    lower = np.array([5e-6, 100e-9, 400e-9])   # Min values
    upper = np.array([30e-6, 400e-9, 600e-9])  # Max values

    param_names = ["length", "gap", "width"]

    print(f"   Parameters: {param_names}")
    print(f"   Bounds: length=[5-30]um, gap=[100-400]nm, width=[400-600]nm")

    # =========================================================================
    # 2. Define objective function
    # =========================================================================

    print("\n2. Defining objective function")

    target_coupling = 0.5  # 50% coupling (3dB splitter)

    def coupling_model(params: np.ndarray) -> float:
        """Simplified coupled-mode theory model."""
        length, gap, width = params

        # Coupling coefficient (simplified)
        kappa = 0.1e6 * np.exp(-gap * 1e9 / 100)

        # Power coupling ratio
        return np.sin(kappa * length) ** 2

    def objective(params: np.ndarray) -> float:
        """Maximize closeness to target coupling."""
        coupling = coupling_model(params)
        # Negative distance to target (maximize = minimize distance)
        return -abs(coupling - target_coupling)

    print(f"   Target coupling: {target_coupling:.0%}")

    # =========================================================================
    # 3. Configure and run agent
    # =========================================================================

    print("\n3. Running design exploration...")

    config = AgentConfig(
        population_size=30,
        max_generations=20,
        elite_fraction=0.15,
        mutation_rate=0.2,
        exploration_factor=0.25,
    )

    best, candidates = run_agent_exploration(
        bounds=(lower, upper),
        objective_func=objective,
        config=config,
        verbose=True,
    )

    # =========================================================================
    # 4. Analyze results
    # =========================================================================

    print("\n4. Results")
    print(f"   Total candidates explored: {len(candidates)}")

    opt_length, opt_gap, opt_width = best.parameters
    opt_coupling = coupling_model(best.parameters)

    print(f"\n   Best design found:")
    print(f"      Length: {opt_length * 1e6:.2f} um")
    print(f"      Gap: {opt_gap * 1e9:.1f} nm")
    print(f"      Width: {opt_width * 1e9:.1f} nm")
    print(f"      Coupling: {opt_coupling:.3f} ({opt_coupling*100:.1f}%)")
    print(f"      Error from target: {abs(opt_coupling - target_coupling)*100:.2f}%")

    # =========================================================================
    # 5. Show exploration statistics
    # =========================================================================

    print("\n5. Exploration Statistics")

    generations = [c.generation for c in candidates]
    scores = [c.score for c in candidates]

    # Group by generation
    gen_stats = {}
    for c in candidates:
        gen = c.generation
        if gen not in gen_stats:
            gen_stats[gen] = []
        gen_stats[gen].append(c.score)

    print("\n   Generation | Best Score | Avg Score")
    print("   " + "-" * 35)
    for gen in sorted(gen_stats.keys())[:10]:
        best_score = max(gen_stats[gen])
        avg_score = np.mean(gen_stats[gen])
        print(f"   {gen:10d} | {best_score:10.6f} | {avg_score:.6f}")

    # =========================================================================
    # 6. Build optimized geometry
    # =========================================================================

    print("\n6. Creating optimized geometry...")

    coupler = DirectionalCoupler(
        length=opt_length,
        gap=opt_gap,
        width=opt_width,
        center=(0, 0),
    )

    bounds_viz = (
        -opt_length/2 - 2e-6,
        -1e-6,
        opt_length/2 + 2e-6,
        1e-6,
    )

    eps = coupler.to_permittivity(
        bounds=bounds_viz,
        resolution=50e-9,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE,
    )

    print(f"   Grid shape: {eps.shape}")

    # =========================================================================
    # 7. Try Latin Hypercube strategy
    # =========================================================================

    print("\n7. Bonus: Latin Hypercube Space-Filling Design")

    lhs = LatinHypercubeStrategy(bounds=(lower, upper), n_samples=20)

    lhs_designs = []
    for i in range(20):
        result = lhs.suggest(np.array([]).reshape(0, 3), np.array([]))
        coupling = coupling_model(result.next_point)
        lhs_designs.append((result.next_point, coupling))

    # Find best from LHS
    lhs_best = max(lhs_designs, key=lambda x: -abs(x[1] - target_coupling))
    lhs_coupling = lhs_best[1]

    print(f"   LHS best coupling: {lhs_coupling:.3f}")
    print(f"   LHS error from target: {abs(lhs_coupling - target_coupling)*100:.2f}%")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Agent Design Exploration Complete!")
    print(f"   Optimal coupler: L={opt_length*1e6:.1f}um, gap={opt_gap*1e9:.0f}nm")
    print(f"   Coupling ratio: {opt_coupling:.1%}")
    print("=" * 60)

    return best, candidates


if __name__ == "__main__":
    main()

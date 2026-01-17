"""Optimization algorithms for photonic inverse design.

Provides gradient-free and optional gradient-based optimization
using scipy and optionally JAX.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize as scipy_opt

# Check for optional JAX support
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False


@dataclass
class OptimizerConfig:
    """Configuration for optimization.

    Attributes:
        method: Optimization method ('L-BFGS-B', 'Powell', 'Nelder-Mead', etc.).
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        callback: Optional callback function called each iteration.
        verbose: Print progress if True.
    """
    method: str = "L-BFGS-B"
    max_iterations: int = 100
    tolerance: float = 1e-6
    callback: Callable[[np.ndarray, float, int], None] | None = None
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Result from optimization.

    Attributes:
        x: Optimal parameters.
        fun: Final objective value.
        success: Whether optimization succeeded.
        message: Status message.
        n_iterations: Number of iterations.
        history: List of (params, objective) at each iteration if tracked.
    """
    x: np.ndarray
    fun: float
    success: bool
    message: str
    n_iterations: int
    history: list[tuple[np.ndarray, float]] = field(default_factory=list)


def run_optimization(
    objective_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    config: OptimizerConfig | None = None,
) -> OptimizationResult:
    """Run optimization on an objective function.

    Args:
        objective_func: Function that takes parameters and returns scalar.
        x0: Initial parameter guess.
        bounds: (lower, upper) bounds arrays, or None for unbounded.
        config: Optimization configuration.

    Returns:
        OptimizationResult with optimal parameters and metadata.
    """
    if config is None:
        config = OptimizerConfig()

    # Track history
    history = []
    iteration_count = [0]  # Use list for closure mutation

    def wrapped_objective(x: np.ndarray) -> float:
        val = objective_func(x)
        history.append((x.copy(), val))

        if config.verbose and iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}: objective = {val:.6f}")

        if config.callback is not None:
            config.callback(x, val, iteration_count[0])

        iteration_count[0] += 1
        return val

    # Convert bounds to scipy format
    scipy_bounds = None
    if bounds is not None:
        lower, upper = bounds
        scipy_bounds = list(zip(lower, upper, strict=False))

    # Build method-specific options
    options = {
        'maxiter': config.max_iterations,
        'disp': False,
    }

    # Add tolerance option based on method
    if config.method in ("L-BFGS-B", "BFGS", "CG"):
        options['gtol'] = config.tolerance
    elif config.method in ("Nelder-Mead", "Powell"):
        options['xatol'] = config.tolerance
        options['fatol'] = config.tolerance
    else:
        options['ftol'] = config.tolerance

    # Run scipy optimization
    if config.verbose:
        print(f"Starting optimization with method={config.method}")
        print(f"  Parameters: {len(x0)}, Max iterations: {config.max_iterations}")

    result = scipy_opt.minimize(
        wrapped_objective,
        x0,
        method=config.method,
        bounds=scipy_bounds,
        options=options,
    )

    if config.verbose:
        print(f"Optimization {'succeeded' if result.success else 'failed'}: {result.message}")
        print(f"  Final objective: {result.fun:.6f}")
        print(f"  Iterations: {len(history)}")

    return OptimizationResult(
        x=result.x,
        fun=result.fun,
        success=result.success,
        message=result.message,
        n_iterations=len(history),
        history=history,
    )


def scipy_minimize(
    objective_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    method: str = "L-BFGS-B",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> OptimizationResult:
    """Convenience wrapper for scipy optimization.

    Args:
        objective_func: Objective function.
        x0: Initial parameters.
        bounds: Parameter bounds.
        method: Scipy method name.
        max_iterations: Max iterations.
        tolerance: Convergence tolerance.
        verbose: Print progress.

    Returns:
        OptimizationResult.
    """
    config = OptimizerConfig(
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )
    return run_optimization(objective_func, x0, bounds, config)


def pattern_search(
    objective_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    step_size: float = 0.1,
    min_step: float = 1e-4,
    max_iterations: int = 1000,
    verbose: bool = True,
) -> OptimizationResult:
    """Simple coordinate-wise pattern search.

    Gradient-free method that explores along each coordinate direction.

    Args:
        objective_func: Objective function.
        x0: Initial parameters.
        bounds: Parameter bounds.
        step_size: Initial step size (fraction of range).
        min_step: Minimum step size for convergence.
        max_iterations: Maximum iterations.
        verbose: Print progress.

    Returns:
        OptimizationResult.
    """
    x = x0.copy()
    n = len(x)

    if bounds is not None:
        lower, upper = bounds
        ranges = upper - lower
    else:
        ranges = np.ones(n)
        lower = -np.inf * np.ones(n)
        upper = np.inf * np.ones(n)

    current_step = step_size * ranges
    f_best = objective_func(x)
    history = [(x.copy(), f_best)]

    iteration = 0
    while iteration < max_iterations and np.max(current_step) > min_step:
        improved = False

        for i in range(n):
            # Try positive step
            x_new = x.copy()
            x_new[i] = np.clip(x[i] + current_step[i], lower[i], upper[i])
            f_new = objective_func(x_new)

            if f_new < f_best:
                x = x_new
                f_best = f_new
                improved = True
                history.append((x.copy(), f_best))
                continue

            # Try negative step
            x_new = x.copy()
            x_new[i] = np.clip(x[i] - current_step[i], lower[i], upper[i])
            f_new = objective_func(x_new)

            if f_new < f_best:
                x = x_new
                f_best = f_new
                improved = True
                history.append((x.copy(), f_best))

        if not improved:
            current_step *= 0.5

        iteration += 1

        if verbose and iteration % 50 == 0:
            print(f"  Pattern search iteration {iteration}: objective = {f_best:.6f}")

    success = np.max(current_step) <= min_step
    message = "Converged" if success else "Max iterations reached"

    if verbose:
        print(f"Pattern search {'converged' if success else 'stopped'}")
        print(f"  Final objective: {f_best:.6f}")

    return OptimizationResult(
        x=x,
        fun=f_best,
        success=success,
        message=message,
        n_iterations=iteration,
        history=history,
    )


def continuation_optimization(
    objective_func_factory: Callable[[float], Callable[[np.ndarray], float]],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    beta_schedule: list[float] = None,
    iterations_per_beta: int = 20,
    verbose: bool = True,
) -> OptimizationResult:
    """Optimization with continuation (beta scheduling).

    Used for topology optimization where beta controls binarization.
    Starts with soft projections and gradually increases sharpness.

    Args:
        objective_func_factory: Function that takes beta and returns objective.
        x0: Initial parameters.
        bounds: Parameter bounds.
        beta_schedule: List of beta values to use.
        iterations_per_beta: Iterations per beta level.
        verbose: Print progress.

    Returns:
        OptimizationResult from final optimization.
    """
    if beta_schedule is None:
        beta_schedule = [1, 2, 4, 8, 16, 32]

    x = x0.copy()
    all_history = []

    for beta in beta_schedule:
        if verbose:
            print(f"\n=== Beta = {beta} ===")

        objective_func = objective_func_factory(beta)

        result = run_optimization(
            objective_func,
            x,
            bounds,
            OptimizerConfig(
                method="L-BFGS-B",
                max_iterations=iterations_per_beta,
                verbose=verbose,
            ),
        )

        x = result.x
        all_history.extend(result.history)

    return OptimizationResult(
        x=x,
        fun=result.fun,
        success=result.success,
        message=f"Continuation completed with beta={beta_schedule[-1]}",
        n_iterations=len(all_history),
        history=all_history,
    )


def genetic_algorithm(
    objective_func: Callable[[np.ndarray], float],
    bounds: tuple[np.ndarray, np.ndarray],
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elite_count: int = 2,
    verbose: bool = True,
) -> OptimizationResult:
    """Genetic Algorithm for global optimization.

    Gradient-free evolutionary algorithm suitable for non-convex problems.

    Args:
        objective_func: Objective function to minimize.
        bounds: (lower, upper) bounds arrays.
        population_size: Number of individuals in population.
        generations: Number of generations.
        mutation_rate: Probability of mutation per gene.
        crossover_rate: Probability of crossover.
        elite_count: Number of best individuals to keep unchanged.
        verbose: Print progress.

    Returns:
        OptimizationResult.
    """
    lower, upper = bounds
    n_params = len(lower)
    ranges = upper - lower

    # Initialize population randomly within bounds
    population = lower + np.random.rand(population_size, n_params) * ranges

    # Evaluate initial fitness
    fitness = np.array([objective_func(ind) for ind in population])

    history = []
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    history.append((best_x.copy(), best_f))

    for gen in range(generations):
        # Selection (tournament)
        new_population = []

        # Keep elites
        elite_indices = np.argsort(fitness)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            t1 = np.random.choice(population_size, 3, replace=False)
            parent1 = population[t1[np.argmin(fitness[t1])]].copy()

            t2 = np.random.choice(population_size, 3, replace=False)
            parent2 = population[t2[np.argmin(fitness[t2])]].copy()

            # Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(n_params)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            # Mutation
            mask = np.random.rand(n_params) < mutation_rate
            mutation = (np.random.rand(n_params) - 0.5) * ranges * 0.1
            child[mask] += mutation[mask]

            # Enforce bounds
            child = np.clip(child, lower, upper)
            new_population.append(child)

        population = np.array(new_population[:population_size])

        # Evaluate new population
        fitness = np.array([objective_func(ind) for ind in population])

        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_x = population[gen_best_idx].copy()
            best_f = fitness[gen_best_idx]
            history.append((best_x.copy(), best_f))

        if verbose and gen % 10 == 0:
            print(f"  GA generation {gen}: best = {best_f:.6f}")

    if verbose:
        print(f"GA completed: best objective = {best_f:.6f}")

    return OptimizationResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"GA completed {generations} generations",
        n_iterations=generations,
        history=history,
    )


def particle_swarm(
    objective_func: Callable[[np.ndarray], float],
    bounds: tuple[np.ndarray, np.ndarray],
    n_particles: int = 30,
    max_iterations: int = 100,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    verbose: bool = True,
) -> OptimizationResult:
    """Particle Swarm Optimization (PSO).

    Swarm-based optimization inspired by bird flocking behavior.

    Args:
        objective_func: Objective function to minimize.
        bounds: (lower, upper) bounds arrays.
        n_particles: Number of particles in swarm.
        max_iterations: Maximum iterations.
        inertia: Inertia weight for velocity update.
        cognitive: Cognitive (personal best) coefficient.
        social: Social (global best) coefficient.
        verbose: Print progress.

    Returns:
        OptimizationResult.
    """
    lower, upper = bounds
    n_params = len(lower)
    ranges = upper - lower

    # Initialize particles
    positions = lower + np.random.rand(n_particles, n_params) * ranges
    velocities = (np.random.rand(n_particles, n_params) - 0.5) * ranges * 0.1

    # Evaluate initial positions
    fitness = np.array([objective_func(p) for p in positions])

    # Personal best
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    # Global best
    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    history = [(gbest_pos.copy(), gbest_fit)]

    for it in range(max_iterations):
        # Update velocities
        r1 = np.random.rand(n_particles, n_params)
        r2 = np.random.rand(n_particles, n_params)

        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest_pos - positions)
            + social * r2 * (gbest_pos - positions)
        )

        # Limit velocity
        max_vel = 0.2 * ranges
        velocities = np.clip(velocities, -max_vel, max_vel)

        # Update positions
        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

        # Evaluate
        fitness = np.array([objective_func(p) for p in positions])

        # Update personal best
        improved = fitness < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]

        # Update global best
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_pos = pbest_pos[best_idx].copy()
            gbest_fit = pbest_fit[best_idx]
            history.append((gbest_pos.copy(), gbest_fit))

        if verbose and it % 10 == 0:
            print(f"  PSO iteration {it}: best = {gbest_fit:.6f}")

    if verbose:
        print(f"PSO completed: best objective = {gbest_fit:.6f}")

    return OptimizationResult(
        x=gbest_pos,
        fun=gbest_fit,
        success=True,
        message=f"PSO completed {max_iterations} iterations",
        n_iterations=max_iterations,
        history=history,
    )


# Optional JAX-based optimization
if HAS_JAX and HAS_JAXOPT:
    def jaxopt_minimize(
        objective_func: Callable,
        x0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        max_iterations: int = 100,
        verbose: bool = True,
    ) -> OptimizationResult:
        """JAX-based optimization using jaxopt.

        Requires JAX and jaxopt to be installed.
        The objective function should be JAX-compatible.

        Args:
            objective_func: JAX-compatible objective function.
            x0: Initial parameters.
            bounds: Parameter bounds (used with ProjectedGradient).
            max_iterations: Maximum iterations.
            verbose: Print progress.

        Returns:
            OptimizationResult.
        """
        from jaxopt import LBFGS, ProjectedGradient
        from jaxopt.projection import projection_box

        x0_jax = jnp.array(x0)

        if bounds is not None:
            lower, upper = bounds
            lower_jax = jnp.array(lower)
            upper_jax = jnp.array(upper)

            def projection(x):
                return projection_box(x, (lower_jax, upper_jax))

            solver = ProjectedGradient(
                fun=objective_func,
                projection=projection,
                maxiter=max_iterations,
            )
        else:
            solver = LBFGS(
                fun=objective_func,
                maxiter=max_iterations,
            )

        result = solver.run(x0_jax)

        return OptimizationResult(
            x=np.array(result.params),
            fun=float(objective_func(result.params)),
            success=True,
            message="jaxopt completed",
            n_iterations=max_iterations,  # jaxopt doesn't easily expose this
            history=[],
        )


__all__ = [
    "OptimizerConfig",
    "OptimizationResult",
    "run_optimization",
    "scipy_minimize",
    "pattern_search",
    "continuation_optimization",
    "genetic_algorithm",
    "particle_swarm",
    "HAS_JAX",
    "HAS_JAXOPT",
]

# Add JAX optimizer to exports if available
if HAS_JAX and HAS_JAXOPT:
    __all__.append("jaxopt_minimize")

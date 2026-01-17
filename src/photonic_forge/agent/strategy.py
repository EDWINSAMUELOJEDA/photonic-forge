"""Exploration strategies for design agent.

Provides different algorithms for exploring the design space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ExplorationResult:
    """Result from exploration strategy.

    Attributes:
        next_point: Next parameter point to evaluate.
        uncertainty: Estimated uncertainty (for Bayesian methods).
        metadata: Additional strategy-specific info.
    """
    next_point: np.ndarray
    uncertainty: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
    ):
        """Initialize strategy.

        Args:
            bounds: (lower, upper) parameter bounds.
        """
        self.lower, self.upper = bounds
        self.n_params = len(self.lower)
        self.ranges = self.upper - self.lower

    @abstractmethod
    def suggest(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
    ) -> ExplorationResult:
        """Suggest next point to evaluate.

        Args:
            history_x: Previous parameter evaluations (n_samples, n_params).
            history_y: Previous objective values (n_samples,).

        Returns:
            ExplorationResult with suggested point.
        """
        pass

    def suggest_batch(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
        batch_size: int = 1,
    ) -> list[ExplorationResult]:
        """Suggest multiple points.

        Default implementation calls suggest() repeatedly.
        """
        results = []
        for _ in range(batch_size):
            result = self.suggest(history_x, history_y)
            results.append(result)
            # Add to history for diversity
            history_x = np.vstack([history_x, result.next_point])
            history_y = np.append(history_y, 0.0)  # Placeholder
        return results


class RandomStrategy(ExplorationStrategy):
    """Random sampling strategy.

    Simple baseline that samples uniformly from the parameter space.
    """

    def suggest(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
    ) -> ExplorationResult:
        """Suggest random point."""
        point = self.lower + np.random.rand(self.n_params) * self.ranges

        return ExplorationResult(
            next_point=point,
            uncertainty=1.0,
            metadata={"strategy": "random"},
        )


class LatinHypercubeStrategy(ExplorationStrategy):
    """Latin Hypercube Sampling for space-filling designs.

    Ensures good coverage of the parameter space.
    """

    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        n_samples: int = 100,
    ):
        super().__init__(bounds)
        self.n_samples = n_samples
        self._samples = self._generate_lhs()
        self._current_idx = 0

    def _generate_lhs(self) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((self.n_samples, self.n_params))

        for i in range(self.n_params):
            # Create stratified samples
            intervals = np.arange(self.n_samples) / self.n_samples
            samples[:, i] = intervals + np.random.rand(self.n_samples) / self.n_samples

        # Shuffle each dimension independently
        for i in range(self.n_params):
            np.random.shuffle(samples[:, i])

        # Scale to bounds
        samples = self.lower + samples * self.ranges

        return samples

    def suggest(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
    ) -> ExplorationResult:
        """Return next LHS sample."""
        if self._current_idx >= self.n_samples:
            # Regenerate if exhausted
            self._samples = self._generate_lhs()
            self._current_idx = 0

        point = self._samples[self._current_idx]
        self._current_idx += 1

        return ExplorationResult(
            next_point=point,
            uncertainty=0.5,
            metadata={"strategy": "lhs", "sample_idx": self._current_idx - 1},
        )


class LocalSearchStrategy(ExplorationStrategy):
    """Local search around best known point.

    Refines around the current best solution.
    """

    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        step_size: float = 0.1,
        decay: float = 0.95,
    ):
        super().__init__(bounds)
        self.step_size = step_size
        self.decay = decay
        self._current_step = step_size

    def suggest(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
    ) -> ExplorationResult:
        """Suggest point near current best."""
        if len(history_y) == 0:
            # No history, return random
            point = self.lower + np.random.rand(self.n_params) * self.ranges
        else:
            # Find best point
            best_idx = np.argmax(history_y)
            best_point = history_x[best_idx]

            # Perturb
            perturbation = np.random.randn(self.n_params) * self._current_step * self.ranges
            point = best_point + perturbation
            point = np.clip(point, self.lower, self.upper)

            # Decay step size
            self._current_step *= self.decay

        return ExplorationResult(
            next_point=point,
            uncertainty=self._current_step,
            metadata={"strategy": "local_search", "step_size": self._current_step},
        )


class HybridStrategy(ExplorationStrategy):
    """Hybrid strategy combining exploration and exploitation.

    Balances global exploration with local refinement.
    """

    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        exploration_rate: float = 0.3,
    ):
        super().__init__(bounds)
        self.exploration_rate = exploration_rate
        self.random = RandomStrategy(bounds)
        self.local = LocalSearchStrategy(bounds)

    def suggest(
        self,
        history_x: np.ndarray,
        history_y: np.ndarray,
    ) -> ExplorationResult:
        """Suggest point using hybrid approach."""
        if np.random.rand() < self.exploration_rate or len(history_y) < 10:
            result = self.random.suggest(history_x, history_y)
            result.metadata["mode"] = "exploration"
        else:
            result = self.local.suggest(history_x, history_y)
            result.metadata["mode"] = "exploitation"

        return result


__all__ = [
    "ExplorationResult",
    "ExplorationStrategy",
    "RandomStrategy",
    "LatinHypercubeStrategy",
    "LocalSearchStrategy",
    "HybridStrategy",
]

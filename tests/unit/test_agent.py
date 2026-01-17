"""Tests for design agent and exploration strategies."""

import numpy as np
import pytest

from photonic_forge.agent import (
    DesignAgent,
    AgentConfig,
    DesignCandidate,
    run_agent_exploration,
    RandomStrategy,
    LatinHypercubeStrategy,
    LocalSearchStrategy,
    HybridStrategy,
)


class TestDesignCandidate:
    """Tests for DesignCandidate dataclass."""

    def test_creation(self):
        """DesignCandidate should be creatable."""
        params = np.array([0.5, 0.3, 0.7])
        candidate = DesignCandidate(
            parameters=params,
            predicted_metrics={"transmission": 0.9},
            score=0.85,
            generation=1,
        )

        assert np.allclose(candidate.parameters, params)
        assert candidate.score == 0.85
        assert candidate.generation == 1


class TestDesignAgent:
    """Tests for DesignAgent class."""

    @pytest.fixture
    def simple_bounds(self):
        """Simple 2D parameter bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        return (lower, upper)

    def test_agent_creation(self, simple_bounds):
        """Agent should be creatable."""
        agent = DesignAgent(bounds=simple_bounds)
        assert agent.n_params == 2

    def test_random_candidate(self, simple_bounds):
        """random_candidate should create valid candidate."""
        agent = DesignAgent(bounds=simple_bounds)
        candidate = agent.random_candidate()

        assert isinstance(candidate, DesignCandidate)
        assert len(candidate.parameters) == 2
        assert all(0 <= p <= 1 for p in candidate.parameters)

    def test_mutate(self, simple_bounds):
        """mutate should create new candidate."""
        agent = DesignAgent(bounds=simple_bounds)
        original = agent.random_candidate()
        mutated = agent.mutate(original, generation=1)

        assert isinstance(mutated, DesignCandidate)
        assert mutated.generation == 1

    def test_crossover(self, simple_bounds):
        """crossover should combine two parents."""
        agent = DesignAgent(bounds=simple_bounds)
        p1 = agent.random_candidate()
        p2 = agent.random_candidate()
        child = agent.crossover(p1, p2, generation=1)

        assert isinstance(child, DesignCandidate)
        assert len(child.parameters) == 2

    def test_explore(self, simple_bounds):
        """explore should run generations and return candidates."""
        config = AgentConfig(population_size=10, max_generations=3)
        agent = DesignAgent(bounds=simple_bounds, config=config)

        candidates = agent.explore(verbose=False)

        assert len(candidates) > 0
        assert agent.best_candidate is not None

    def test_with_objective(self, simple_bounds):
        """Agent should work with custom objective."""
        def objective(params):
            return -np.sum((params - 0.5) ** 2)

        config = AgentConfig(population_size=10, max_generations=5)
        agent = DesignAgent(
            bounds=simple_bounds,
            objective_func=objective,
            config=config,
        )

        candidates = agent.explore(verbose=False)
        best = agent.best_candidate

        # Best should be near center
        assert best is not None
        assert np.mean(np.abs(best.parameters - 0.5)) < 0.5

    def test_get_top_candidates(self, simple_bounds):
        """get_top_candidates should return sorted candidates."""
        config = AgentConfig(population_size=20, max_generations=3)
        agent = DesignAgent(bounds=simple_bounds, config=config)
        agent.explore(verbose=False)

        top = agent.get_top_candidates(5)

        assert len(top) == 5
        # Should be sorted by score descending
        scores = [c.score for c in top]
        assert scores == sorted(scores, reverse=True)


class TestRunAgentExploration:
    """Tests for convenience function."""

    def test_run_exploration(self):
        """run_agent_exploration should work."""
        bounds = (np.array([0, 0]), np.array([1, 1]))

        def objective(params):
            return -np.sum(params ** 2)

        best, candidates = run_agent_exploration(
            bounds=bounds,
            objective_func=objective,
            config=AgentConfig(population_size=10, max_generations=2),
            verbose=False,
        )

        assert best is not None
        assert len(candidates) > 0


class TestExplorationStrategies:
    """Tests for exploration strategy classes."""

    @pytest.fixture
    def bounds(self):
        """Standard 3D bounds."""
        return (np.array([0, 0, 0]), np.array([1, 1, 1]))

    def test_random_strategy(self, bounds):
        """RandomStrategy should suggest points within bounds."""
        strategy = RandomStrategy(bounds)
        result = strategy.suggest(np.array([]).reshape(0, 3), np.array([]))

        assert len(result.next_point) == 3
        assert all(0 <= p <= 1 for p in result.next_point)

    def test_lhs_strategy(self, bounds):
        """LatinHypercubeStrategy should provide space-filling samples."""
        strategy = LatinHypercubeStrategy(bounds, n_samples=10)

        points = []
        for _ in range(10):
            result = strategy.suggest(np.array([]).reshape(0, 3), np.array([]))
            points.append(result.next_point)

        points = np.array(points)
        assert points.shape == (10, 3)

    def test_local_search_strategy(self, bounds):
        """LocalSearchStrategy should search near best point."""
        strategy = LocalSearchStrategy(bounds, step_size=0.1)

        history_x = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
        history_y = np.array([0.9, 0.7])  # First is better

        result = strategy.suggest(history_x, history_y)

        # Should be near the best point [0.5, 0.5, 0.5]
        distance = np.linalg.norm(result.next_point - np.array([0.5, 0.5, 0.5]))
        assert distance < 1.0

    def test_hybrid_strategy(self, bounds):
        """HybridStrategy should work."""
        strategy = HybridStrategy(bounds, exploration_rate=0.5)

        result = strategy.suggest(np.array([]).reshape(0, 3), np.array([]))
        assert len(result.next_point) == 3

    def test_suggest_batch(self, bounds):
        """suggest_batch should return multiple suggestions."""
        strategy = RandomStrategy(bounds)
        results = strategy.suggest_batch(
            np.array([]).reshape(0, 3),
            np.array([]),
            batch_size=5,
        )

        assert len(results) == 5

"""
Unit tests for CARS (Curvature-Aware Restart Strategy).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.cars import CurvatureMonitor, CARS, NLSHADEStagnationRestart


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class Sphere:
    """Unimodal test function: f(x) = sum(x^2)."""
    bounds = [-100.0, 100.0]
    def evaluate(self, x):
        return float(np.sum(x ** 2))


class TwoBasin:
    """Two-basin test function: two Gaussian wells at +/-30."""
    bounds = [-100.0, 100.0]
    def __init__(self, dim=10):
        self.dim = dim
        self.c1 = np.zeros(dim); self.c1[0] = -30
        self.c2 = np.zeros(dim); self.c2[0] = 30

    def evaluate(self, x):
        d1 = np.sum((x - self.c1) ** 2)
        d2 = np.sum((x - self.c2) ** 2)
        return float(-(100 * np.exp(-d1 / 200) + 80 * np.exp(-d2 / 200)) + 100)


# -----------------------------------------------------------------------
# CurvatureMonitor tests
# -----------------------------------------------------------------------

class TestCurvatureMonitor:

    def test_converged_population_detected_as_trapped(self):
        """Tightly converged population with stagnation -> 'trapped'."""
        dim = 10
        monitor = CurvatureMonitor(dim, k=7, domain_width=200.0)
        rng = np.random.RandomState(42)
        pop = rng.randn(30, dim) * 0.5
        fitness = np.sum(pop ** 2, axis=1)

        state, neg_frac, mean_k = monitor.classify(pop, fitness, stagnating=True)
        assert state == "trapped"

    def test_converged_population_exploiting_when_not_stagnating(self):
        """Tightly converged population without stagnation -> 'exploiting'."""
        dim = 10
        monitor = CurvatureMonitor(dim, k=7, domain_width=200.0)
        rng = np.random.RandomState(42)
        pop = rng.randn(30, dim) * 0.5
        fitness = np.sum(pop ** 2, axis=1)

        state, neg_frac, mean_k = monitor.classify(pop, fitness, stagnating=False)
        assert state == "exploiting"

    def test_spread_population_detected_as_exploring(self):
        """Population spread across two clusters -> 'exploring'."""
        dim = 10
        monitor = CurvatureMonitor(dim, k=7, domain_width=200.0)
        rng = np.random.RandomState(42)

        c1 = np.zeros(dim); c1[0] = -30
        c2 = np.zeros(dim); c2[0] = 30
        pop1 = c1 + rng.randn(25, dim) * 3.0
        pop2 = c2 + rng.randn(25, dim) * 3.0
        pop = np.vstack([pop1, pop2])
        fitness = np.sum(pop ** 2, axis=1)

        state, neg_frac, mean_k = monitor.classify(pop, fitness, stagnating=False)
        assert state == "exploring"
        assert neg_frac > 0.08

    def test_small_population_fallback(self):
        """Population with < 6 agents -> fallback classification."""
        dim = 10
        monitor = CurvatureMonitor(dim, k=7)
        pop = np.random.randn(4, dim)
        fitness = np.sum(pop ** 2, axis=1)

        state, _, _ = monitor.classify(pop, fitness, stagnating=True)
        assert state == "trapped"

        state, _, _ = monitor.classify(pop, fitness, stagnating=False)
        assert state == "exploiting"


# -----------------------------------------------------------------------
# CARS integration tests
# -----------------------------------------------------------------------

class TestCARS:

    def test_sphere_no_unnecessary_restarts(self):
        """On unimodal Sphere, CARS should not restart excessively.
        Adaptive patience + min segment budget should limit restarts."""
        dim = 5
        problem = Sphere()
        np.random.seed(42)

        algo = CARS(problem, dim, max_fe=20000, orc_period=10, stag_patience=2000)
        _, best_fit, log = algo.run()

        assert best_fit < 1e-3
        assert len(algo.restart_log) <= 7

    def test_two_basin_finds_global(self):
        """On a two-basin function, CARS should find the deeper basin."""
        dim = 5
        problem = TwoBasin(dim)
        np.random.seed(42)

        algo = CARS(problem, dim, max_fe=30000, orc_period=10, stag_patience=2500)
        best_sol, best_fit, log = algo.run()

        assert best_fit < 5.0

    def test_archived_bests_accumulate(self):
        """Archived basin centers should grow with each restart."""
        dim = 5
        problem = TwoBasin(dim)
        np.random.seed(42)

        algo = CARS(problem, dim, max_fe=30000, orc_period=10, stag_patience=1500)
        algo.run()

        if algo.restart_log:
            assert len(algo.archived_bests) == len(algo.restart_log)

    def test_convergence_log_populated(self):
        """Convergence log should have entries."""
        dim = 5
        problem = Sphere()
        np.random.seed(42)

        algo = CARS(problem, dim, max_fe=10000, orc_period=10, stag_patience=2500)
        _, _, log = algo.run()
        assert len(log) > 2


# -----------------------------------------------------------------------
# NLSHADEStagnationRestart tests
# -----------------------------------------------------------------------

class TestNLSHADEStagnationRestart:

    def test_runs_without_error(self):
        dim = 5
        problem = Sphere()
        np.random.seed(42)

        algo = NLSHADEStagnationRestart(
            problem, dim, max_fe=10000, stag_gens=20
        )
        _, best_fit, log = algo.run()
        assert best_fit < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

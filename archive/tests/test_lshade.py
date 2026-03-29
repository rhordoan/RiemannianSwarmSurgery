"""
Tests for the standalone L-SHADE implementation.

Validates:
1. L-SHADE converges on Sphere function (should reach ~0)
2. L-SHADE converges on Rastrigin (should reach reasonable error)
3. Success history updates correctly
4. Population size reduces over time (LPSR)
5. External archive stores replaced parents
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.lshade import LSHADE


class SphereProblem:
    """Sphere function: f(x) = sum(x^2). Optimum at origin = 0."""
    def __init__(self, dim=10):
        self.bounds = [-100, 100]
        self.dim = dim

    def evaluate(self, x):
        return float(np.sum(x ** 2))


class RastriginProblem:
    """Rastrigin function. Optimum at origin = 0."""
    def __init__(self, dim=10):
        self.bounds = [-5.12, 5.12]
        self.dim = dim

    def evaluate(self, x):
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def test_lshade_sphere():
    """L-SHADE should converge near 0 on Sphere in 10D."""
    print("Test: L-SHADE on Sphere (10D)...")

    np.random.seed(42)
    problem = SphereProblem(dim=10)
    opt = LSHADE(problem, dim=10, pop_size=50, max_fe=50000)
    history = opt.run()

    final = opt.best_fitness
    print(f"  Final error: {final:.4e}")
    assert final < 1e-5, f"FAIL: L-SHADE should reach <1e-5 on Sphere, got {final:.4e}"
    print("  PASS: L-SHADE converges on Sphere")


def test_lshade_rastrigin():
    """L-SHADE should reach reasonable error on Rastrigin 10D."""
    print("\nTest: L-SHADE on Rastrigin (10D)...")

    np.random.seed(42)
    problem = RastriginProblem(dim=10)
    opt = LSHADE(problem, dim=10, pop_size=100, max_fe=100000)
    history = opt.run()

    final = opt.best_fitness
    print(f"  Final error: {final:.4e}")
    assert final < 100, f"FAIL: L-SHADE should reach <100 on Rastrigin, got {final:.4e}"
    print("  PASS: L-SHADE reaches reasonable error on Rastrigin")


def test_success_history_updates():
    """Success history M_F and M_CR should change from initial 0.5."""
    print("\nTest: Success history adaptation...")

    np.random.seed(42)
    problem = SphereProblem(dim=5)
    opt = LSHADE(problem, dim=5, pop_size=30, max_fe=5000, H=6)

    # Run a few steps
    for _ in range(20):
        opt.step()

    # Check that history has been updated (not all 0.5)
    f_changed = np.any(opt.M_F != 0.5)
    cr_changed = np.any(opt.M_CR != 0.5)

    print(f"  M_F: {opt.M_F}")
    print(f"  M_CR: {opt.M_CR}")
    assert f_changed, "FAIL: M_F was never updated"
    assert cr_changed, "FAIL: M_CR was never updated"
    print("  PASS: Success history adapted from initial values")


def test_lpsr():
    """Population should reduce over time."""
    print("\nTest: Linear Population Size Reduction...")

    np.random.seed(42)
    problem = SphereProblem(dim=5)
    initial_pop = 50
    opt = LSHADE(problem, dim=5, pop_size=initial_pop, max_fe=10000)

    sizes = [len(opt.pop)]
    for _ in range(50):
        opt.step()
        sizes.append(len(opt.pop))

    print(f"  Population sizes: {sizes[0]} -> {sizes[-1]}")
    assert sizes[-1] < sizes[0], "FAIL: Population should decrease"
    assert sizes[-1] >= 4, "FAIL: Population should not go below minimum"
    print("  PASS: LPSR reduces population over time")


def test_external_archive():
    """External archive should accumulate replaced parents."""
    print("\nTest: External archive...")

    np.random.seed(42)
    problem = SphereProblem(dim=5)
    opt = LSHADE(problem, dim=5, pop_size=30, max_fe=5000)

    # Run enough steps for some replacements
    for _ in range(30):
        opt.step()

    archive_size = len(opt.archive)
    print(f"  Archive size after 30 gens: {archive_size}")
    assert archive_size > 0, "FAIL: Archive should contain replaced parents"
    assert archive_size <= opt.archive_max_size, \
        "FAIL: Archive exceeds max size"
    print("  PASS: External archive collects replaced parents")


def test_convergence_history():
    """run() should return a non-empty, non-increasing history."""
    print("\nTest: Convergence history monotonicity...")

    np.random.seed(42)
    problem = SphereProblem(dim=5)
    opt = LSHADE(problem, dim=5, pop_size=30, max_fe=3000)
    history = opt.run()

    assert len(history) > 0, "FAIL: History should not be empty"

    # Check monotonicity (best-ever should never increase)
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-10, \
            f"FAIL: History not monotonic at step {i}: " \
            f"{history[i]} > {history[i-1]}"

    print(f"  History length: {len(history)}")
    print(f"  Start: {history[0]:.4e}, End: {history[-1]:.4e}")
    print("  PASS: Convergence history is monotonically non-increasing")


if __name__ == "__main__":
    test_lshade_sphere()
    test_lshade_rastrigin()
    test_success_history_updates()
    test_lpsr()
    test_external_archive()
    test_convergence_history()
    print("\n=== All L-SHADE tests passed ===")

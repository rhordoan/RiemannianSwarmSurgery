"""
Synthetic Validation: prove ORC correctly detects basin structure.

Constructs landscapes with known numbers of basins (2, 4, 8) and
verifies that the CurvatureMonitor classifies population topology
correctly at each phase of optimisation.

Usage:
    python benchmarks/synthetic_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.cars import CurvatureMonitor, CARS
from benchmarks.nlshade import NLSHADE


# -----------------------------------------------------------------------
# Synthetic multimodal landscapes with known basin structure
# -----------------------------------------------------------------------

class MultiBasinFunction:
    """
    Sum of inverted Gaussians creating `n_basins` distinct basins.

    The global optimum is at `centers[0]` with depth `depths[0]`.
    All basins are separated by ridges proportional to `separation`.

    evaluate() returns a non-negative error (0 at the global optimum),
    matching the convention of CEC benchmark wrappers.
    """

    def __init__(self, dim, n_basins=2, separation=40.0, seed=42):
        self.dim = dim
        self.n_basins = n_basins
        self.bounds = [-100.0, 100.0]

        rng = np.random.RandomState(seed)
        self.centers = []
        self.depths = []
        self.widths = []

        for i in range(n_basins):
            angle = 2 * np.pi * i / n_basins
            center = np.zeros(dim)
            center[0] = separation * np.cos(angle)
            if dim > 1:
                center[1] = separation * np.sin(angle)
            if dim > 2:
                center[2:] = rng.randn(dim - 2) * 5.0
            self.centers.append(center)
            self.depths.append(100.0 * (1.0 - 0.1 * i))
            self.widths.append(10.0 + rng.rand() * 5.0)

        self._global_min = self._raw(self.centers[0])

    def _raw(self, x):
        val = 0.0
        for c, d, w in zip(self.centers, self.depths, self.widths):
            dist2 = np.sum((x - c) ** 2)
            val -= d * np.exp(-dist2 / (2.0 * w ** 2))
        return val

    def evaluate(self, x):
        return max(0.0, self._raw(x) - self._global_min)


# -----------------------------------------------------------------------
# ORC topology validation
# -----------------------------------------------------------------------

def validate_orc_detection(dim=10, n_basins=4, n_agents=80, seed=0):
    """
    Place agents across known basins and verify ORC detects multi-basin
    structure, then concentrate agents in one basin and verify ORC
    detects single-basin convergence.

    Returns dict with detection results.
    """
    rng = np.random.RandomState(seed)
    func = MultiBasinFunction(dim, n_basins=n_basins, seed=seed)
    monitor = CurvatureMonitor(dim, k=7)

    # Phase 1: agents spread across all basins (early/mid-run scenario)
    agents_per_basin = n_agents // n_basins
    pop_multi = []
    for center in func.centers:
        for _ in range(agents_per_basin):
            x = center + rng.randn(dim) * 3.0
            x = np.clip(x, -100.0, 100.0)
            pop_multi.append(x)
    pop_multi = np.array(pop_multi)
    fit_multi = np.array([func.evaluate(x) for x in pop_multi])

    state_multi, neg_frac_multi, mean_k_multi = monitor.classify(
        pop_multi, fit_multi, stagnating=False
    )

    # Phase 2: realistic late-run NL-SHADE population after PSR
    # (~25 agents remaining, moderately tight cluster in one basin)
    n_converged = max(25, n_agents // 4)
    pop_single = []
    for _ in range(n_converged):
        x = func.centers[0] + rng.randn(dim) * 1.0
        x = np.clip(x, -100.0, 100.0)
        pop_single.append(x)
    pop_single = np.array(pop_single)
    fit_single = np.array([func.evaluate(x) for x in pop_single])

    state_single, neg_frac_single, mean_k_single = monitor.classify(
        pop_single, fit_single, stagnating=True
    )

    return {
        "n_basins": n_basins,
        "multi_basin_state": state_multi,
        "multi_basin_neg_frac": neg_frac_multi,
        "multi_basin_mean_k": mean_k_multi,
        "single_basin_state": state_single,
        "single_basin_neg_frac": neg_frac_single,
        "single_basin_mean_k": mean_k_single,
        "single_basin_n_agents": n_converged,
    }


def validate_cars_restarts(dim=10, n_basins=4, max_fe=50000, seed=0):
    """
    Run CARS on a multi-basin function and verify that restarts occur
    and the algorithm explores multiple basins.
    """
    func = MultiBasinFunction(dim, n_basins=n_basins, seed=seed)
    np.random.seed(seed)

    algo = CARS(func, dim, max_fe=max_fe, orc_period=15, stag_gens=30)
    best_sol, best_fit, log = algo.run()

    closest_basin = -1
    min_dist = float("inf")
    for i, c in enumerate(func.centers):
        d = np.linalg.norm(best_sol - c)
        if d < min_dist:
            min_dist = d
            closest_basin = i

    return {
        "n_basins": n_basins,
        "best_fitness": best_fit,
        "n_restarts": len(algo.restart_log),
        "closest_basin": closest_basin,
        "dist_to_basin": min_dist,
        "n_archived": len(algo.archived_bests),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Validation: ORC Basin Detection")
    print("=" * 70)

    for n_basins in [2, 4, 8]:
        result = validate_orc_detection(dim=10, n_basins=n_basins, seed=42,
                                        n_agents=80)
        multi_ok = result["multi_basin_state"] == "exploring"
        single_ok = result["single_basin_state"] == "trapped"
        print(
            f"\n  {n_basins} basins: "
            f"multi-basin -> {result['multi_basin_state']!r} "
            f"(neg_frac={result['multi_basin_neg_frac']:.3f}) "
            f"{'PASS' if multi_ok else 'FAIL'}"
        )
        print(
            f"           "
            f"single-basin -> {result['single_basin_state']!r} "
            f"(neg_frac={result['single_basin_neg_frac']:.3f}) "
            f"{'PASS' if single_ok else 'FAIL'}"
        )

    print("\n" + "=" * 70)
    print("Synthetic Validation: CARS Restart Behaviour")
    print("=" * 70)

    for n_basins in [2, 4]:
        result = validate_cars_restarts(dim=10, n_basins=n_basins,
                                        max_fe=50000, seed=42)
        print(
            f"\n  {n_basins} basins: "
            f"best={result['best_fitness']:.6e}  "
            f"restarts={result['n_restarts']}  "
            f"closest_basin={result['closest_basin']}  "
            f"dist={result['dist_to_basin']:.2f}  "
            f"archived={result['n_archived']}"
        )

    print("\n" + "=" * 70)
    print("Done.")

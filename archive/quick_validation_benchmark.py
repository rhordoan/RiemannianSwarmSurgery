"""
Quick validation benchmark: CEC 2017 subset, D=10, 5 runs.

Targets composition/hybrid functions where CARS restart logic matters,
plus a unimodal control (F1) to verify no regression.

Usage:
    python quick_validation_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import csv
import time
import numpy as np
import opfunu
import warnings
warnings.filterwarnings("ignore")

from src.cars import CARS, NLSHADEStagnationRestart
from benchmarks.nlshade import NLSHADE
from benchmarks.jso import JSO


class CEC2017Wrapper:
    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2017")
        self.problem = problem_class(ndim=dim)
        self.f_bias = self.problem.f_bias
        self.bounds = [-100.0, 100.0]

    def evaluate(self, x):
        return max(0.0, self.problem.evaluate(x) - self.f_bias)


def run_cars(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = CARS(problem, dim, max_fe=max_fe)
    _, best_fit, log = algo.run()
    return best_fit, len(algo.restart_log)


def run_nlshade(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = NLSHADE(problem, dim, max_fe=max_fe)
    algo.run()
    return algo.best_fitness, 0


def run_nlshade_r(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = NLSHADEStagnationRestart(problem, dim, max_fe=max_fe)
    _, best_fit, _ = algo.run()
    return best_fit, len(algo.tabu_zones)


def run_jso(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = JSO(problem, dim, max_fe=max_fe)
    algo.run()
    return algo.best_fitness, 0


ALGORITHMS = {
    "CARS":       run_cars,
    "NL-SHADE":   run_nlshade,
    "NL-SHADE-R": run_nlshade_r,
    "jSO":        run_jso,
}

FUNC_NUMS = [1, 4, 15, 16, 21, 22, 23, 24, 25]
DIM = 10
N_RUNS = 5
BUDGET = 10000 * DIM


def main():
    output_dir = "results/quick_validation"
    os.makedirs(output_dir, exist_ok=True)

    from collections import defaultdict
    all_results = defaultdict(list)
    algo_names = list(ALGORITHMS.keys())

    total = len(FUNC_NUMS) * len(algo_names) * N_RUNS
    done = 0
    t_global = time.time()

    for func_num in FUNC_NUMS:
        for algo_name in algo_names:
            for run_id in range(N_RUNS):
                seed = 1000 * func_num + 100 * DIM + run_id
                problem = CEC2017Wrapper(func_num, DIM)
                t0 = time.time()
                best_fit, n_restarts = ALGORITHMS[algo_name](
                    problem, DIM, BUDGET, seed
                )
                elapsed = time.time() - t0
                done += 1

                key = (algo_name, func_num)
                all_results[key].append({
                    "error": best_fit,
                    "restarts": n_restarts,
                    "time": elapsed,
                })

                if done % 10 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {algo_name} F{func_num} "
                        f"run{run_id}: {best_fit:.4e}  "
                        f"({elapsed:.1f}s)"
                    )

    total_time = time.time() - t_global
    print(f"\nAll {total} trials completed in {total_time:.0f}s\n")

    summary_rows = []
    for (algo_name, func_num), results in sorted(all_results.items()):
        errors = [r["error"] for r in results]
        restarts = [r["restarts"] for r in results]
        times = [r["time"] for r in results]
        errs = np.array(errors)
        row = {
            "algorithm": algo_name,
            "function": func_num,
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "std": float(np.std(errs)),
            "best": float(np.min(errs)),
            "worst": float(np.max(errs)),
            "mean_restarts": float(np.mean(restarts)),
            "mean_time": float(np.mean(times)),
        }
        summary_rows.append(row)

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary saved to {csv_path}\n")

    print("=" * 100)
    header = f"{'F':>3s} | " + " | ".join(f"{a:>16s}" for a in algo_names)
    print(header)
    print("=" * 100)

    from collections import defaultdict as dd
    wins = dd(int)
    losses = dd(int)

    for func_num in FUNC_NUMS:
        means = {}
        for algo_name in algo_names:
            key = (algo_name, func_num)
            if key in all_results:
                errors = [r["error"] for r in all_results[key]]
                means[algo_name] = np.mean(errors)

        best_mean = min(means.values()) if means else float("inf")
        vals = []
        for a in algo_names:
            if a in means:
                m = means[a]
                marker = " *" if m <= best_mean * 1.001 else "  "
                vals.append(f"{m:14.4e}{marker}")
            else:
                vals.append(f"{'N/A':>16s}")
        print(f"F{func_num:>2d} | " + " | ".join(vals))

        cars_mean = means.get("CARS", float("inf"))
        for a in algo_names:
            if a == "CARS":
                continue
            other_mean = means.get(a, float("inf"))
            if cars_mean < other_mean * 0.999:
                wins[a] += 1
            elif other_mean < cars_mean * 0.999:
                losses[a] += 1

    print("=" * 100)
    print("\nCARS win/loss (by mean error):")
    for a in algo_names:
        if a == "CARS":
            continue
        w = wins[a]
        l = losses[a]
        t = len(FUNC_NUMS) - w - l
        print(f"  vs {a:>12s}: {w}W / {l}L / {t}T")

    restarts_info = []
    for func_num in FUNC_NUMS:
        key = ("CARS", func_num)
        if key in all_results:
            mean_r = np.mean([r["restarts"] for r in all_results[key]])
            restarts_info.append(f"F{func_num}={mean_r:.1f}")
    print(f"\nCARS avg restarts: {', '.join(restarts_info)}")


if __name__ == "__main__":
    main()

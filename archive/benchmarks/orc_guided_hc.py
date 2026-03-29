"""
ORC-Guided Hill Climber Experiment.

Compares the quality of escape decisions: from a local optimum, each strategy
makes a fixed number of escape attempts. We measure:
  - Fraction of escapes that reach a strictly better local optimum
  - Mean fitness improvement per escape
  - Best fitness found after T escape attempts

Strategies:
  1. ORC-escape: follow the min-ORC direction, then HC to new optimum
  2. Random-neighbor: follow a uniformly random neighbor, then HC
  3. Worst-ORC: follow the max-ORC (worst) direction, then HC
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nk_landscape import NKLandscape
from src.orc_discrete import (
    hill_climb,
    compute_orc_neighborhood,
)


def run_escape_chain(fitness, neighbor_fn, space_size, n_escapes, strategy, rng, gamma=1.0):
    """
    Start from a random solution, HC to a local optimum, then make n_escapes
    escape attempts using the given strategy. Track fitness improvement.
    """
    start = rng.randint(space_size)
    current = hill_climb(start, fitness, neighbor_fn)
    best = fitness[current]
    trajectory = [float(fitness[current])]
    n_improved = 0

    for _ in range(n_escapes):
        nbrs = neighbor_fn(current)

        if strategy == "orc":
            orc_dict = compute_orc_neighborhood(current, fitness, neighbor_fn, gamma)
            escape = min(orc_dict, key=orc_dict.get)
        elif strategy == "worst_orc":
            orc_dict = compute_orc_neighborhood(current, fitness, neighbor_fn, gamma)
            escape = max(orc_dict, key=orc_dict.get)
        elif strategy == "random":
            escape = nbrs[rng.randint(len(nbrs))]
        else:
            raise ValueError(strategy)

        new_opt = hill_climb(escape, fitness, neighbor_fn)

        if fitness[new_opt] < fitness[current]:
            n_improved += 1
        if fitness[new_opt] < best:
            best = fitness[new_opt]

        current = new_opt
        trajectory.append(float(fitness[current]))

    return {
        "best_fitness": float(best),
        "final_fitness": float(fitness[current]),
        "n_improved": n_improved,
        "trajectory": trajectory,
    }


def _run_one(args):
    K, seed, trial, n_escapes = args
    nk = NKLandscape(N=16, K=K, seed=seed)
    global_best = float(nk.fitness.min())

    results = {}
    for name, strategy in [("ORC-escape", "orc"), ("Random-neighbor", "random"), ("Worst-ORC", "worst_orc")]:
        rng = np.random.RandomState(1000 * seed + trial)
        results[name] = run_escape_chain(
            nk.fitness, nk.neighbor_fn, nk.space_size, n_escapes, strategy, rng
        )

    return {"K": K, "seed": seed, "trial": trial, "global_best": global_best, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-escapes", type=int, default=20)
    parser.add_argument("--out", default="results/orc_guided_hc.json")
    args = parser.parse_args()

    tasks = []
    for K in [4, 8, 12]:
        for seed in range(args.n_instances):
            for trial in range(args.n_trials):
                tasks.append((K, seed, trial, args.n_escapes))

    print(f"ORC-Guided HC Experiment")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Workers: {args.workers}")
    print(f"  Escapes per run: {args.n_escapes}")

    with Pool(args.workers) as pool:
        all_results = list(pool.imap_unordered(_run_one, tasks))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {args.out}")

    for K in [4, 8, 12]:
        entries = [e for e in all_results if e["K"] == K]
        print(f"\n  NK K={K} ({len(entries)} runs, {args.n_escapes} escapes each):")
        for alg in ["ORC-escape", "Random-neighbor", "Worst-ORC"]:
            best_fits = [e["results"][alg]["best_fitness"] for e in entries]
            n_imp = [e["results"][alg]["n_improved"] for e in entries]
            global_bests = [e["global_best"] for e in entries]
            gap = [bf - gb for bf, gb in zip(best_fits, global_bests)]
            print(f"    {alg:18s}: mean_best={np.mean(best_fits):.4f}  "
                  f"mean_gap={np.mean(gap):.4f}  "
                  f"mean_improvements={np.mean(n_imp):.1f}/{args.n_escapes}")


if __name__ == "__main__":
    main()

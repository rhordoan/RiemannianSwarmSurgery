#!/usr/bin/env python3
"""
Compute random/worst-ORC/HC baselines for NAS-Bench-201 local optima.

Uses the existing landscape_analysis.json (which has ORC values and local optima
indices) plus the real NATS-Bench fitness array to compute:
  1. %Rand: for each local optimum, pick 30 random neighbors, hill-climb from
     each, fraction that reach a strictly better local optimum.
  2. %Worst: follow the max-ORC (most positive curvature) neighbor, hill-climb,
     check if better.
  3. HC%: 50 hill-climbing trials from random starts, fraction that find the
     global optimum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.orc_nas import NASBench201, SPACE_SIZE, get_neighbor_indices


def hill_climb_to_local_optimum(start_idx: int, fitness: np.ndarray) -> int:
    current = start_idx
    while True:
        nbrs = get_neighbor_indices(current)
        current_f = fitness[current]
        improving = [(n, fitness[n]) for n in nbrs if fitness[n] < current_f]
        if not improving:
            return current
        best_nbr = min(improving, key=lambda x: (x[1], x[0]))[0]
        current = best_nbr


def main():
    print("Loading NAS-Bench-201 real data...", flush=True)
    bench = NASBench201(dataset="cifar10",
                        data_path="data/NATS-tss-v1_0-3ffb9-simple")
    fitness = bench.fitness

    print("Loading landscape_analysis.json...", flush=True)
    with open("results/landscape_analysis.json") as f:
        data = json.load(f)

    local_optima_indices = data["local_optima_indices"]
    local_optima_fitnesses = data["local_optima_fitnesses"]
    orc_analyses = data["orc_analyses"]

    n_optima = len(local_optima_indices)
    print(f"Local optima: {n_optima}", flush=True)

    global_opt_idx = local_optima_indices[
        int(np.argmin(local_optima_fitnesses))
    ]
    global_opt_fit = fitness[global_opt_idx]
    print(f"Global optimum: idx={global_opt_idx}, "
          f"fitness={global_opt_fit:.4f}, "
          f"accuracy={100 - global_opt_fit:.4f}%", flush=True)

    opt_set = set(local_optima_indices)
    opt_fitness_map = dict(zip(local_optima_indices, local_optima_fitnesses))

    n_random_trials = 30
    rng = np.random.RandomState(42)

    rand_leads_better = []
    worst_leads_better = []
    orc_leads_better = []

    for i, analysis in enumerate(orc_analyses):
        opt_idx = analysis["opt_idx"]
        opt_fit = fitness[opt_idx]
        orc_values = analysis["orc_values"]  # {str(nbr_idx): orc_val}

        neighbors = [int(k) for k in orc_values.keys()]
        n_nbrs = len(neighbors)

        # --- %Rand: 30 random neighbor picks, hill-climb, check if better ---
        n_better_rand = 0
        for _ in range(n_random_trials):
            rand_nbr = neighbors[rng.randint(n_nbrs)]
            dest = hill_climb_to_local_optimum(rand_nbr, fitness)
            if fitness[dest] < opt_fit:
                n_better_rand += 1
        rand_leads_better.append(n_better_rand / n_random_trials)

        # --- %Worst: follow max-ORC neighbor, hill-climb ---
        max_orc_nbr = max(orc_values, key=lambda k: orc_values[k])
        max_orc_nbr_idx = int(max_orc_nbr)
        dest_worst = hill_climb_to_local_optimum(max_orc_nbr_idx, fitness)
        worst_leads_better.append(1.0 if fitness[dest_worst] < opt_fit else 0.0)

        # --- %ORC: follow min-ORC neighbor (already in JSON, recompute for consistency) ---
        min_orc_nbr = analysis["min_orc_neighbor"]
        if analysis["has_negative_orc"]:
            dest_orc = hill_climb_to_local_optimum(min_orc_nbr, fitness)
            orc_leads_better.append(
                1.0 if fitness[dest_orc] < opt_fit else 0.0
            )
        else:
            orc_leads_better.append(0.0)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_optima} local optima", flush=True)

    pct_rand = 100.0 * np.mean(rand_leads_better)
    pct_worst = 100.0 * np.mean(worst_leads_better)
    pct_orc = 100.0 * np.mean(orc_leads_better)
    adv = pct_orc / pct_rand if pct_rand > 0 else float("inf")

    print(f"\n--- NAS-Bench-201 Baselines ---")
    print(f"%ORC->better:   {pct_orc:.1f}%")
    print(f"%Rand->better:  {pct_rand:.1f}%")
    print(f"%Worst->better: {pct_worst:.1f}%")
    print(f"Advantage:      {adv:.1f}x")

    # --- HC success rate: 50 random-start hill climbs ---
    print("\nComputing HC success rate (50 trials)...", flush=True)
    n_hc_trials = 50
    n_hc_success = 0
    for t in range(n_hc_trials):
        start = rng.randint(SPACE_SIZE)
        dest = hill_climb_to_local_optimum(start, fitness)
        if fitness[dest] <= global_opt_fit + 1e-9:
            n_hc_success += 1
    hc_pct = 100.0 * n_hc_success / n_hc_trials
    print(f"HC success rate: {hc_pct:.1f}% ({n_hc_success}/{n_hc_trials})")

    print(f"\n--- Table 1 row ---")
    print(f"NAS-201 & 625 & {pct_orc:.1f} & {pct_rand:.1f} & "
          f"{pct_worst:.1f} & {adv:.1f}$\\times$ & {hc_pct:.1f}")

    results = {
        "pct_orc": pct_orc,
        "pct_rand": pct_rand,
        "pct_worst": pct_worst,
        "advantage": adv,
        "hc_pct": hc_pct,
        "n_optima": n_optima,
        "n_hc_trials": n_hc_trials,
        "n_random_trials": n_random_trials,
    }
    out_path = Path("results/nas201_baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

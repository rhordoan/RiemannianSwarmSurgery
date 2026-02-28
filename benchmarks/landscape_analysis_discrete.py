#!/usr/bin/env python3
"""
Unified Landscape Analysis for PPSN 2026 Paper.

Runs ORC analysis + classical FLA metrics + algorithm performance
on NK landscapes and W-model instances.

Output: results/landscape_analysis_discrete.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Worker initialization (shared data via globals)
# ---------------------------------------------------------------------------

_W_FITNESS = None
_W_NEIGHBOR_FN = None


def _init_worker(fitness: np.ndarray, neighbor_fn_name: str, landscape_args: dict):
    global _W_FITNESS, _W_NEIGHBOR_FN
    _W_FITNESS = fitness
    if neighbor_fn_name == 'bitflip':
        n_bits = landscape_args['n_bits']
        _W_NEIGHBOR_FN = lambda idx: [idx ^ (1 << i) for i in range(n_bits)]


# ---------------------------------------------------------------------------
# Per-instance analysis worker
# ---------------------------------------------------------------------------

def _analyze_instance(args: dict) -> dict:
    """Analyze a single landscape instance (called in worker process)."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.orc_discrete import (
        full_landscape_analysis,
        find_all_local_optima,
        compute_basin_sizes,
    )
    from src.landscape_metrics import compute_all_metrics

    benchmark_type = args['type']
    config = args['config']
    seed = args['seed']

    if benchmark_type == 'NK':
        from src.nk_landscape import NKLandscape
        landscape = NKLandscape(
            N=config['N'], K=config['K'],
            model=config.get('model', 'adjacent'), seed=seed,
        )
        fitness = landscape.fitness
        neighbor_fn = landscape.neighbor_fn
        space_size = landscape.space_size
        global_opt = landscape.global_optimum()
        label = f"NK(N={config['N']},K={config['K']},model={config.get('model','adjacent')})"

    elif benchmark_type == 'WModel':
        from src.wmodel import WModel
        landscape = WModel(
            n=config['n'], nu=config['nu'],
            gamma=config.get('gamma', 0),
            mu=config.get('mu', 1), seed=seed,
        )
        fitness = landscape.fitness
        neighbor_fn = landscape.neighbor_fn
        space_size = landscape.space_size
        global_opt = landscape.global_optimum()
        label = f"W(n={config['n']},nu={config['nu']})"
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    gamma_orc = args.get('gamma', 1.0)

    # ORC landscape analysis (with random-direction baseline)
    t0 = time.perf_counter()
    orc_result = full_landscape_analysis(
        space_size, fitness, neighbor_fn, gamma_orc,
        n_random_trials=30, seed=seed,
    )
    orc_time = time.perf_counter() - t0

    # Classical metrics
    t0 = time.perf_counter()
    classical = compute_all_metrics(
        fitness, neighbor_fn, global_opt,
        orc_result['basin_sizes'],
        orc_result['n_local_optima'],
        seed=seed,
    )
    classical_time = time.perf_counter() - t0

    # Algorithm performance
    t0 = time.perf_counter()
    algo_results = _run_algorithms(fitness, neighbor_fn, space_size, seed)
    algo_time = time.perf_counter() - t0

    # ORC summary statistics
    all_min_orcs = [a['min_orc'] for a in orc_result['orc_analyses']]
    all_orc_values = []
    for a in orc_result['orc_analyses']:
        all_orc_values.extend(a['orc_values'].values())

    basin_list = list(orc_result['basin_sizes'].values())

    return {
        'label': label,
        'type': benchmark_type,
        'config': config,
        'seed': seed,
        'space_size': space_size,
        # ORC features
        'n_local_optima': orc_result['n_local_optima'],
        'frac_negative_orc': orc_result['frac_with_negative_orc'],
        'frac_leads_to_better': orc_result['frac_leads_to_better'],
        'n_leads_to_better': orc_result['n_leads_to_better'],
        'frac_random_leads_to_better': orc_result['frac_random_leads_to_better'],
        'frac_worst_orc_leads_to_better': orc_result['frac_worst_orc_leads_to_better'],
        'mean_min_orc': float(np.mean(all_min_orcs)) if all_min_orcs else 0.0,
        'std_min_orc': float(np.std(all_min_orcs)) if all_min_orcs else 0.0,
        'mean_orc': float(np.mean(all_orc_values)) if all_orc_values else 0.0,
        'basin_entropy': classical['basin_entropy'],
        'mean_basin_size': float(np.mean(basin_list)) if basin_list else 0.0,
        'max_basin_size': int(np.max(basin_list)) if basin_list else 0,
        # Classical metrics
        'fdc': classical['fdc'],
        'autocorrelation_length': classical['autocorrelation_length'],
        'information_content_H': classical['information_content_H'],
        'partial_information_content_M': classical['partial_information_content_M'],
        # Algorithm performance
        **algo_results,
        # Timing
        'orc_time_s': round(orc_time, 3),
        'classical_time_s': round(classical_time, 3),
        'algo_time_s': round(algo_time, 3),
    }


# ---------------------------------------------------------------------------
# Algorithm benchmarks (embedded for simplicity)
# ---------------------------------------------------------------------------

def _run_algorithms(
    fitness: np.ndarray,
    neighbor_fn,
    space_size: int,
    seed: int,
    n_trials: int = 50,
    budget_frac: float = 0.05,
) -> dict:
    """Run search algorithms and collect performance metrics."""
    budget = max(100, int(space_size * budget_frac))
    global_best = float(fitness.min())
    results = {}

    for algo_name, algo_fn in [
        ('RS', _random_search),
        ('HC', _hill_climbing),
        ('EA', _one_plus_one_ea),
    ]:
        best_fits = []
        success = 0
        for trial in range(n_trials):
            rng = np.random.RandomState(seed * 1000 + trial)
            best_f = algo_fn(fitness, neighbor_fn, space_size, budget, rng)
            best_fits.append(best_f)
            if abs(best_f - global_best) < 1e-10:
                success += 1

        arr = np.array(best_fits)
        results[f'algo_{algo_name}_mean'] = float(arr.mean())
        results[f'algo_{algo_name}_std'] = float(arr.std())
        results[f'algo_{algo_name}_success_rate'] = success / n_trials

    return results


def _random_search(fitness, neighbor_fn, space_size, budget, rng):
    best = np.inf
    for _ in range(budget):
        idx = rng.randint(0, space_size)
        if fitness[idx] < best:
            best = fitness[idx]
    return best


def _hill_climbing(fitness, neighbor_fn, space_size, budget, rng):
    best_overall = np.inf
    evals = 0
    while evals < budget:
        current = rng.randint(0, space_size)
        evals += 1
        while evals < budget:
            nbrs = neighbor_fn(current)
            improved = False
            for nbr in nbrs:
                evals += 1
                if evals > budget:
                    break
                if fitness[nbr] < fitness[current]:
                    current = nbr
                    improved = True
                    break
            if not improved:
                break
        if fitness[current] < best_overall:
            best_overall = fitness[current]
    return best_overall


def _one_plus_one_ea(fitness, neighbor_fn, space_size, budget, rng):
    n_bits = 0
    test = space_size
    while test > 1:
        n_bits += 1
        test >>= 1

    current = rng.randint(0, space_size)
    current_f = fitness[current]
    best_f = current_f

    for _ in range(budget - 1):
        pos = rng.randint(0, n_bits)
        child = current ^ (1 << pos)
        child_f = fitness[child]
        if child_f <= current_f:
            current = child
            current_f = child_f
        if current_f < best_f:
            best_f = current_f

    return best_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Discrete landscape analysis')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/landscape_discrete.json')
    parser.add_argument('--n20', action='store_true',
                        help='Include N=20 NK experiments')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []

    # NK landscapes: N=16, K in {0, 2, 4, 6, 8, 12, 15}, adjacent model
    for K in [0, 2, 4, 6, 8, 12, 15]:
        for seed in range(args.n_instances):
            tasks.append({
                'type': 'NK', 'config': {'N': 16, 'K': K, 'model': 'adjacent'},
                'seed': seed, 'gamma': args.gamma,
            })

    # NK landscapes: N=16, K in {2, 6, 12}, random model
    for K in [2, 6, 12]:
        for seed in range(args.n_instances):
            tasks.append({
                'type': 'NK', 'config': {'N': 16, 'K': K, 'model': 'random'},
                'seed': seed, 'gamma': args.gamma,
            })

    # W-model: n=16, nu in {1, 3, 4, 6, 8, 16}
    for nu in [1, 3, 4, 6, 8, 16]:
        for seed in range(args.n_instances):
            tasks.append({
                'type': 'WModel', 'config': {'n': 16, 'nu': nu},
                'seed': seed, 'gamma': args.gamma,
            })

    # N=20 NK landscapes (larger scale validation)
    if args.n20:
        n20_instances = min(args.n_instances, 10)
        for K in [0, 4, 8, 15, 19]:
            for seed in range(n20_instances):
                tasks.append({
                    'type': 'NK', 'config': {'N': 20, 'K': K, 'model': 'adjacent'},
                    'seed': seed, 'gamma': args.gamma,
                })

    total = len(tasks)
    workers = min(args.workers, total)

    print(f"Discrete Landscape Analysis (PPSN 2026)")
    print(f"  Tasks: {total}")
    print(f"  Workers: {workers}")
    print(f"  ORC gamma: {args.gamma}")
    print(f"  Output: {out_path}")
    print(flush=True)

    results = []
    completed = 0
    t_start = time.perf_counter()

    if workers <= 1:
        for task in tasks:
            row = _analyze_instance(task)
            results.append(row)
            completed += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / completed * (total - completed)
            print(f"  [{completed}/{total}] {row['label']} seed={row['seed']} "
                  f"optima={row['n_local_optima']} neg_orc={row['frac_negative_orc']:.2f} "
                  f"fdc={row['fdc']:.3f}  ETA {eta:.0f}s", flush=True)
    else:
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_analyze_instance, tasks):
                results.append(row)
                completed += 1
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                if completed % 10 == 0 or completed == total:
                    print(f"  [{completed}/{total}] "
                          f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer)
                  else float(o) if isinstance(o, np.floating) else None)

    print(f"\nSaved {len(results)} results to {out_path}")
    _print_summary(results)


def _print_summary(results):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    print(f"\n{'='*120}")
    print(f"{'Config':<35} {'#Opt':>6} {'%NegORC':>8} {'%ORC':>7} {'%Rand':>7} {'%Worst':>7} "
          f"{'FDC':>7} {'ACL':>5} {'HC_sr':>7} {'EA_sr':>7}")
    print(f"{'='*120}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        n_opt = np.mean([r['n_local_optima'] for r in rows])
        neg = np.mean([r['frac_negative_orc'] for r in rows])
        better = np.mean([r['frac_leads_to_better'] for r in rows])
        rand_better = np.mean([r.get('frac_random_leads_to_better', 0) for r in rows])
        worst_better = np.mean([r.get('frac_worst_orc_leads_to_better', 0) for r in rows])
        fdc = np.mean([r['fdc'] for r in rows])
        acl = np.mean([r['autocorrelation_length'] for r in rows])
        hc_sr = np.mean([r['algo_HC_success_rate'] for r in rows])
        ea_sr = np.mean([r['algo_EA_success_rate'] for r in rows])
        print(f"{label:<35} {n_opt:>6.0f} {100*neg:>7.1f}% {100*better:>6.1f}% "
              f"{100*rand_better:>6.1f}% {100*worst_better:>6.1f}% "
              f"{fdc:>7.3f} {acl:>5.1f} {100*hc_sr:>6.1f}% {100*ea_sr:>6.1f}%")


if __name__ == '__main__':
    freeze_support()
    main()

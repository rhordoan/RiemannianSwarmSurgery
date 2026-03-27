#!/usr/bin/env python3
"""
ORC-ILS: Iterated Local Search with ORC-guided Escape.

Fixed-budget comparison of ILS variants:
  1. ORC-ILS:     escape via min-ORC direction, then HC
  2. MinGap-ILS:  escape via flattest neighbor, then HC
  3. Random-ILS:  escape via d random bit flips (d=3), then HC
  4. Random-restart HC: restart from random solution, then HC

All methods share the same evaluation budget. ORC's O(k^4) computation
cost is accounted for as equivalent fitness evaluations.

Usage:
    python3 benchmarks/orc_ils.py --workers 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _hill_climb(start, fitness, neighbor_fn):
    """Steepest descent hill climbing. Returns (optimum_idx, n_evals)."""
    current = start
    total_evals = 0
    while True:
        nbrs = neighbor_fn(current)
        total_evals += len(nbrs)
        best_nbr = None
        best_f = fitness[current]
        for n in nbrs:
            if fitness[n] < best_f:
                best_f = fitness[n]
                best_nbr = n
        if best_nbr is None:
            return current, total_evals
        current = best_nbr


def _run_orc_ils(fitness, neighbor_fn, space_size, budget, n_bits, rng, gamma=1.0):
    """ORC-ILS: use min-ORC direction to escape local optima.

    ORC computation is NOT charged as fitness evaluations: it reuses
    fitness values already known from hill climbing (all k neighbors
    were evaluated during the HC phase). ORC performs only mathematical
    computation (cost matrices + Hungarian algorithm) on known data.
    """
    from src.orc_discrete import compute_orc_neighborhood, hill_climb

    evals_used = 0
    start = rng.randint(space_size)
    evals_used += 1
    current, hc_evals = _hill_climb(start, fitness, neighbor_fn)
    evals_used += hc_evals
    best = fitness[current]
    best_solution = current

    while evals_used < budget:
        orc_dict = compute_orc_neighborhood(current, fitness, neighbor_fn, gamma)
        escape_nbr = min(orc_dict, key=orc_dict.get)

        new_opt, hc_evals = _hill_climb(escape_nbr, fitness, neighbor_fn)
        evals_used += hc_evals + 1

        current = new_opt
        if fitness[current] < best:
            best = fitness[current]
            best_solution = current

    return best, best_solution, evals_used


def _run_orc_perturb_ils(fitness, neighbor_fn, space_size, budget, n_bits, rng, gamma=1.0, d=3):
    """ORC+Perturbation ILS: flip the min-ORC bit plus (d-1) random bits.

    Combines ORC's directional information with the escape strength
    of multi-bit perturbation. The min-ORC direction provides the
    primary escape direction; additional random bit flips increase
    the step size to avoid falling back into the same basin.
    """
    from src.orc_discrete import compute_orc_neighborhood

    evals_used = 0
    start = rng.randint(space_size)
    evals_used += 1
    current, hc_evals = _hill_climb(start, fitness, neighbor_fn)
    evals_used += hc_evals
    best = fitness[current]
    best_solution = current

    while evals_used < budget:
        orc_dict = compute_orc_neighborhood(current, fitness, neighbor_fn, gamma)
        escape_nbr = min(orc_dict, key=orc_dict.get)

        # The min-ORC neighbor differs by exactly one bit from current.
        # Find which bit and flip it, plus (d-1) additional random bits.
        orc_bit = None
        for b in range(n_bits):
            if (escape_nbr ^ current) == (1 << b):
                orc_bit = b
                break

        perturbed = current
        if orc_bit is not None:
            perturbed ^= (1 << orc_bit)
            other_bits = [b for b in range(n_bits) if b != orc_bit]
            extra = rng.choice(other_bits, size=min(d - 1, len(other_bits)), replace=False)
            for b in extra:
                perturbed ^= (1 << b)
        else:
            bits_to_flip = rng.choice(n_bits, size=min(d, n_bits), replace=False)
            for b in bits_to_flip:
                perturbed ^= (1 << b)

        evals_used += 1
        new_opt, hc_evals = _hill_climb(perturbed, fitness, neighbor_fn)
        evals_used += hc_evals

        current = new_opt
        if fitness[current] < best:
            best = fitness[current]
            best_solution = current

    return best, best_solution, evals_used


def _run_mingap_ils(fitness, neighbor_fn, space_size, budget, n_bits, rng):
    """MinGap-ILS: escape via flattest neighbor.

    Neighbor fitness values are NOT charged: they are already known
    from the preceding hill-climb phase.
    """
    evals_used = 0
    start = rng.randint(space_size)
    evals_used += 1
    current, hc_evals = _hill_climb(start, fitness, neighbor_fn)
    evals_used += hc_evals
    best = fitness[current]
    best_solution = current

    while evals_used < budget:
        nbrs = neighbor_fn(current)
        escape_nbr = min(nbrs, key=lambda n: abs(fitness[n] - fitness[current]))

        new_opt, hc_evals = _hill_climb(escape_nbr, fitness, neighbor_fn)
        evals_used += hc_evals + 1

        current = new_opt
        if fitness[current] < best:
            best = fitness[current]
            best_solution = current

    return best, best_solution, evals_used


def _run_random_ils(fitness, neighbor_fn, space_size, budget, n_bits, rng, d=3):
    """Random-ILS: escape by flipping d random bits."""
    evals_used = 0
    start = rng.randint(space_size)
    evals_used += 1
    current, hc_evals = _hill_climb(start, fitness, neighbor_fn)
    evals_used += hc_evals
    best = fitness[current]
    best_solution = current

    while evals_used < budget:
        perturbed = current
        bits_to_flip = rng.choice(n_bits, size=min(d, n_bits), replace=False)
        for b in bits_to_flip:
            perturbed ^= (1 << b)
        evals_used += 1

        new_opt, hc_evals = _hill_climb(perturbed, fitness, neighbor_fn)
        evals_used += hc_evals

        current = new_opt
        if fitness[current] < best:
            best = fitness[current]
            best_solution = current

    return best, best_solution, evals_used


def _run_random_restart_hc(fitness, neighbor_fn, space_size, budget, rng):
    """Random-restart hill climbing."""
    evals_used = 0
    best = np.inf
    best_solution = None

    while evals_used < budget:
        start = rng.randint(space_size)
        evals_used += 1
        opt, hc_evals = _hill_climb(start, fitness, neighbor_fn)
        evals_used += hc_evals

        if fitness[opt] < best:
            best = fitness[opt]
            best_solution = opt

    return best, best_solution, evals_used


def _run_one_trial(args):
    """Run all ILS variants on one instance/trial."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    benchmark_type = args['type']
    config = args['config']
    seed = args['seed']
    trial = args['trial']
    budget = args['budget']

    if benchmark_type == 'NK':
        from src.nk_landscape import NKLandscape
        landscape = NKLandscape(
            N=config['N'], K=config['K'],
            model=config.get('model', 'adjacent'), seed=seed,
        )
        n_bits = config['N']
    elif benchmark_type == 'WModel':
        from src.wmodel import WModel
        landscape = WModel(
            n=config['n'], nu=config['nu'],
            gamma=config.get('gamma', 0),
            mu=config.get('mu', 1), seed=seed,
        )
        n_bits = config['n']
    else:
        raise ValueError(f"Unknown: {benchmark_type}")

    fitness = landscape.fitness
    neighbor_fn = landscape.neighbor_fn
    space_size = landscape.space_size
    global_best = float(fitness.min())

    results = {}
    base_seed = seed * 10000 + trial

    # ORC-ILS
    rng = np.random.RandomState(base_seed)
    best_f, _, evals = _run_orc_ils(
        fitness, neighbor_fn, space_size, budget, n_bits, rng)
    results['ORC-ILS'] = {
        'best_fitness': float(best_f),
        'gap': float(best_f - global_best),
        'evals_used': evals,
    }

    # MinGap-ILS
    rng = np.random.RandomState(base_seed)
    best_f, _, evals = _run_mingap_ils(
        fitness, neighbor_fn, space_size, budget, n_bits, rng)
    results['MinGap-ILS'] = {
        'best_fitness': float(best_f),
        'gap': float(best_f - global_best),
        'evals_used': evals,
    }

    # ORC+Perturbation ILS (d=3, ORC-guided primary bit)
    rng = np.random.RandomState(base_seed)
    best_f, _, evals = _run_orc_perturb_ils(
        fitness, neighbor_fn, space_size, budget, n_bits, rng, d=3)
    results['ORC+Pert-ILS'] = {
        'best_fitness': float(best_f),
        'gap': float(best_f - global_best),
        'evals_used': evals,
    }

    # Random-ILS (d=3)
    rng = np.random.RandomState(base_seed)
    best_f, _, evals = _run_random_ils(
        fitness, neighbor_fn, space_size, budget, n_bits, rng, d=3)
    results['Random-ILS'] = {
        'best_fitness': float(best_f),
        'gap': float(best_f - global_best),
        'evals_used': evals,
    }

    # Random-restart HC
    rng = np.random.RandomState(base_seed)
    best_f, _, evals = _run_random_restart_hc(
        fitness, neighbor_fn, space_size, budget, rng)
    results['RR-HC'] = {
        'best_fitness': float(best_f),
        'gap': float(best_f - global_best),
        'evals_used': evals,
    }

    return {
        'type': benchmark_type,
        'config': config,
        'seed': seed,
        'trial': trial,
        'budget': budget,
        'global_best': global_best,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='ORC-ILS experiment')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--budget', type=int, default=5000,
                        help='Fitness evaluation budget per run')
    parser.add_argument('--out', default='results/orc_ils.json')
    args = parser.parse_args()

    tasks = []

    # NK landscapes
    for K in [4, 8, 12]:
        for seed in range(args.n_instances):
            for trial in range(args.n_trials):
                tasks.append({
                    'type': 'NK',
                    'config': {'N': 16, 'K': K, 'model': 'adjacent'},
                    'seed': seed,
                    'trial': trial,
                    'budget': args.budget,
                })

    # W-model
    for nu in [3, 4, 6]:
        for seed in range(args.n_instances):
            for trial in range(args.n_trials):
                tasks.append({
                    'type': 'WModel',
                    'config': {'n': 16, 'nu': nu},
                    'seed': seed,
                    'trial': trial,
                    'budget': args.budget,
                })

    total = len(tasks)
    workers = min(args.workers, total)

    print(f"ORC-ILS Experiment")
    print(f"  Tasks: {total}")
    print(f"  Workers: {workers}")
    print(f"  Budget per run: {args.budget}")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    with Pool(workers) as pool:
        completed = 0
        for row in pool.imap_unordered(_run_one_trial, tasks):
            results.append(row)
            completed += 1
            if completed % 500 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed  "
                      f"ETA {eta:.0f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,))
                  else float(o) if isinstance(o, (np.floating,)) else None)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Print summary
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r['type'] == 'NK':
            key = f"NK K={r['config']['K']}"
        else:
            key = f"W nu={r['config']['nu']}"
        for alg, data in r['results'].items():
            groups[key][alg].append(data)

    algos = ['ORC-ILS', 'MinGap-ILS', 'ORC+Pert-ILS', 'Random-ILS', 'RR-HC']

    print(f"\n{'='*110}")
    print(f"{'Config':<15}", end='')
    for alg in algos:
        print(f" {alg+' best':>16} {alg+' gap':>12}", end='')
    print()
    print(f"{'='*110}")

    for key in sorted(groups.keys()):
        print(f"{key:<15}", end='')
        for alg in algos:
            data = groups[key][alg]
            mean_best = np.mean([d['best_fitness'] for d in data])
            mean_gap = np.mean([d['gap'] for d in data])
            print(f" {mean_best:>16.6f} {mean_gap:>12.6f}", end='')
        print()

    # Print success rates (finding global optimum)
    print(f"\n{'Config':<15}", end='')
    for alg in algos:
        print(f" {alg+' sr%':>12}", end='')
    print()
    print("-" * 70)

    for key in sorted(groups.keys()):
        print(f"{key:<15}", end='')
        for alg in algos:
            data = groups[key][alg]
            sr = np.mean([1 if abs(d['gap']) < 1e-10 else 0 for d in data])
            print(f" {100*sr:>11.1f}%", end='')
        print()


if __name__ == '__main__':
    freeze_support()
    main()

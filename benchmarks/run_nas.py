#!/usr/bin/env python3
"""
ORC-NAS Benchmark Runner
========================
Compares ORC-guided NAS search algorithms against baselines on the
NAS-Bench-201 / NATS-Bench topology search space.

Algorithms:
  - Random Search (sanity baseline)
  - Regularized Evolution (RE, standard NAS baseline)
  - Local Search (LS, hill climbing with restarts)
  - SMAC (RF surrogate-guided search)
  - ORC-RE (RE + ORC saddle-guided mutation)
  - ORC-Tabu (Tabu + ORC saddle escape)

Usage
-----
  # Quick test (10 trials, synthetic landscape)
  python3 benchmarks/run_nas.py --trials 10

  # Full benchmark (500 trials, 160 cores)
  python3 benchmarks/run_nas.py --trials 500 --workers 160

  # With real NATS-Bench data
  python3 benchmarks/run_nas.py --data data/NATS-tss-v1_0-3ffb9.pickle.pbz2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    'Random': 'random_search',
    'RE': 'regularized_evolution',
    'LS': 'local_search',
    'Tabu': 'tabu_search',
    'ORC-RE': 'orc_regularized_evolution',
    'ORC-Tabu': 'orc_tabu_search',
    'SMAC': 'smac_search',
}

MAIN_ALGORITHMS = ['Random', 'RE', 'LS', 'Tabu', 'SMAC', 'ORC-RE', 'ORC-Tabu']


# ---------------------------------------------------------------------------
# Single-trial worker
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> dict:
    algo_name, budget, trial_seed, data_path, dataset = args

    import sys, os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.orc_nas import NASBench201
    from src import search_nas

    benchmark = NASBench201(dataset=dataset, data_path=data_path)
    func_name = ALGORITHMS[algo_name]
    func = getattr(search_nas, func_name)

    t0 = time.perf_counter()
    result = func(benchmark, budget=budget, seed=trial_seed)
    elapsed = time.perf_counter() - t0

    return {
        'algorithm': algo_name,
        'dataset': dataset,
        'budget': budget,
        'trial': trial_seed,
        'best_accuracy': round(result.best_accuracy, 4),
        'best_fitness': round(result.best_fitness, 4),
        'n_queries': result.n_queries,
        'elapsed_s': round(elapsed, 3),
        'is_real_data': benchmark.is_real,
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

FIELDS = [
    'algorithm', 'dataset', 'budget', 'trial',
    'best_accuracy', 'best_fitness', 'n_queries', 'elapsed_s', 'is_real_data',
]


def _load_done(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['algorithm'], row['dataset'],
                       int(row['budget']), int(row['trial'])))
    return done


def _append_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(csv_path: Path):
    from collections import defaultdict
    from scipy.stats import wilcoxon

    data = defaultdict(list)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            k = (row['algorithm'], row['dataset'], int(row['budget']))
            data[k].append(float(row['best_accuracy']))

    datasets = sorted(set(d for _, d, _ in data.keys()))
    budgets = sorted(set(b for _, _, b in data.keys()))
    algos = sorted(set(a for a, _, _ in data.keys()))

    for dataset in datasets:
        for budget in budgets:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset}  |  Budget: {budget} queries")
            print(f"{'='*80}")

            hdr = f"{'Algorithm':<12} | {'Mean Acc':>10} | {'Std':>8} | {'Median':>10} | {'Best':>8} | {'n':>4}"
            print(hdr)
            print('-' * len(hdr))

            ref_vals = None
            for algo in algos:
                vals = np.array(data.get((algo, dataset, budget), []))
                if len(vals) == 0:
                    continue
                print(f"{algo:<12} | {vals.mean():>10.4f} | {vals.std():>8.4f} | "
                      f"{np.median(vals):>10.4f} | {vals.max():>8.4f} | {len(vals):>4}")

                if algo == 'RE':
                    ref_vals = vals

            # Wilcoxon tests vs RE
            if ref_vals is not None and len(ref_vals) >= 5:
                print(f"\nWilcoxon signed-rank test vs RE:")
                for algo in algos:
                    if algo == 'RE':
                        continue
                    vals = np.array(data.get((algo, dataset, budget), []))
                    if len(vals) < 5:
                        continue
                    n = min(len(ref_vals), len(vals))
                    a, b = ref_vals[:n], vals[:n]
                    try:
                        if np.allclose(a, b):
                            p = 1.0
                        else:
                            p = wilcoxon(a, b, alternative='two-sided').pvalue
                    except Exception:
                        p = float('nan')
                    better = 'BETTER' if b.mean() > a.mean() and p < 0.05 else (
                        'WORSE' if b.mean() < a.mean() and p < 0.05 else 'tie')
                    print(f"  {algo:<12} vs RE: p={p:.6f}  ({better})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ORC-NAS Benchmark on NAS-Bench-201')
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--budget', type=int, default=200,
                        help='Query budget per trial (default 200)')
    parser.add_argument('--budgets', type=int, nargs='+', default=None,
                        help='Multiple budgets to test')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to NATS-Bench TSS database')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'])
    parser.add_argument('--algorithms', nargs='+', default=None)
    parser.add_argument('--workers', type=int,
                        default=max(1, cpu_count() - 2))
    parser.add_argument('--out', default='results/orc_nas_benchmark.csv')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    budgets = args.budgets if args.budgets else [args.budget]
    algos = args.algorithms if args.algorithms else MAIN_ALGORITHMS
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done(out_path) if args.resume else set()

    tasks = []
    for budget in budgets:
        for algo in algos:
            for trial in range(args.trials):
                if (algo, args.dataset, budget, trial) not in done:
                    tasks.append((algo, budget, trial, args.data, args.dataset))

    total = len(tasks)
    if not total:
        print('All runs complete.')
        _print_summary(out_path)
        return

    workers = min(args.workers, total)
    print(
        f"\nORC-NAS Benchmark\n"
        f"  Algorithms : {algos}\n"
        f"  Budgets    : {budgets}\n"
        f"  Dataset    : {args.dataset}\n"
        f"  Trials     : {args.trials}  |  Tasks: {total}\n"
        f"  Workers    : {workers}\n"
        f"  Data       : {'REAL' if args.data else 'SYNTHETIC'}\n"
        f"  Output     : {out_path}\n",
        flush=True,
    )

    completed = 0
    t_start = time.perf_counter()

    def _handle(row: dict):
        nonlocal completed
        _append_row(out_path, row)
        completed += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / completed * (total - completed)
        pct = 100.0 * completed / total
        print(
            f"  [{completed:>{len(str(total))}}/{total}] "
            f"{row['algorithm']:<12} budget={row['budget']} "
            f"acc={row['best_accuracy']:.4f}  "
            f"{pct:.1f}%  ETA {eta:.0f}s  ({row['elapsed_s']:.3f}s)",
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            _handle(_run_one(task))
    else:
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_run_one, tasks):
                _handle(row)

    print('\n\nAll runs complete.\n')
    _print_summary(out_path)


if __name__ == '__main__':
    freeze_support()
    main()

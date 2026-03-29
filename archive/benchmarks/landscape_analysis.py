#!/usr/bin/env python3
"""
Landscape Analysis for ORC-Guided Neural Architecture Search
============================================================

NeurIPS paper: Analyzes the NAS-Bench-201 / NATS-Bench topology search space
to understand whether ORC (Ollivier-Ricci Curvature) at local optima points
toward better basins.

Analysis outputs:
  a) Total number of local optima
  b) Distribution of accuracies at local optima
  c) ORC values for all 24 neighbors at each local optimum
  d) Fraction of local optima with at least one negative ORC direction
  e) For those with negative ORC: does following the negative direction
     (greedy hill climb from that neighbor) reach a BETTER local optimum?
  f) Basin sizes (how many architectures hill-climb to each local optimum)

Usage
-----
  python3 benchmarks/landscape_analysis.py
  python3 benchmarks/landscape_analysis.py --data data/NATS-tss-v1_0-3ffb9-simple
  python3 benchmarks/landscape_analysis.py --dataset cifar100 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.orc_nas import (
    NASBench201,
    SPACE_SIZE,
    index_to_tuple,
    tuple_to_index,
    get_neighbor_indices,
    compute_orc_neighborhood,
    find_saddle_direction,
)


# ---------------------------------------------------------------------------
# Hill-climbing (deterministic: best neighbor, ties broken by index)
# ---------------------------------------------------------------------------

def hill_climb_to_local_optimum(start_idx: int, fitness: np.ndarray) -> int:
    """
    Greedy hill-climb from start_idx to a local optimum.
    Always move to the best neighbor (lowest fitness); break ties by index.
    Returns the index of the local optimum reached.
    """
    current = start_idx
    while True:
        nbrs = get_neighbor_indices(current)
        current_f = fitness[current]
        improving = [(n, fitness[n]) for n in nbrs if fitness[n] < current_f]
        if not improving:
            return current
        best_nbr = min(improving, key=lambda x: (x[1], x[0]))[0]
        current = best_nbr


def is_local_optimum(idx: int, fitness: np.ndarray) -> bool:
    """True iff no Hamming-1 neighbor has lower fitness (higher accuracy)."""
    f = fitness[idx]
    for nbr in get_neighbor_indices(idx):
        if fitness[nbr] < f:
            return False
    return True


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing (use initializer to avoid pickling arrays)
# ---------------------------------------------------------------------------

_WORKER_FITNESS = None
_WORKER_ACCURACY = None


def _init_workers(fitness: np.ndarray, accuracy: np.ndarray):
    global _WORKER_FITNESS, _WORKER_ACCURACY
    _WORKER_FITNESS = np.asarray(fitness, dtype=np.float64)
    _WORKER_ACCURACY = np.asarray(accuracy, dtype=np.float64)


def _find_local_optima_chunk(args: tuple) -> list:
    """Worker: find local optima in a chunk of indices."""
    start, end = args
    fitness = _WORKER_FITNESS
    optima = []
    for idx in range(start, end):
        if is_local_optimum(idx, fitness):
            optima.append(idx)
    return optima


def _basin_worker(indices: list) -> list:
    """Worker: for each start idx in chunk, return (start_idx, local_opt_idx)."""
    fitness = _WORKER_FITNESS
    results = []
    for idx in indices:
        opt = hill_climb_to_local_optimum(idx, fitness)
        results.append((idx, opt))
    return results


def _orc_analysis_worker(args: tuple) -> dict:
    """
    Worker: for a local optimum, compute ORC neighborhood, find most negative
    ORC neighbor, hill-climb from it, check if reaches better optimum.
    """
    opt_idx, gamma = args
    fitness = _WORKER_FITNESS
    accuracy = _WORKER_ACCURACY
    opt_fit = float(fitness[opt_idx])
    opt_acc = float(accuracy[opt_idx])

    orc_dict = compute_orc_neighborhood(opt_idx, fitness, gamma)
    orc_values = {int(k): float(v) for k, v in orc_dict.items()}

    min_orc_nbr = min(orc_dict, key=orc_dict.get)
    min_orc = orc_dict[min_orc_nbr]

    has_negative_orc = min_orc < 0
    leads_to_better = False
    dest_opt_idx = None
    dest_accuracy = None

    if has_negative_orc:
        dest_opt_idx = hill_climb_to_local_optimum(min_orc_nbr, fitness)
        dest_fit = float(fitness[dest_opt_idx])
        dest_accuracy = float(accuracy[dest_opt_idx])
        leads_to_better = dest_fit < opt_fit

    return {
        "opt_idx": opt_idx,
        "opt_fitness": opt_fit,
        "opt_accuracy": opt_acc,
        "orc_values": orc_values,
        "min_orc": float(min_orc),
        "min_orc_neighbor": int(min_orc_nbr),
        "has_negative_orc": has_negative_orc,
        "leads_to_better": leads_to_better,
        "dest_opt_idx": dest_opt_idx,
        "dest_accuracy": dest_accuracy,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    bench: NASBench201,
    workers: int = max(1, cpu_count() - 2),
    gamma: float = 1.0,
) -> dict:
    fitness = np.ascontiguousarray(bench.fitness, dtype=np.float64)
    accuracy = np.ascontiguousarray(bench.accuracy, dtype=np.float64)
    initargs = (fitness, accuracy)

    # 1. Find all local optima
    print("Finding local optima...", flush=True)
    chunk_size = (SPACE_SIZE + workers - 1) // workers
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, SPACE_SIZE))
        for i in range(workers)
    ]
    if workers <= 1:
        _init_workers(fitness, accuracy)
        local_optima = []
        for start, end in chunks:
            local_optima.extend(_find_local_optima_chunk((start, end)))
    else:
        with Pool(workers, initializer=_init_workers, initargs=initargs) as pool:
            results = pool.map(_find_local_optima_chunk, chunks)
        local_optima = [idx for chunk_result in results for idx in chunk_result]

    local_optima = sorted(local_optima)
    n_optima = len(local_optima)
    print(f"  Found {n_optima} local optima", flush=True)

    # 2. Basin sizes: for each architecture, which local optimum does it reach?
    print("Computing basin sizes...", flush=True)
    all_indices = list(range(SPACE_SIZE))
    chunk_len = (len(all_indices) + workers - 1) // workers
    basin_chunks = [
        all_indices[i * chunk_len : (i + 1) * chunk_len]
        for i in range(workers)
    ]
    if workers <= 1:
        basin_results = []
        for inds in basin_chunks:
            basin_results.extend(_basin_worker(inds))
    else:
        with Pool(workers, initializer=_init_workers, initargs=initargs) as pool:
            basin_results_list = pool.map(_basin_worker, basin_chunks)
        basin_results = [r for chunk in basin_results_list for r in chunk]

    basin_counts = {}
    for _, opt_idx in basin_results:
        basin_counts[opt_idx] = basin_counts.get(opt_idx, 0) + 1

    basin_sizes = [basin_counts.get(opt, 0) for opt in local_optima]
    opt_to_basin = dict(zip(local_optima, basin_sizes))

    # 3. ORC analysis at each local optimum
    print("Computing ORC neighborhoods and saddle analysis...", flush=True)
    orc_tasks = [(opt_idx, gamma) for opt_idx in local_optima]
    if workers <= 1:
        orc_results = [_orc_analysis_worker(t) for t in orc_tasks]
    else:
        with Pool(workers, initializer=_init_workers, initargs=initargs) as pool:
            orc_results = pool.map(_orc_analysis_worker, orc_tasks)

    # 4. Aggregate statistics
    opt_accuracies = [r["opt_accuracy"] for r in orc_results]
    n_with_negative_orc = sum(1 for r in orc_results if r["has_negative_orc"])
    n_leads_to_better = sum(
        1 for r in orc_results if r["has_negative_orc"] and r["leads_to_better"]
    )

    return {
        "dataset": bench.dataset,
        "data_path": getattr(bench, "_data_path", None),
        "is_real_data": bench.is_real,
        "space_size": SPACE_SIZE,
        "gamma": gamma,
        "n_local_optima": n_optima,
        "local_optima_indices": local_optima,
        "local_optima_accuracies": opt_accuracies,
        "local_optima_fitnesses": [r["opt_fitness"] for r in orc_results],
        "basin_sizes": opt_to_basin,
        "basin_size_list": basin_sizes,
        "orc_analyses": orc_results,
        "frac_with_negative_orc": n_with_negative_orc / n_optima if n_optima else 0,
        "n_with_negative_orc": n_with_negative_orc,
        "frac_leads_to_better": (
            n_leads_to_better / n_with_negative_orc if n_with_negative_orc else 0
        ),
        "n_leads_to_better": n_leads_to_better,
    }


def print_summary(results: dict):
    """Print human-readable summary for the paper."""
    print("\n" + "=" * 72)
    print("LANDSCAPE ANALYSIS — ORC-Guided NAS")
    print("=" * 72)
    print(f"Dataset:        {results['dataset']}")
    print(f"Data:           {'REAL (NATS-Bench)' if results['is_real_data'] else 'SYNTHETIC'}")
    print(f"Space size:     {results['space_size']:,}")
    print(f"Gamma:          {results['gamma']}")
    print("-" * 72)
    print("LOCAL OPTIMA")
    print("-" * 72)
    n_opt = results["n_local_optima"]
    print(f"Total local optima:              {n_opt:,}")
    accs = np.array(results["local_optima_accuracies"])
    print(f"Accuracy at local optima:")
    print(f"  min:   {accs.min():.2f}%")
    print(f"  max:   {accs.max():.2f}%")
    print(f"  mean:  {accs.mean():.2f}%")
    print(f"  std:   {accs.std():.2f}%")
    print(f"  median:{np.median(accs):.2f}%")
    print("-" * 72)
    print("BASIN SIZES")
    print("-" * 72)
    sizes = np.array(results["basin_size_list"])
    print(f"Min basin size:    {sizes.min()}")
    print(f"Max basin size:    {sizes.max()}")
    print(f"Mean basin size:   {sizes.mean():.1f}")
    print("-" * 72)
    print("ORC SADDLE ANALYSIS")
    print("-" * 72)
    n_neg = results["n_with_negative_orc"]
    print(f"Local optima with ≥1 negative ORC direction:  {n_neg} / {n_opt}  ({100 * results['frac_with_negative_orc']:.1f}%)")
    n_better = results["n_leads_to_better"]
    if n_neg > 0:
        print(f"Of those: negative ORC leads to BETTER optimum: {n_better} / {n_neg}  ({100 * results['frac_leads_to_better']:.1f}%)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# JSON serialization (handle numpy types)
# ---------------------------------------------------------------------------

def _json_serialize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Landscape analysis for ORC-guided NAS (NeurIPS paper)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/NATS-tss-v1_0-3ffb9-simple",
        help="Path to NATS-Bench TSS database",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 2),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="ORC gamma (fitness weight in lifted distance)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/landscape_analysis.json",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading NASBench201: dataset={args.dataset}, data={args.data}", flush=True)
    bench = NASBench201(dataset=args.dataset, data_path=args.data)
    bench._data_path = args.data

    print(f"Running analysis (workers={args.workers})...", flush=True)
    results = run_analysis(bench, workers=args.workers, gamma=args.gamma)

    # Save full results (compact for JSON)
    out_data = {
        "dataset": results["dataset"],
        "is_real_data": results["is_real_data"],
        "space_size": results["space_size"],
        "gamma": results["gamma"],
        "n_local_optima": results["n_local_optima"],
        "local_optima_indices": results["local_optima_indices"],
        "local_optima_accuracies": results["local_optima_accuracies"],
        "local_optima_fitnesses": results["local_optima_fitnesses"],
        "basin_sizes": results["basin_sizes"],
        "frac_with_negative_orc": results["frac_with_negative_orc"],
        "n_with_negative_orc": results["n_with_negative_orc"],
        "frac_leads_to_better": results["frac_leads_to_better"],
        "n_leads_to_better": results["n_leads_to_better"],
        "orc_analyses": [
            {
                "opt_idx": r["opt_idx"],
                "opt_accuracy": r["opt_accuracy"],
                "orc_values": r["orc_values"],
                "min_orc": r["min_orc"],
                "min_orc_neighbor": r["min_orc_neighbor"],
                "has_negative_orc": r["has_negative_orc"],
                "leads_to_better": r["leads_to_better"],
                "dest_opt_idx": r["dest_opt_idx"],
                "dest_accuracy": r["dest_accuracy"],
            }
            for r in results["orc_analyses"]
        ],
    }

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=_json_serialize)
    print(f"Saved to {out_path}", flush=True)

    print_summary(results)


if __name__ == "__main__":
    freeze_support()
    main()

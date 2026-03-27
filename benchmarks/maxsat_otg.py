#!/usr/bin/env python3
"""
OTG Analysis on Random 3-SAT/MAX-SAT at the Satisfiability Phase Transition.

Generates random 3-SAT instances at varying clause-to-variable ratios (alpha),
builds the ORC Transition Graph, and analyzes whether OTG topology exhibits
a phase transition at the satisfiability threshold (alpha ~4.27 for 3-SAT).

Usage:
    python3 benchmarks/maxsat_otg.py --workers 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def generate_random_3sat(n_vars: int, alpha: float, seed: int = 0):
    """
    Generate a random 3-SAT instance.

    Returns:
        clauses: list of tuples, each a 3-tuple of signed literals.
                 Positive = variable (1-indexed), negative = negated variable.
        n_clauses: number of clauses
    """
    rng = np.random.RandomState(seed)
    n_clauses = int(round(alpha * n_vars))
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        variables = rng.choice(n_vars, size=3, replace=False) + 1  # 1-indexed
        # Random sign for each
        signs = rng.choice([-1, 1], size=3)
        clause = tuple(int(v * s) for v, s in zip(variables, signs))
        clauses.append(clause)
    return clauses, n_clauses


def build_maxsat_fitness(n_vars: int, clauses: list) -> np.ndarray:
    """
    Build fitness array for MAX-SAT: fitness[x] = number of unsatisfied clauses.
    Solution x is encoded as an integer where bit i = variable (i+1).
    Minimization: fitness 0 means all clauses satisfied.

    Vectorized over the full solution space for speed.
    """
    space_size = 2 ** n_vars
    all_solutions = np.arange(space_size, dtype=np.int32)

    fitness = np.zeros(space_size, dtype=np.float64)
    for clause in clauses:
        clause_sat = np.zeros(space_size, dtype=np.bool_)
        for lit in clause:
            var_idx = abs(lit) - 1
            bit_vals = (all_solutions >> var_idx) & 1
            if lit > 0:
                clause_sat |= (bit_vals == 1)
            else:
                clause_sat |= (bit_vals == 0)
        fitness += ~clause_sat

    return fitness


def bitflip_neighbor_fn(n_vars: int):
    """Return a neighbor function for bit-flip on n_vars-bit strings."""
    def neighbor_fn(x: int) -> list:
        return [x ^ (1 << i) for i in range(n_vars)]
    return neighbor_fn


def _analyze_maxsat_instance(args: dict) -> dict:
    """Analyze a single MAX-SAT instance: build OTG, compute all metrics."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.orc_discrete import (
        full_landscape_analysis,
        find_all_local_optima,
        hill_climb,
        compute_orc_neighborhood,
    )

    n_vars = args['n_vars']
    alpha = args['alpha']
    seed = args['seed']
    gamma = args.get('gamma', 1.0)

    clauses, n_clauses = generate_random_3sat(n_vars, alpha, seed)
    fitness = build_maxsat_fitness(n_vars, clauses)
    neighbor_fn = bitflip_neighbor_fn(n_vars)
    space_size = 2 ** n_vars

    global_min_fitness = float(fitness.min())
    is_satisfiable = (global_min_fitness == 0.0)
    n_global_optima = int(np.sum(fitness == global_min_fitness))

    t0 = time.perf_counter()

    orc_result = full_landscape_analysis(
        space_size, fitness, neighbor_fn, gamma,
        n_random_trials=30, seed=seed,
    )
    orc_time = time.perf_counter() - t0

    local_optima = orc_result['local_optima']
    basins = orc_result['basin_sizes']
    analyses = orc_result['orc_analyses']
    n_opt = len(local_optima)

    if n_opt == 0:
        return {
            'alpha': alpha, 'seed': seed, 'n_vars': n_vars,
            'n_clauses': n_clauses, 'n_local_optima': 0,
            'is_satisfiable': is_satisfiable, 'global_min_fitness': global_min_fitness,
            'orc_time_s': round(orc_time, 3), 'skip': True,
        }

    # Build OTG
    opt_set = set(local_optima)
    global_opt = int(np.argmin(fitness))

    otg_edges = {}
    for a in analyses:
        src = a['opt_idx']
        dst = a['dest_opt']
        if dst is None:
            dst = src
        otg_edges[src] = dst

    # Trace paths
    def trace_path(start):
        visited_order = []
        visited_set = set()
        current = start
        while current not in visited_set:
            visited_set.add(current)
            visited_order.append(current)
            nxt = otg_edges.get(current, current)
            if nxt == current:
                return visited_order, {current}
            current = nxt
        cycle_start_idx = visited_order.index(current)
        cycle = set(visited_order[cycle_start_idx:])
        return visited_order, cycle

    all_cycles = []
    seen_cycles = set()
    opt_paths = {}
    opt_terminal_cycle = {}
    for opt in local_optima:
        path, cycle = trace_path(opt)
        opt_paths[opt] = path
        opt_terminal_cycle[opt] = cycle
        cycle_key = frozenset(cycle)
        if cycle_key not in seen_cycles:
            seen_cycles.add(cycle_key)
            all_cycles.append(cycle)

    sinks = set()
    for cycle in all_cycles:
        sinks.update(cycle)

    n_sinks = len(sinks)
    n_attractors = len(all_cycles)

    # Multi-node cycles
    multi_node_cycles = [c for c in all_cycles if len(c) > 1]
    n_in_multi_cycle = sum(len(c) for c in multi_node_cycles)
    frac_in_multi_cycle = n_in_multi_cycle / n_opt if n_opt > 0 else 0.0

    # Global optimum reachability
    hops_to_global = []
    reaches_global_count = 0
    for opt in local_optima:
        path = opt_paths[opt]
        if global_opt in path:
            idx = path.index(global_opt)
            hops_to_global.append(idx)
            reaches_global_count += 1
        else:
            cycle = opt_terminal_cycle[opt]
            if global_opt in cycle:
                hops_to_global.append(len(path))
                reaches_global_count += 1

    frac_reach_global = reaches_global_count / n_opt if n_opt > 0 else 0.0

    # Attractor quality
    all_opt_fitnesses = np.array([fitness[o] for o in local_optima])
    sorted_fitnesses = np.sort(all_opt_fitnesses)

    terminal_ranks = []
    for opt in local_optima:
        cycle = opt_terminal_cycle[opt]
        best_in_cycle = min(cycle, key=lambda c: fitness[c])
        term_fit = float(fitness[best_in_cycle])
        rank = np.searchsorted(sorted_fitnesses, term_fit) / max(n_opt - 1, 1)
        terminal_ranks.append(float(rank))

    mean_terminal_rank = float(np.mean(terminal_ranks))
    frac_terminal_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in terminal_ranks]))
    frac_terminal_top10 = float(np.mean([1 if r <= 0.10 else 0 for r in terminal_ranks]))

    # Path lengths to terminal
    hops_to_terminal = []
    hops_to_terminal_top5 = []
    for i, opt in enumerate(local_optima):
        path = opt_paths[opt]
        length = 0
        for node in path:
            if node in sinks:
                break
            length += 1
        hops_to_terminal.append(length)
        if terminal_ranks[i] <= 0.05:
            hops_to_terminal_top5.append(length)

    path_stats = {
        'mean': float(np.mean(hops_to_terminal)),
        'median': float(np.median(hops_to_terminal)),
        'p90': float(np.percentile(hops_to_terminal, 90)),
        'max': int(np.max(hops_to_terminal)),
    }
    if hops_to_terminal_top5:
        path_top5_stats = {
            'mean': float(np.mean(hops_to_terminal_top5)),
            'median': float(np.median(hops_to_terminal_top5)),
            'p90': float(np.percentile(hops_to_terminal_top5, 90)),
            'max': int(np.max(hops_to_terminal_top5)),
            'count': len(hops_to_terminal_top5),
        }
    else:
        path_top5_stats = {'mean': float('nan'), 'median': float('nan'),
                           'p90': float('nan'), 'max': 0, 'count': 0}

    # Compression ratio
    compression = n_attractors / n_opt if n_opt > 0 else 1.0

    # DAG depth (via networkx condensation)
    import networkx as nx
    G = nx.DiGraph()
    for src, dst in otg_edges.items():
        if src != dst:
            G.add_edge(src, dst)
        else:
            G.add_node(src)

    sccs = list(nx.strongly_connected_components(G))
    cond = nx.condensation(G)
    if len(cond) > 0:
        dag_depth = nx.dag_longest_path_length(cond) if len(cond) > 1 else 0
    else:
        dag_depth = 0

    # LON-d1 comparison (mode destination from 1-hop hill-climb)
    lon_d1_edges = {}
    for a in analyses:
        src = a['opt_idx']
        nbrs = list(a['orc_values'].keys())
        dest_counts = defaultdict(int)
        for nbr in nbrs:
            dest = hill_climb(nbr, fitness, neighbor_fn)
            dest_counts[dest] += 1
        mode_dest = max(dest_counts, key=dest_counts.get)
        lon_d1_edges[src] = mode_dest

    # LON-d1 attractor quality
    def trace_functional(edges, start, max_steps=1000):
        visited = set()
        current = start
        for _ in range(max_steps):
            if current in visited:
                return current
            visited.add(current)
            nxt = edges.get(current, current)
            if nxt == current:
                return current
            current = nxt
        return current

    d1_terminal_ranks = []
    for opt in local_optima:
        term = trace_functional(lon_d1_edges, opt)
        term_fit = float(fitness[term])
        rank = np.searchsorted(sorted_fitnesses, term_fit) / max(n_opt - 1, 1)
        d1_terminal_ranks.append(float(rank))

    d1_mean_rank = float(np.mean(d1_terminal_ranks))
    d1_frac_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in d1_terminal_ranks]))

    # LON-d1 compression
    d1_terminals = set()
    for opt in local_optima:
        d1_terminals.add(trace_functional(lon_d1_edges, opt))
    d1_compression = len(d1_terminals) / n_opt if n_opt > 0 else 1.0

    # MinGap comparison
    mingap_edges = {}
    for a in analyses:
        src = a['opt_idx']
        dst = a.get('min_gap_dest', src)
        if dst is None:
            dst = src
        mingap_edges[src] = dst

    mg_terminal_ranks = []
    for opt in local_optima:
        term = trace_functional(mingap_edges, opt)
        term_fit = float(fitness[term])
        rank = np.searchsorted(sorted_fitnesses, term_fit) / max(n_opt - 1, 1)
        mg_terminal_ranks.append(float(rank))

    mg_mean_rank = float(np.mean(mg_terminal_ranks))
    mg_frac_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in mg_terminal_ranks]))

    # Does the OTG distinguish satisfiable instances?
    # Terminal attractor fitness (best attractor's unsat count)
    best_attractor_fitness = min(
        float(fitness[min(c, key=lambda x: fitness[x])]) for c in all_cycles
    )

    # Mean ORC across all optima
    mean_orc_values = []
    for a in analyses:
        orc_vals = list(a['orc_values'].values())
        mean_orc_values.append(np.mean(orc_vals))
    mean_orc = float(np.mean(mean_orc_values))

    # Escape rates
    frac_orc_better = orc_result['frac_leads_to_better']
    frac_rand_better = orc_result['frac_random_leads_to_better']
    frac_mg_better = orc_result['frac_mingap_leads_to_better']

    return {
        'alpha': alpha,
        'seed': seed,
        'n_vars': n_vars,
        'n_clauses': n_clauses,
        'is_satisfiable': is_satisfiable,
        'global_min_fitness': global_min_fitness,
        'n_global_optima': n_global_optima,
        'n_local_optima': n_opt,
        'n_sinks': n_sinks,
        'n_attractors': n_attractors,
        'compression': compression,
        'frac_in_multi_cycle': frac_in_multi_cycle,
        'dag_depth': dag_depth,
        'n_sccs': len(sccs),
        'frac_reach_global': frac_reach_global,
        'mean_terminal_rank': mean_terminal_rank,
        'frac_terminal_top5': frac_terminal_top5,
        'frac_terminal_top10': frac_terminal_top10,
        'path_to_terminal': path_stats,
        'path_to_top5': path_top5_stats,
        'd1_mean_rank': d1_mean_rank,
        'd1_frac_top5': d1_frac_top5,
        'd1_compression': d1_compression,
        'mg_mean_rank': mg_mean_rank,
        'mg_frac_top5': mg_frac_top5,
        'mean_orc': mean_orc,
        'frac_orc_better': frac_orc_better,
        'frac_rand_better': frac_rand_better,
        'frac_mg_better': frac_mg_better,
        'best_attractor_fitness': best_attractor_fitness,
        'orc_time_s': round(orc_time, 3),
        'skip': False,
    }


def main():
    parser = argparse.ArgumentParser(description='MAX-SAT OTG phase transition analysis')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-vars', type=int, default=16,
                        help='Number of variables (default: 16, use 20 for publication)')
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/maxsat_otg.json')
    args = parser.parse_args()

    alphas = [2.0, 3.0, 3.5, 4.0, 4.27, 4.5, 5.0, 6.0, 8.0]

    tasks = []
    for alpha in alphas:
        for seed in range(args.n_instances):
            tasks.append({
                'n_vars': args.n_vars,
                'alpha': alpha,
                'seed': seed,
                'gamma': args.gamma,
            })

    total = len(tasks)
    workers = min(args.workers, total)

    print(f"MAX-SAT OTG Phase Transition Analysis")
    print(f"  Variables: {args.n_vars}")
    print(f"  Alpha values: {alphas}")
    print(f"  Instances per alpha: {args.n_instances}")
    print(f"  Total tasks: {total}")
    print(f"  Workers: {workers}")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    if workers <= 1:
        for i, task in enumerate(tasks):
            r = _analyze_maxsat_instance(task)
            results.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  [{i+1}/{total}] {elapsed:.0f}s", flush=True)
    else:
        with Pool(workers) as pool:
            completed = 0
            for row in pool.imap_unordered(_analyze_maxsat_instance, tasks):
                results.append(row)
                completed += 1
                if completed % 20 == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / completed * (total - completed)
                    print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed  "
                          f"ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.0f}s")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else
                  float(o) if isinstance(o, (np.floating,)) else
                  bool(o) if isinstance(o, (np.bool_,)) else None)
    print(f"Saved {len(results)} results to {out_path}")

    # Print summary
    valid = [r for r in results if not r.get('skip', False)]
    groups = defaultdict(list)
    for r in valid:
        groups[r['alpha']].append(r)

    print(f"\n{'='*160}")
    print(f"MAX-SAT OTG Phase Transition Summary (N={args.n_vars})")
    print(f"{'='*160}")
    print(f"{'alpha':>6} {'%SAT':>5} {'#Opt':>7} {'#Attr':>6} {'Compr%':>7} "
          f"{'%Cycle':>7} {'DAGd':>5} {'OTG Rank':>9} {'d1 Rank':>8} {'MG Rank':>8} "
          f"{'OTG T5%':>8} {'d1 T5%':>7} {'MG T5%':>7} "
          f"{'%ORC':>6} {'%Rnd':>6} {'%MG':>6} {'MeanORC':>8} {'BestAtr':>8}")
    print(f"{'-'*160}")

    for alpha in sorted(groups.keys()):
        rows = groups[alpha]
        n = len(rows)
        psat = 100 * np.mean([r['is_satisfiable'] for r in rows])
        nopt = np.mean([r['n_local_optima'] for r in rows])
        nattr = np.mean([r['n_attractors'] for r in rows])
        compr = 100 * np.mean([r['compression'] for r in rows])
        fcyc = 100 * np.mean([r['frac_in_multi_cycle'] for r in rows])
        dagd = np.mean([r['dag_depth'] for r in rows])
        otg_r = np.mean([r['mean_terminal_rank'] for r in rows])
        d1_r = np.mean([r['d1_mean_rank'] for r in rows])
        mg_r = np.mean([r['mg_mean_rank'] for r in rows])
        otg_t5 = 100 * np.mean([r['frac_terminal_top5'] for r in rows])
        d1_t5 = 100 * np.mean([r['d1_frac_top5'] for r in rows])
        mg_t5 = 100 * np.mean([r['mg_frac_top5'] for r in rows])
        orc_b = 100 * np.mean([r['frac_orc_better'] for r in rows])
        rnd_b = 100 * np.mean([r['frac_rand_better'] for r in rows])
        mg_b = 100 * np.mean([r['frac_mg_better'] for r in rows])
        morc = np.mean([r['mean_orc'] for r in rows])
        batr = np.mean([r['best_attractor_fitness'] for r in rows])

        print(f"{alpha:>6.2f} {psat:>4.0f}% {nopt:>7.0f} {nattr:>6.0f} {compr:>6.1f}% "
              f"{fcyc:>6.1f}% {dagd:>5.1f} {otg_r:>9.3f} {d1_r:>8.3f} {mg_r:>8.3f} "
              f"{otg_t5:>7.1f}% {d1_t5:>6.1f}% {mg_t5:>6.1f}% "
              f"{orc_b:>5.1f}% {rnd_b:>5.1f}% {mg_b:>5.1f}% {morc:>8.4f} {batr:>8.2f}")

    # Satisfiability prediction
    print(f"\n{'='*100}")
    print(f"OTG Features vs Satisfiability (instances near phase transition, alpha=4.0-4.5)")
    print(f"{'='*100}")

    near_transition = [r for r in valid if 4.0 <= r['alpha'] <= 4.5]
    if near_transition:
        sat = [r for r in near_transition if r['is_satisfiable']]
        unsat = [r for r in near_transition if not r['is_satisfiable']]
        if sat and unsat:
            print(f"  {'Metric':<25} {'SAT (n={})'.format(len(sat)):>15} "
                  f"{'UNSAT (n={})'.format(len(unsat)):>15} {'Separation':>12}")
            print(f"  {'-'*70}")
            for metric, label in [
                ('mean_terminal_rank', 'Mean terminal rank'),
                ('frac_terminal_top5', 'Frac top-5%'),
                ('compression', 'Compression'),
                ('frac_in_multi_cycle', 'Frac in cycles'),
                ('dag_depth', 'DAG depth'),
                ('mean_orc', 'Mean ORC'),
                ('n_local_optima', '#Local optima'),
                ('best_attractor_fitness', 'Best attractor unsat'),
            ]:
                sat_v = np.mean([r[metric] for r in sat])
                unsat_v = np.mean([r[metric] for r in unsat])
                sep = abs(sat_v - unsat_v) / (max(abs(sat_v), abs(unsat_v), 1e-10))
                print(f"  {label:<25} {sat_v:>15.4f} {unsat_v:>15.4f} {100*sep:>11.1f}%")
        else:
            print(f"  All instances are {'SAT' if sat else 'UNSAT'} — no comparison possible.")
    else:
        print("  No instances near the phase transition.")


if __name__ == '__main__':
    freeze_support()
    main()

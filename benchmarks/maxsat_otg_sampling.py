#!/usr/bin/env python3
"""
Sampling-Based OTG on Random 3-SAT/MAX-SAT at Scale (N=50-500).

No full enumeration — discovers local optima via random hill climbing,
computes ORC at each discovered optimum, builds OTG over the sample.

Usage:
    python3 benchmarks/maxsat_otg_sampling.py --workers 150
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
from scipy.optimize import linear_sum_assignment


# ---- MAX-SAT primitives (no precomputed arrays) ----

def generate_random_3sat(n_vars: int, alpha: float, seed: int = 0):
    rng = np.random.RandomState(seed)
    n_clauses = int(round(alpha * n_vars))
    clauses = []
    for _ in range(n_clauses):
        variables = rng.choice(n_vars, size=3, replace=False)
        signs = rng.choice([False, True], size=3)  # True = positive literal
        clauses.append((variables, signs))
    return clauses, n_clauses


def eval_fitness(x_bits: np.ndarray, clauses: list) -> int:
    """Count unsatisfied clauses for a bitstring (numpy bool/int array)."""
    unsat = 0
    for variables, signs in clauses:
        satisfied = False
        for var, sign in zip(variables, signs):
            if x_bits[var] == sign:
                satisfied = True
                break
        if not satisfied:
            unsat += 1
    return unsat


def eval_fitness_batch(x_bits: np.ndarray, clauses: list, neighbor_indices: list) -> np.ndarray:
    """Evaluate fitness for x and all its single-bit-flip neighbors.
    Returns array of length 1 + len(neighbor_indices): [f(x), f(n0), f(n1), ...]"""
    n = len(neighbor_indices)
    results = np.empty(1 + n, dtype=np.float64)
    results[0] = eval_fitness(x_bits, clauses)
    for i, bit_idx in enumerate(neighbor_indices):
        x_bits[bit_idx] ^= True
        results[1 + i] = eval_fitness(x_bits, clauses)
        x_bits[bit_idx] ^= True
    return results


def hill_climb(start_bits: np.ndarray, clauses: list, max_steps: int = 10000):
    """Steepest descent hill climbing on MAX-SAT (minimize unsatisfied clauses).
    Returns (optimum_bits, fitness, n_evals)."""
    current = start_bits.copy()
    n_vars = len(current)
    current_fit = eval_fitness(current, clauses)
    n_evals = 1

    for _ in range(max_steps):
        best_bit = -1
        best_fit = current_fit
        for i in range(n_vars):
            current[i] ^= True
            f = eval_fitness(current, clauses)
            n_evals += 1
            if f < best_fit:
                best_fit = f
                best_bit = i
            current[i] ^= True
        if best_bit == -1:
            break
        current[best_bit] ^= True
        current_fit = best_fit

    return current, current_fit, n_evals


def bits_to_key(bits: np.ndarray) -> bytes:
    """Convert bit array to hashable key."""
    return bits.tobytes()


# ---- ORC computation (function-based, no precomputed array) ----

def compute_orc_at_optimum(opt_bits: np.ndarray, clauses: list, gamma: float = 1.0):
    """
    Compute ORC for all N edges incident to a local optimum.
    Returns dict {bit_index: orc_value}, and the fitness cache populated.
    """
    n_vars = len(opt_bits)

    sup_u_bits = [opt_bits.copy()]
    for i in range(n_vars):
        nbr = opt_bits.copy()
        nbr[i] ^= True
        sup_u_bits.append(nbr)

    sup_u_fit = np.array([eval_fitness(b, clauses) for b in sup_u_bits], dtype=np.float64)
    sup_u_set = set()
    for b in sup_u_bits:
        sup_u_set.add(bits_to_key(b))

    orc_values = {}
    for nbr_idx in range(n_vars):
        nbr_bits = sup_u_bits[1 + nbr_idx]  # neighbor = opt with bit nbr_idx flipped

        sup_v_bits = [nbr_bits.copy()]
        for i in range(n_vars):
            vv = nbr_bits.copy()
            vv[i] ^= True
            sup_v_bits.append(vv)

        sup_v_fit = np.array([eval_fitness(b, clauses) for b in sup_v_bits], dtype=np.float64)
        sup_v_set = set()
        for b in sup_v_bits:
            sup_v_set.add(bits_to_key(b))

        m = min(len(sup_u_bits), len(sup_v_bits))
        C = np.empty((m, m), dtype=np.float64)
        for i in range(m):
            a_key = bits_to_key(sup_u_bits[i])
            for j in range(m):
                b_key = bits_to_key(sup_v_bits[j])
                if a_key == b_key:
                    C[i, j] = 0.0
                else:
                    graph_d = 1.0 if (b_key in sup_u_set or a_key in sup_v_set) else 2.0
                    C[i, j] = graph_d + gamma * abs(sup_u_fit[i] - sup_v_fit[j])

        d_uv = C[0, 0]
        if d_uv < 1e-12:
            orc_values[nbr_idx] = 0.0
            continue

        row_ind, col_ind = linear_sum_assignment(C)
        W1 = float(np.sum(C[row_ind, col_ind])) / m
        orc_values[nbr_idx] = float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))

    return orc_values, float(sup_u_fit[0])


def _analyze_maxsat_sampling(args: dict) -> dict:
    """Full sampling-based OTG analysis for one MAX-SAT instance."""
    n_vars = args['n_vars']
    alpha = args['alpha']
    seed = args['seed']
    gamma = args.get('gamma', 1.0)
    n_restarts = args.get('n_restarts', 5000)

    rng = np.random.RandomState(seed + 10000)
    clauses, n_clauses = generate_random_3sat(n_vars, alpha, seed)

    t0 = time.perf_counter()

    # Phase 1: Discover local optima via random restarts
    optima_dict = {}  # key -> (bits, fitness)
    hc_destinations = []  # which optimum each restart lands on
    total_evals = 0

    for _ in range(n_restarts):
        start = rng.randint(0, 2, size=n_vars).astype(bool)
        opt_bits, opt_fit, evals = hill_climb(start, clauses)
        total_evals += evals
        key = bits_to_key(opt_bits)
        if key not in optima_dict:
            optima_dict[key] = (opt_bits.copy(), int(opt_fit))
        hc_destinations.append(key)

    n_opt_discovered = len(optima_dict)
    t_discover = time.perf_counter() - t0

    if n_opt_discovered < 2:
        return {
            'alpha': alpha, 'seed': seed, 'n_vars': n_vars,
            'n_clauses': n_clauses, 'n_restarts': n_restarts,
            'n_optima_discovered': n_opt_discovered,
            'total_evals': total_evals, 'skip': True,
            'discover_time_s': round(t_discover, 2),
        }

    # Basin sizes (from restarts)
    basin_counts = defaultdict(int)
    for key in hc_destinations:
        basin_counts[key] += 1

    # Global best among discovered
    best_key = min(optima_dict, key=lambda k: optima_dict[k][1])
    global_best_fitness = optima_dict[best_key][1]
    is_satisfiable_found = (global_best_fitness == 0)

    # Phase 2: Compute ORC at each discovered optimum + build OTG
    t_orc_start = time.perf_counter()

    otg_edges = {}  # key -> key
    opt_orc_data = {}  # key -> {mean_orc, min_orc, min_orc_bit}
    orc_escape_better = 0
    rand_escape_better_total = 0.0
    mg_escape_better = 0

    optima_list = list(optima_dict.keys())
    optima_fitnesses = {k: optima_dict[k][1] for k in optima_list}

    for opt_key in optima_list:
        opt_bits, opt_fit = optima_dict[opt_key]

        orc_values, _ = compute_orc_at_optimum(opt_bits, clauses, gamma)

        min_orc_bit = min(orc_values, key=orc_values.get)
        min_orc_val = orc_values[min_orc_bit]
        mean_orc = float(np.mean(list(orc_values.values())))

        opt_orc_data[opt_key] = {
            'mean_orc': mean_orc,
            'min_orc': min_orc_val,
            'min_orc_bit': min_orc_bit,
        }

        # Follow min-ORC direction: flip that bit, hill-climb
        escape_bits = opt_bits.copy()
        escape_bits[min_orc_bit] ^= True
        dest_bits, dest_fit, _ = hill_climb(escape_bits, clauses)
        dest_key = bits_to_key(dest_bits)

        if dest_key not in optima_dict:
            optima_dict[dest_key] = (dest_bits.copy(), int(dest_fit))
            optima_fitnesses[dest_key] = int(dest_fit)

        otg_edges[opt_key] = dest_key

        if dest_fit < opt_fit:
            orc_escape_better += 1

        # MinGap baseline
        mg_bit = -1
        mg_gap = float('inf')
        for i in range(n_vars):
            opt_bits[i] ^= True
            f = eval_fitness(opt_bits, clauses)
            gap = abs(f - opt_fit)
            if gap < mg_gap:
                mg_gap = gap
                mg_bit = i
            opt_bits[i] ^= True

        mg_escape = opt_bits.copy()
        mg_escape[mg_bit] ^= True
        mg_dest_bits, mg_dest_fit, _ = hill_climb(mg_escape, clauses)
        if mg_dest_fit < opt_fit:
            mg_escape_better += 1

        # Random baseline (10 trials for speed)
        rand_better = 0
        for _ in range(10):
            rand_bit = rng.randint(n_vars)
            rand_escape = opt_bits.copy()
            rand_escape[rand_bit] ^= True
            rand_dest_bits, rand_dest_fit, _ = hill_climb(rand_escape, clauses)
            if rand_dest_fit < opt_fit:
                rand_better += 1
        rand_escape_better_total += rand_better / 10.0

    t_orc = time.perf_counter() - t_orc_start
    n_opt = len(optima_list)

    # Phase 3: Analyze OTG structure
    def trace_path(start_key):
        visited_order = []
        visited_set = set()
        current = start_key
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
    for opt_key in optima_list:
        path, cycle = trace_path(opt_key)
        opt_paths[opt_key] = path
        opt_terminal_cycle[opt_key] = cycle
        cycle_key = frozenset(cycle)
        if cycle_key not in seen_cycles:
            seen_cycles.add(cycle_key)
            all_cycles.append(cycle)

    sinks = set()
    for cycle in all_cycles:
        sinks.update(cycle)

    n_attractors = len(all_cycles)
    multi_node_cycles = [c for c in all_cycles if len(c) > 1]
    n_in_multi_cycle = sum(len(c) for c in multi_node_cycles)
    frac_in_multi_cycle = n_in_multi_cycle / n_opt if n_opt > 0 else 0.0

    # Attractor quality
    sorted_fits = sorted(optima_fitnesses[k] for k in optima_list)
    n_sorted = len(sorted_fits)
    sorted_fits_arr = np.array(sorted_fits)

    terminal_ranks = []
    for opt_key in optima_list:
        cycle = opt_terminal_cycle[opt_key]
        best_in_cycle = min(cycle, key=lambda k: optima_fitnesses.get(k, 1e9))
        term_fit = optima_fitnesses.get(best_in_cycle, 1e9)
        rank = float(np.searchsorted(sorted_fits_arr, term_fit)) / max(n_sorted - 1, 1)
        terminal_ranks.append(rank)

    mean_terminal_rank = float(np.mean(terminal_ranks))
    frac_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in terminal_ranks]))
    frac_top10 = float(np.mean([1 if r <= 0.10 else 0 for r in terminal_ranks]))

    # Path lengths
    hops_to_terminal = []
    for opt_key in optima_list:
        path = opt_paths[opt_key]
        length = 0
        for node in path:
            if node in sinks:
                break
            length += 1
        hops_to_terminal.append(length)

    # Best attractor fitness
    best_attractor_fit = min(
        optima_fitnesses.get(min(c, key=lambda k: optima_fitnesses.get(k, 1e9)), 1e9)
        for c in all_cycles
    )

    # Global best reachability
    reaches_global = sum(1 for opt_key in optima_list
                         if best_key in opt_paths[opt_key] or
                         best_key in opt_terminal_cycle[opt_key])
    frac_reach_global = reaches_global / n_opt if n_opt > 0 else 0.0

    compression = n_attractors / n_opt if n_opt > 0 else 1.0
    mean_orc_all = float(np.mean([opt_orc_data[k]['mean_orc'] for k in optima_list]))

    # DAG depth
    import networkx as nx
    G = nx.DiGraph()
    for src, dst in otg_edges.items():
        if src != dst:
            G.add_edge(src, dst)
        else:
            G.add_node(src)
    cond = nx.condensation(G)
    dag_depth = nx.dag_longest_path_length(cond) if len(cond) > 1 else 0

    total_time = time.perf_counter() - t0

    return {
        'alpha': alpha,
        'seed': seed,
        'n_vars': n_vars,
        'n_clauses': n_clauses,
        'n_restarts': n_restarts,
        'n_optima_discovered': n_opt_discovered,
        'n_optima_total': len(optima_dict),
        'is_satisfiable_found': is_satisfiable_found,
        'global_best_fitness': global_best_fitness,
        'n_attractors': n_attractors,
        'compression': compression,
        'frac_in_multi_cycle': frac_in_multi_cycle,
        'dag_depth': dag_depth,
        'mean_terminal_rank': mean_terminal_rank,
        'frac_top5': frac_top5,
        'frac_top10': frac_top10,
        'best_attractor_fitness': best_attractor_fit,
        'frac_reach_global': frac_reach_global,
        'mean_orc': mean_orc_all,
        'frac_orc_better': orc_escape_better / n_opt if n_opt > 0 else 0.0,
        'frac_rand_better': rand_escape_better_total / n_opt if n_opt > 0 else 0.0,
        'frac_mg_better': mg_escape_better / n_opt if n_opt > 0 else 0.0,
        'path_median': float(np.median(hops_to_terminal)),
        'path_mean': float(np.mean(hops_to_terminal)),
        'path_max': int(np.max(hops_to_terminal)),
        'total_evals': total_evals,
        'discover_time_s': round(t_discover, 2),
        'orc_time_s': round(t_orc, 2),
        'total_time_s': round(total_time, 2),
        'skip': False,
    }


def main():
    parser = argparse.ArgumentParser(description='Sampling-based MAX-SAT OTG')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=20)
    parser.add_argument('--n-restarts', type=int, default=2000,
                        help='Random hill-climb restarts per instance')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/maxsat_otg_scaling.json')
    args = parser.parse_args()

    configs = []
    alphas = [3.0, 4.0, 4.27, 4.5, 5.0, 6.0]

    for n_vars in [50, 100, 200]:
        restarts = args.n_restarts
        for alpha in alphas:
            for seed in range(args.n_instances):
                configs.append({
                    'n_vars': n_vars,
                    'alpha': alpha,
                    'seed': seed,
                    'gamma': args.gamma,
                    'n_restarts': restarts,
                })

    total = len(configs)
    workers = min(args.workers, total)

    print(f"Sampling-Based MAX-SAT OTG Scaling Analysis")
    print(f"  N values: [50, 100, 200]")
    print(f"  Alpha values: {alphas}")
    print(f"  Instances per config: {args.n_instances}")
    print(f"  Restarts per instance: {args.n_restarts}")
    print(f"  Total tasks: {total}")
    print(f"  Workers: {workers}")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    if workers <= 1:
        for i, task in enumerate(configs):
            r = _analyze_maxsat_sampling(task)
            results.append(r)
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{total}] {elapsed:.0f}s  N={task['n_vars']} a={task['alpha']}", flush=True)
    else:
        with Pool(workers) as pool:
            completed = 0
            for row in pool.imap_unordered(_analyze_maxsat_sampling, configs):
                results.append(row)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / completed * (total - completed)
                    print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed  "
                          f"ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.0f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else
                  float(o) if isinstance(o, (np.floating,)) else
                  bool(o) if isinstance(o, (np.bool_,)) else None)
    print(f"Saved {len(results)} results to {out_path}")

    # Summary
    valid = [r for r in results if not r.get('skip', False)]
    groups = defaultdict(list)
    for r in valid:
        groups[(r['n_vars'], r['alpha'])].append(r)

    print(f"\n{'='*170}")
    print(f"Sampling-Based MAX-SAT OTG Summary")
    print(f"{'='*170}")
    print(f"{'N':>4} {'alpha':>6} {'%SAT':>5} {'#Opt':>7} {'#Attr':>6} {'Compr%':>7} "
          f"{'%Cycle':>7} {'DAGd':>5} {'OTG Rank':>9} "
          f"{'OTG T5%':>8} {'%ORC':>6} {'%Rnd':>6} {'%MG':>6} "
          f"{'MeanORC':>8} {'BestAtr':>8} {'Time':>6}")
    print(f"{'-'*170}")

    for n_vars in [50, 100, 200]:
        for alpha in sorted(set(r['alpha'] for r in valid)):
            key = (n_vars, alpha)
            if key not in groups:
                continue
            rows = groups[key]
            n = len(rows)
            psat = 100 * np.mean([r['is_satisfiable_found'] for r in rows])
            nopt = np.mean([r['n_optima_discovered'] for r in rows])
            nattr = np.mean([r['n_attractors'] for r in rows])
            compr = 100 * np.mean([r['compression'] for r in rows])
            fcyc = 100 * np.mean([r['frac_in_multi_cycle'] for r in rows])
            dagd = np.mean([r['dag_depth'] for r in rows])
            otg_r = np.mean([r['mean_terminal_rank'] for r in rows])
            otg_t5 = 100 * np.mean([r['frac_top5'] for r in rows])
            orc_b = 100 * np.mean([r['frac_orc_better'] for r in rows])
            rnd_b = 100 * np.mean([r['frac_rand_better'] for r in rows])
            mg_b = 100 * np.mean([r['frac_mg_better'] for r in rows])
            morc = np.mean([r['mean_orc'] for r in rows])
            batr = np.mean([r['best_attractor_fitness'] for r in rows])
            tt = np.mean([r['total_time_s'] for r in rows])

            print(f"{n_vars:>4} {alpha:>6.2f} {psat:>4.0f}% {nopt:>7.0f} {nattr:>6.0f} {compr:>6.1f}% "
                  f"{fcyc:>6.1f}% {dagd:>5.1f} {otg_r:>9.3f} "
                  f"{otg_t5:>7.1f}% {orc_b:>5.1f}% {rnd_b:>5.1f}% {mg_b:>5.1f}% "
                  f"{morc:>8.4f} {batr:>8.2f} {tt:>5.0f}s")
        print()

    # Phase transition analysis: does OTG sharpen with N?
    print(f"\n{'='*100}")
    print(f"ORC Escape Advantage (ORC/Random ratio) vs N and alpha")
    print(f"{'='*100}")
    print(f"{'alpha':>6}", end='')
    for n_vars in [50, 100, 200]:
        print(f"  {'N='+str(n_vars):>10}", end='')
    print()
    for alpha in alphas:
        print(f"{alpha:>6.2f}", end='')
        for n_vars in [50, 100, 200]:
            key = (n_vars, alpha)
            if key in groups:
                rows = groups[key]
                orc_b = np.mean([r['frac_orc_better'] for r in rows])
                rnd_b = np.mean([r['frac_rand_better'] for r in rows])
                ratio = orc_b / max(rnd_b, 0.001)
                print(f"  {ratio:>10.2f}x", end='')
            else:
                print(f"  {'N/A':>10}", end='')
        print()


if __name__ == '__main__':
    freeze_support()
    main()

#!/usr/bin/env python3
"""
Sampling-Based OTG on Random MAX-SAT at Scale (N=50-500).

Key optimizations over the naive approach:
  1. Proposition 1: On the hypercube, ORC reduces to 1D optimal transport
     (sort + match), eliminating the O(N^3) Hungarian algorithm entirely.
  2. Vectorized numpy fitness evaluation for all 2-hop solutions at once.
  3. Incremental hill climbing with batch neighbor evaluation.

Usage:
    python3 benchmarks/maxsat_otg_scaling.py --workers 150
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np


class MaxSATInstance:
    """Efficient MAX-SAT with vectorized operations."""

    __slots__ = ('n_vars', 'n_clauses', 'alpha', 'clause_vars', 'clause_signs',
                 'clauses_for_var')

    def __init__(self, n_vars: int, alpha: float, seed: int = 0):
        self.n_vars = n_vars
        self.alpha = alpha
        self.n_clauses = int(round(alpha * n_vars))

        rng = np.random.RandomState(seed)
        cv = np.empty((self.n_clauses, 3), dtype=np.intp)
        cs = np.empty((self.n_clauses, 3), dtype=np.bool_)
        for i in range(self.n_clauses):
            cv[i] = rng.choice(n_vars, size=3, replace=False)
            cs[i] = rng.randint(0, 2, size=3).astype(bool)
        self.clause_vars = cv
        self.clause_signs = cs

        cfv = [[] for _ in range(n_vars)]
        for c in range(self.n_clauses):
            for v in cv[c]:
                cfv[v].append(c)
        self.clauses_for_var = cfv

    def eval_batch(self, solutions: np.ndarray) -> np.ndarray:
        """Evaluate fitness for a batch of solutions. (batch, N) -> (batch,)"""
        lit_vals = solutions[:, self.clause_vars]       # (batch, M, 3)
        lit_match = lit_vals == self.clause_signs        # (batch, M, 3)
        clause_sat = lit_match.any(axis=2)               # (batch, M)
        return (~clause_sat).sum(axis=1).astype(np.float64)

    def eval_single(self, x: np.ndarray) -> int:
        lit_vals = x[self.clause_vars]       # (M, 3)
        lit_match = lit_vals == self.clause_signs
        return int((~lit_match.any(axis=1)).sum())

    def hill_climb(self, start: np.ndarray) -> tuple:
        """Steepest descent with batch neighbor evaluation. Returns (opt, fitness, steps)."""
        n = self.n_vars
        current = start.copy()
        current_fit = self.eval_single(current)
        steps = 0

        idx = np.arange(n)
        while True:
            neighbors = np.tile(current, (n, 1))
            neighbors[idx, idx] ^= True
            fits = self.eval_batch(neighbors)
            best = int(np.argmin(fits))
            if fits[best] >= current_fit:
                break
            current = neighbors[best]
            current_fit = int(fits[best])
            steps += 1

        return current, current_fit, steps


def compute_orc_fast(opt_bits: np.ndarray, inst: MaxSATInstance,
                     gamma: float = 1.0) -> dict:
    """
    Compute ORC for all N edges at a local optimum using Proposition 1.

    On the hypercube, for edge (x*, y) where y = x* XOR e_i:
      - 2 shared elements (x*, y) with cost 0
      - (N-1) exclusive pairs at graph distance 2
      - Matching cost = sum of sorted |f(a_j) - f(b_sigma(j))|
        solved optimally by sorting (1D Wasserstein)

    Total complexity: O(N^2 * M) for fitness + O(N^2 log N) for sorts.
    """
    n = inst.n_vars

    # Step 1: Compute fitness of x* and all 1-hop neighbors
    opt_fit = inst.eval_single(opt_bits)
    idx = np.arange(n)
    nbr_1hop = np.tile(opt_bits, (n, 1))
    nbr_1hop[idx, idx] ^= True
    fit_1hop = inst.eval_batch(nbr_1hop)  # (N,)  f(x* XOR e_j) for j=0..N-1

    # Step 2: Compute fitness of all 2-hop neighbors
    # x* XOR e_i XOR e_k for i<k (unique pairs)
    # We store them in a (N, N) matrix: fit_2hop[i, k] = f(x* XOR e_i XOR e_k)
    # Only need i != k; diagonal is x* itself.

    pairs_i, pairs_k = np.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    n_pairs = len(pairs_i)

    # Build the 2-hop solutions
    nbr_2hop = np.tile(opt_bits, (n_pairs, 1))
    nbr_2hop[np.arange(n_pairs), pairs_i] ^= True
    nbr_2hop[np.arange(n_pairs), pairs_k] ^= True
    fit_2hop_vals = inst.eval_batch(nbr_2hop)  # (n_pairs,)

    # Fill symmetric matrix
    fit_2hop = np.full((n, n), np.nan)
    fit_2hop[pairs_i, pairs_k] = fit_2hop_vals
    fit_2hop[pairs_k, pairs_i] = fit_2hop_vals

    # Step 3: Compute ORC for each edge using Proposition 1
    orc_values = {}
    for i in range(n):
        # Edge (x*, x* XOR e_i)
        # Exclusive neighbors of x*: {x* XOR e_j : j != i} with fitnesses fit_1hop[j]
        # Exclusive neighbors of y: {x* XOR e_i XOR e_k : k != i} with fitnesses fit_2hop[i, k]

        mask = np.ones(n, dtype=bool)
        mask[i] = False

        a_fits = np.sort(fit_1hop[mask])        # sorted exclusive x* neighbor fitnesses
        b_fits = np.sort(fit_2hop[i, mask])      # sorted exclusive y neighbor fitnesses

        sort_cost = float(np.sum(np.abs(a_fits - b_fits)))
        total_cost = 2.0 * (n - 1) + gamma * sort_cost
        W1 = total_cost / (n + 1)

        delta = abs(opt_fit - fit_1hop[i])
        d_uv = 1.0 + gamma * delta
        if d_uv < 1e-12:
            orc_values[i] = 0.0
        else:
            orc_values[i] = float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))

    return orc_values


def _analyze_instance(args: dict) -> dict:
    """Full sampling-based OTG analysis for one MAX-SAT instance."""
    n_vars = args['n_vars']
    alpha = args['alpha']
    seed = args['seed']
    gamma = args.get('gamma', 1.0)
    n_restarts = args.get('n_restarts', 2000)

    rng = np.random.RandomState(seed + 10000)
    inst = MaxSATInstance(n_vars, alpha, seed)

    t0 = time.perf_counter()

    # Phase 1: Discover local optima
    optima = {}  # key(bytes) -> (bits, fitness)
    basin_counts = defaultdict(int)

    for _ in range(n_restarts):
        start = rng.randint(0, 2, size=n_vars).astype(bool)
        opt_bits, opt_fit, _ = inst.hill_climb(start)
        key = opt_bits.tobytes()
        if key not in optima:
            optima[key] = (opt_bits.copy(), opt_fit)
        basin_counts[key] += 1

    n_discovered = len(optima)
    t_discover = time.perf_counter() - t0

    if n_discovered < 2:
        return {'alpha': alpha, 'seed': seed, 'n_vars': n_vars,
                'n_clauses': inst.n_clauses, 'n_restarts': n_restarts,
                'n_discovered': n_discovered, 'skip': True,
                'discover_time_s': round(t_discover, 2)}

    # Global best among discovered
    best_key = min(optima, key=lambda k: optima[k][1])
    global_best_fit = optima[best_key][1]

    # Phase 2: Compute ORC at each optimum, build OTG
    t_orc = time.perf_counter()

    opt_keys = list(optima.keys())
    otg_edges = {}          # key -> key
    orc_data = {}           # key -> mean_orc
    escape_better = 0
    mg_better = 0
    rand_better_sum = 0.0

    for opt_key in opt_keys:
        opt_bits, opt_fit = optima[opt_key]

        orc_vals = compute_orc_fast(opt_bits, inst, gamma)
        min_bit = min(orc_vals, key=orc_vals.get)
        orc_data[opt_key] = float(np.mean(list(orc_vals.values())))

        # ORC escape: flip min-ORC bit, hill-climb
        esc = opt_bits.copy()
        esc[min_bit] ^= True
        dest_bits, dest_fit, _ = inst.hill_climb(esc)
        dest_key = dest_bits.tobytes()
        if dest_key not in optima:
            optima[dest_key] = (dest_bits.copy(), dest_fit)
        otg_edges[opt_key] = dest_key
        if dest_fit < opt_fit:
            escape_better += 1

        # MinGap baseline
        n = n_vars
        idx = np.arange(n)
        nbrs = np.tile(opt_bits, (n, 1))
        nbrs[idx, idx] ^= True
        fits = inst.eval_batch(nbrs)
        gaps = np.abs(fits - opt_fit)
        mg_bit = int(np.argmin(gaps))
        mg_esc = opt_bits.copy()
        mg_esc[mg_bit] ^= True
        mg_dest, mg_fit, _ = inst.hill_climb(mg_esc)
        if mg_fit < opt_fit:
            mg_better += 1

        # Random baseline (10 trials)
        rb = 0
        for _ in range(10):
            rbit = rng.randint(n)
            resc = opt_bits.copy()
            resc[rbit] ^= True
            rd, rf, _ = inst.hill_climb(resc)
            if rf < opt_fit:
                rb += 1
        rand_better_sum += rb / 10.0

    orc_time = time.perf_counter() - t_orc
    n_opt = len(opt_keys)

    # Phase 3: Analyze OTG
    def trace(start):
        path, seen = [], set()
        cur = start
        while cur not in seen:
            seen.add(cur)
            path.append(cur)
            nxt = otg_edges.get(cur, cur)
            if nxt == cur:
                return path, {cur}
            cur = nxt
        ci = path.index(cur)
        return path, set(path[ci:])

    cycles_seen = set()
    all_cycles = []
    opt_terminal = {}
    opt_paths = {}
    for k in opt_keys:
        path, cycle = trace(k)
        opt_paths[k] = path
        opt_terminal[k] = cycle
        ck = frozenset(cycle)
        if ck not in cycles_seen:
            cycles_seen.add(ck)
            all_cycles.append(cycle)

    sinks = set()
    for c in all_cycles:
        sinks.update(c)

    n_attractors = len(all_cycles)
    multi_cycles = [c for c in all_cycles if len(c) > 1]
    n_in_multi = sum(len(c) for c in multi_cycles)

    # Attractor quality
    all_fits = sorted(optima[k][1] for k in opt_keys)
    all_fits_arr = np.array(all_fits)
    n_sorted = len(all_fits_arr)

    terminal_ranks = []
    for k in opt_keys:
        cycle = opt_terminal[k]
        best_c = min(cycle, key=lambda ck: optima.get(ck, (None, 1e9))[1])
        tf = optima.get(best_c, (None, 1e9))[1]
        rank = float(np.searchsorted(all_fits_arr, tf)) / max(n_sorted - 1, 1)
        terminal_ranks.append(rank)

    # Path lengths
    hops = []
    for k in opt_keys:
        p = opt_paths[k]
        l = 0
        for node in p:
            if node in sinks:
                break
            l += 1
        hops.append(l)

    # Best attractor
    best_attr_fit = min(
        optima.get(min(c, key=lambda ck: optima.get(ck, (None, 1e9))[1]),
                   (None, 1e9))[1]
        for c in all_cycles
    )

    # Global reachability
    reaches = sum(1 for k in opt_keys
                  if best_key in opt_paths[k] or best_key in opt_terminal[k])

    # DAG depth
    import networkx as nx
    G = nx.DiGraph()
    for s, d in otg_edges.items():
        if s != d:
            G.add_edge(s, d)
        else:
            G.add_node(s)
    cond = nx.condensation(G)
    dag_depth = nx.dag_longest_path_length(cond) if len(cond) > 1 else 0

    total_time = time.perf_counter() - t0

    return {
        'alpha': alpha, 'seed': seed, 'n_vars': n_vars,
        'n_clauses': inst.n_clauses, 'n_restarts': n_restarts,
        'n_discovered': n_discovered,
        'is_sat_found': global_best_fit == 0,
        'global_best_fit': global_best_fit,
        'n_attractors': n_attractors,
        'compression': n_attractors / n_opt if n_opt > 0 else 1.0,
        'frac_multi_cycle': n_in_multi / n_opt if n_opt > 0 else 0.0,
        'dag_depth': dag_depth,
        'mean_rank': float(np.mean(terminal_ranks)),
        'frac_top5': float(np.mean([1 if r <= 0.05 else 0 for r in terminal_ranks])),
        'frac_top10': float(np.mean([1 if r <= 0.10 else 0 for r in terminal_ranks])),
        'best_attr_fit': best_attr_fit,
        'frac_reach_best': reaches / n_opt if n_opt > 0 else 0.0,
        'mean_orc': float(np.mean([orc_data[k] for k in opt_keys])),
        'frac_orc_better': escape_better / n_opt if n_opt > 0 else 0.0,
        'frac_rand_better': rand_better_sum / n_opt if n_opt > 0 else 0.0,
        'frac_mg_better': mg_better / n_opt if n_opt > 0 else 0.0,
        'path_median': float(np.median(hops)),
        'path_mean': float(np.mean(hops)),
        'path_max': int(np.max(hops)),
        'discover_s': round(t_discover, 2),
        'orc_s': round(orc_time, 2),
        'total_s': round(total_time, 2),
        'skip': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=20)
    parser.add_argument('--n-restarts', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/maxsat_otg_scaling.json')
    args = parser.parse_args()

    alphas = [3.0, 4.0, 4.27, 4.5, 5.0, 6.0]
    n_values = [50, 100, 200]

    configs = []
    for n_vars in n_values:
        for alpha in alphas:
            for seed in range(args.n_instances):
                configs.append({
                    'n_vars': n_vars, 'alpha': alpha, 'seed': seed,
                    'gamma': args.gamma, 'n_restarts': args.n_restarts,
                })

    total = len(configs)
    workers = min(args.workers, total)

    print(f"Scaling MAX-SAT OTG (Proposition 1 fast path)")
    print(f"  N: {n_values}  alpha: {alphas}")
    print(f"  {args.n_instances} instances × {len(alphas)} alphas × {len(n_values)} N = {total} tasks")
    print(f"  {args.n_restarts} restarts/instance, {workers} workers")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    with Pool(workers) as pool:
        done = 0
        for row in pool.imap_unordered(_analyze_instance, configs):
            results.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                el = time.perf_counter() - t_start
                eta = el / done * (total - done)
                r = row
                print(f"  [{done}/{total}] {el:.0f}s  ETA {eta:.0f}s  "
                      f"last: N={r.get('n_vars')} a={r.get('alpha')} "
                      f"t={r.get('total_s',0)}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer) else
                  float(o) if isinstance(o, np.floating) else
                  bool(o) if isinstance(o, np.bool_) else None)
    print(f"Saved {len(results)} to {args.out}")

    # ---- Summary ----
    valid = [r for r in results if not r.get('skip')]
    grp = defaultdict(list)
    for r in valid:
        grp[(r['n_vars'], r['alpha'])].append(r)

    print(f"\n{'='*160}")
    hdr = (f"{'N':>4} {'α':>5} {'%SAT':>5} {'#Opt':>6} {'#Atr':>5} "
           f"{'Comp%':>6} {'%Cyc':>5} {'DAG':>4} {'Rank':>6} "
           f"{'T5%':>5} {'%ORC':>5} {'%Rnd':>5} {'%MG':>5} "
           f"{'mORC':>7} {'BstAt':>5} {'time':>5}")
    print(hdr)
    print(f"{'-'*160}")

    for nv in n_values:
        for a in alphas:
            rows = grp.get((nv, a), [])
            if not rows:
                continue
            ps = 100*np.mean([r['is_sat_found'] for r in rows])
            no = np.mean([r['n_discovered'] for r in rows])
            na = np.mean([r['n_attractors'] for r in rows])
            co = 100*np.mean([r['compression'] for r in rows])
            cy = 100*np.mean([r['frac_multi_cycle'] for r in rows])
            dd = np.mean([r['dag_depth'] for r in rows])
            rk = np.mean([r['mean_rank'] for r in rows])
            t5 = 100*np.mean([r['frac_top5'] for r in rows])
            ob = 100*np.mean([r['frac_orc_better'] for r in rows])
            rb = 100*np.mean([r['frac_rand_better'] for r in rows])
            mb = 100*np.mean([r['frac_mg_better'] for r in rows])
            mo = np.mean([r['mean_orc'] for r in rows])
            ba = np.mean([r['best_attr_fit'] for r in rows])
            tt = np.mean([r['total_s'] for r in rows])
            print(f"{nv:>4} {a:>5.2f} {ps:>4.0f}% {no:>6.0f} {na:>5.0f} "
                  f"{co:>5.1f}% {cy:>4.1f}% {dd:>4.1f} {rk:>6.3f} "
                  f"{t5:>4.1f}% {ob:>4.1f}% {rb:>4.1f}% {mb:>4.1f}% "
                  f"{mo:>7.4f} {ba:>5.1f} {tt:>4.0f}s")
        print()

    # ORC/Random ratio scaling
    print(f"\nORC/Random escape ratio scaling:")
    print(f"{'α':>6}", end='')
    for nv in n_values:
        print(f"  N={nv:>3}", end='')
    print()
    for a in alphas:
        print(f"{a:>6.2f}", end='')
        for nv in n_values:
            rows = grp.get((nv, a), [])
            if rows:
                o = np.mean([r['frac_orc_better'] for r in rows])
                r_ = np.mean([r['frac_rand_better'] for r in rows])
                print(f"  {o/max(r_,.001):>5.2f}x", end='')
            else:
                print(f"  {'N/A':>5}", end='')
        print()


if __name__ == '__main__':
    freeze_support()
    main()

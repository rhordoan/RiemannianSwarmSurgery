#!/usr/bin/env python3
"""
ORC+Pert-ILS vs baselines on MAX-SAT at N=100.

Uses wall-clock time budget for fair comparison: ORC's 2-hop evaluations
are cheap in absolute time (~2ms) but expensive in FE count (~5000),
so a time budget avoids penalizing ORC for its richer information usage.
"""

import numpy as np
import time
import json
from collections import defaultdict
from multiprocessing import Pool
from benchmarks.maxsat_otg_scaling import MaxSATInstance, compute_orc_fast


def run_orc_pert_ils(inst, time_budget, seed, gamma=1.0, d=3):
    rng = np.random.RandomState(seed)
    n = inst.n_vars
    t_start = time.perf_counter()

    start = rng.randint(0, 2, size=n).astype(bool)
    current, current_fit, _ = inst.hill_climb(start)
    best_fit = current_fit
    best_sol = current.copy()
    steps = 0

    while time.perf_counter() - t_start < time_budget:
        orc = compute_orc_fast(current, inst, gamma)
        min_orc_bit = min(orc, key=orc.get)

        perturbed = current.copy()
        perturbed[min_orc_bit] ^= True
        other_bits = [b for b in range(n) if b != min_orc_bit]
        extra = rng.choice(other_bits, size=min(d - 1, len(other_bits)), replace=False)
        for b in extra:
            perturbed[b] ^= True

        new_opt, new_fit, _ = inst.hill_climb(perturbed)
        current = new_opt
        current_fit = new_fit
        steps += 1

        if new_fit < best_fit:
            best_fit = new_fit
            best_sol = new_opt.copy()

    return best_fit, best_sol, steps


def run_random_ils(inst, time_budget, seed, d=3):
    rng = np.random.RandomState(seed)
    n = inst.n_vars
    t_start = time.perf_counter()

    start = rng.randint(0, 2, size=n).astype(bool)
    current, current_fit, _ = inst.hill_climb(start)
    best_fit = current_fit
    best_sol = current.copy()
    steps = 0

    while time.perf_counter() - t_start < time_budget:
        perturbed = current.copy()
        bits = rng.choice(n, size=min(d, n), replace=False)
        for b in bits:
            perturbed[b] ^= True

        new_opt, new_fit, _ = inst.hill_climb(perturbed)
        current = new_opt
        current_fit = new_fit
        steps += 1

        if new_fit < best_fit:
            best_fit = new_fit
            best_sol = new_opt.copy()

    return best_fit, best_sol, steps


def run_mingap_pert_ils(inst, time_budget, seed, d=3):
    rng = np.random.RandomState(seed)
    n = inst.n_vars
    idx = np.arange(n)
    t_start = time.perf_counter()

    start = rng.randint(0, 2, size=n).astype(bool)
    current, current_fit, _ = inst.hill_climb(start)
    best_fit = current_fit
    best_sol = current.copy()
    steps = 0

    while time.perf_counter() - t_start < time_budget:
        neighbors = np.tile(current, (n, 1))
        neighbors[idx, idx] ^= True
        fits = inst.eval_batch(neighbors)
        gaps = np.abs(fits - current_fit)
        mg_bit = int(np.argmin(gaps))

        perturbed = current.copy()
        perturbed[mg_bit] ^= True
        other_bits = [b for b in range(n) if b != mg_bit]
        extra = rng.choice(other_bits, size=min(d - 1, len(other_bits)), replace=False)
        for b in extra:
            perturbed[b] ^= True

        new_opt, new_fit, _ = inst.hill_climb(perturbed)
        current = new_opt
        current_fit = new_fit
        steps += 1

        if new_fit < best_fit:
            best_fit = new_fit
            best_sol = new_opt.copy()

    return best_fit, best_sol, steps


def run_random_restart_hc(inst, time_budget, seed):
    rng = np.random.RandomState(seed)
    n = inst.n_vars
    t_start = time.perf_counter()

    best_fit = float('inf')
    best_sol = None
    steps = 0

    while time.perf_counter() - t_start < time_budget:
        start = rng.randint(0, 2, size=n).astype(bool)
        opt, fit, _ = inst.hill_climb(start)
        steps += 1
        if fit < best_fit:
            best_fit = fit
            best_sol = opt.copy()

    return best_fit, best_sol, steps


def run_one_trial(args):
    n_vars, alpha, inst_seed, trial_seed, time_budget, gamma = args
    inst = MaxSATInstance(n_vars, alpha, inst_seed)

    results = {}
    for name, fn in [
        ('ORC+Pert', lambda: run_orc_pert_ils(inst, time_budget, trial_seed, gamma)),
        ('Random-ILS', lambda: run_random_ils(inst, time_budget, trial_seed)),
        ('MinGap+Pert', lambda: run_mingap_pert_ils(inst, time_budget, trial_seed)),
        ('RR-HC', lambda: run_random_restart_hc(inst, time_budget, trial_seed)),
    ]:
        t0 = time.perf_counter()
        best_fit, best_sol, steps = fn()
        wall = time.perf_counter() - t0
        results[name] = {
            'best_fit': int(best_fit),
            'is_sat': int(best_fit) == 0,
            'steps': steps,
            'wall_s': wall,
        }

    return {
        'n_vars': n_vars,
        'alpha': alpha,
        'inst_seed': inst_seed,
        'trial_seed': trial_seed,
        'n_clauses': inst.n_clauses,
        'methods': results,
    }


if __name__ == '__main__':
    n_vars = 100
    alphas = [3.5, 4.0, 4.27, 5.0, 6.0]
    n_instances = 30
    n_trials = 10
    time_budget = 10.0
    gamma = 1.0

    configs = []
    for alpha in alphas:
        for inst_seed in range(n_instances):
            for trial_seed in range(n_trials):
                configs.append((n_vars, alpha, inst_seed, trial_seed,
                                time_budget, gamma))

    total = len(configs)
    print(f"MAX-SAT ILS at N={n_vars}: {total} tasks "
          f"({len(alphas)} alphas x {n_instances} inst x {n_trials} trials, "
          f"{time_budget}s budget)")

    t0 = time.perf_counter()
    results = []
    with Pool(150) as pool:
        done = 0
        for row in pool.imap_unordered(run_one_trial, configs):
            results.append(row)
            done += 1
            if done % 100 == 0 or done == total:
                el = time.perf_counter() - t0
                eta = el / done * (total - done)
                print(f"  [{done}/{total}] {el:.0f}s  ETA {eta:.0f}s",
                      flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s")

    with open('results/maxsat_ils_n100.json', 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer) else
                  float(o) if isinstance(o, np.floating) else
                  bool(o) if isinstance(o, np.bool_) else None)

    # Summary
    grp = defaultdict(lambda: defaultdict(list))
    for r in results:
        a = r['alpha']
        for name, m in r['methods'].items():
            grp[a][name].append(m)

    methods = ['ORC+Pert', 'Random-ILS', 'MinGap+Pert', 'RR-HC']
    print(f"\n{'='*100}")
    print(f"{'alpha':>6}", end='')
    for m in methods:
        print(f"  {m:>12}({' steps':>6})", end='')
    print()
    print(f"{'-'*100}")

    for a in alphas:
        print(f"{a:>6.2f}", end='')
        for m in methods:
            rows = grp[a][m]
            mean_fit = np.mean([r['best_fit'] for r in rows])
            mean_steps = np.mean([r['steps'] for r in rows])
            print(f"  {mean_fit:>12.2f}({mean_steps:>6.0f})", end='')
        print()

    # SAT success rate for underconstrained instances
    print(f"\nSAT success rate (% finding 0 unsat):")
    print(f"{'alpha':>6}", end='')
    for m in methods:
        print(f"  {m:>12}", end='')
    print()
    for a in alphas:
        print(f"{a:>6.2f}", end='')
        for m in methods:
            rows = grp[a][m]
            sat_rate = 100 * np.mean([r['is_sat'] for r in rows])
            print(f"  {sat_rate:>11.1f}%", end='')
        print()

    # Fitness improvement ratio (ORC vs others)
    print(f"\nMean best fitness (lower is better):")
    print(f"{'alpha':>6}  {'ORC+P':>7} {'Rand':>7} {'MG+P':>7} {'RRHC':>7}  "
          f"{'ORC/Rand':>9} {'ORC/MG':>8} {'ORC/RRHC':>9}")
    for a in alphas:
        fits = {}
        for m in methods:
            fits[m] = np.mean([r['best_fit'] for r in grp[a][m]])
        orc_f = fits['ORC+Pert']
        print(f"{a:>6.2f}  {orc_f:>7.2f} {fits['Random-ILS']:>7.2f} "
              f"{fits['MinGap+Pert']:>7.2f} {fits['RR-HC']:>7.2f}  "
              f"{orc_f/max(fits['Random-ILS'],.01):>9.3f} "
              f"{orc_f/max(fits['MinGap+Pert'],.01):>8.3f} "
              f"{orc_f/max(fits['RR-HC'],.01):>9.3f}")

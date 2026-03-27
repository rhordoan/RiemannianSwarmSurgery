#!/usr/bin/env python3
"""
ORC Transition Graph on QAP (Quadratic Assignment Problem).

Tests whether ORC landscape analysis generalizes from binary (hypercube)
to permutation (swap graph) search spaces, where adjacent nodes share
many common neighbors — unlike the hypercube's disjoint-neighborhood property.
"""

import numpy as np
from itertools import permutations as iter_perms
from scipy.optimize import linear_sum_assignment
from collections import Counter, defaultdict
import time, json, sys
from multiprocessing import Pool


def generate_qap(n, seed=0):
    rng = np.random.RandomState(seed)
    F = rng.randint(1, 100, size=(n, n)).astype(np.float64)
    D = rng.randint(1, 100, size=(n, n)).astype(np.float64)
    np.fill_diagonal(F, 0)
    np.fill_diagonal(D, 0)
    return F, D


def build_landscape(n, F, D):
    all_perms = list(iter_perms(range(n)))
    n_perms = len(all_perms)
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    perms_arr = np.array(all_perms, dtype=np.int32)

    fitness = np.zeros(n_perms, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if F[i, j] != 0:
                fitness += F[i, j] * D[perms_arr[:, i], perms_arr[:, j]]

    swap_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    k = len(swap_pairs)
    neighbors = np.zeros((n_perms, k), dtype=np.int32)

    for idx in range(n_perms):
        p = list(all_perms[idx])
        for s, (i, j) in enumerate(swap_pairs):
            p[i], p[j] = p[j], p[i]
            neighbors[idx, s] = perm_to_idx[tuple(p)]
            p[i], p[j] = p[j], p[i]

    nbr_fits = fitness[neighbors]
    is_opt = np.all(nbr_fits >= fitness[:, None], axis=1)
    opt_indices = np.where(is_opt)[0]

    return fitness, neighbors, opt_indices


def hill_climb(start, fitness, neighbors):
    cur = start
    while True:
        nbrs = neighbors[cur]
        best = nbrs[np.argmin(fitness[nbrs])]
        if fitness[best] >= fitness[cur]:
            return cur
        cur = best


def compute_orc(opt_idx, fitness, neighbors, gamma=1.0):
    nbrs = neighbors[opt_idx]
    k = len(nbrs)
    supp_x = set([int(opt_idx)]) | set(nbrs.tolist())
    supp_x_list = [int(opt_idx)] + nbrs.tolist()
    n_supp = k + 1

    f_x = fitness[np.array(supp_x_list)]

    results = {}
    for y_idx_raw in nbrs:
        y_idx = int(y_idx_raw)
        y_nbrs = neighbors[y_idx]
        supp_y = set([y_idx]) | set(y_nbrs.tolist())
        supp_y_list = [y_idx] + y_nbrs.tolist()

        shared = supp_x & supp_y

        sx = np.array(supp_x_list)
        sy = np.array(supp_y_list)

        a_shared = np.array([a in shared for a in supp_x_list])
        b_shared = np.array([b in shared for b in supp_y_list])

        same = (sx[:, None] == sy[None, :])
        either_shared = a_shared[:, None] | b_shared[None, :]
        delta = np.where(same, 0.0, np.where(either_shared, 1.0, 2.0))

        f_y = fitness[np.array(supp_y_list)]
        fit_diff = np.abs(f_x[:, None] - f_y[None, :])

        cost = delta + gamma * fit_diff

        ri, ci = linear_sum_assignment(cost)
        W1 = cost[ri, ci].sum() / n_supp
        d_xy = 1.0 + gamma * abs(fitness[opt_idx] - fitness[y_idx])

        if d_xy < 1e-12:
            results[y_idx] = 0.0
        else:
            results[y_idx] = float(1.0 - W1 / d_xy)

    return results


def follow_chains(opt_indices, dest_map, opt_to_rank):
    terminal = {}
    for opt in opt_indices:
        o = int(opt)
        visited = set()
        cur = o
        while cur not in visited:
            visited.add(cur)
            nxt = dest_map.get(cur, cur)
            if nxt == cur:
                break
            cur = nxt
        terminal[o] = cur

    n_opt = len(opt_indices)
    ranks, top5 = [], 0
    for opt in opt_indices:
        o = int(opt)
        r = opt_to_rank.get(terminal[o], 1.0)
        ranks.append(r)
        if r < 0.05:
            top5 += 1

    n_attractors = len(set(terminal.values()))
    return {
        'mean_rank': float(np.mean(ranks)),
        'top5': top5 / n_opt,
        'compression': n_attractors / n_opt,
    }


def analyze_instance(args):
    n, seed, gamma = args
    t_start = time.perf_counter()

    F, D = generate_qap(n, seed)
    fitness, neighbors, opt_indices = build_landscape(n, F, D)

    n_opt = len(opt_indices)
    k = neighbors.shape[1]
    if n_opt < 3:
        return {'n': n, 'seed': seed, 'skip': True, 'n_optima': n_opt}

    opt_fits = fitness[opt_indices]
    order = np.argsort(opt_fits)
    rank = np.zeros(n_opt)
    rank[order] = np.arange(n_opt) / max(n_opt - 1, 1)
    opt_to_rank = {int(opt_indices[i]): rank[i] for i in range(n_opt)}

    # Neighborhood overlap analysis
    n_shared_total, n_edges = 0, 0
    sample = opt_indices[:min(50, n_opt)]
    for opt in sample:
        supp_x = set([int(opt)]) | set(neighbors[opt].tolist())
        for y in neighbors[opt][:5]:  # subsample edges
            supp_y = set([int(y)]) | set(neighbors[y].tolist())
            n_shared_total += len(supp_x & supp_y)
            n_edges += 1
    shared_frac = (n_shared_total / n_edges) / (k + 1) if n_edges > 0 else 0

    # --- ORC escape ---
    orc_dest, orc_better = {}, 0
    mean_orcs = []
    for opt in opt_indices:
        orc = compute_orc(opt, fitness, neighbors, gamma)
        min_nbr = min(orc, key=orc.get)
        dest = hill_climb(min_nbr, fitness, neighbors)
        orc_dest[int(opt)] = int(dest)
        if fitness[dest] < fitness[opt]:
            orc_better += 1
        mean_orcs.append(np.mean(list(orc.values())))

    # --- MinGap escape ---
    mg_dest, mg_better = {}, 0
    for opt in opt_indices:
        gaps = np.abs(fitness[neighbors[opt]] - fitness[opt])
        min_nbr = int(neighbors[opt, np.argmin(gaps)])
        dest = hill_climb(min_nbr, fitness, neighbors)
        mg_dest[int(opt)] = int(dest)
        if fitness[dest] < fitness[opt]:
            mg_better += 1

    # --- Random escape ---
    rng = np.random.RandomState(seed + 10000)
    rand_better = 0.0
    rand_trials = 30
    for opt in opt_indices:
        succ = sum(
            1 for _ in range(rand_trials)
            if fitness[hill_climb(int(neighbors[opt, rng.randint(k)]),
                                  fitness, neighbors)] < fitness[opt]
        )
        rand_better += succ / rand_trials

    # --- LON-d1 escape (mode destination from all k neighbors) ---
    lon_dest, lon_better = {}, 0
    for opt in opt_indices:
        dests = [hill_climb(int(nbr), fitness, neighbors)
                 for nbr in neighbors[opt]]
        mode = Counter(dests).most_common(1)[0][0]
        lon_dest[int(opt)] = int(mode)
        if fitness[mode] < fitness[opt]:
            lon_better += 1

    # --- Terminal attractor analysis ---
    otg_m = follow_chains(opt_indices, orc_dest, opt_to_rank)
    mg_m = follow_chains(opt_indices, mg_dest, opt_to_rank)
    lon_m = follow_chains(opt_indices, lon_dest, opt_to_rank)

    elapsed = time.perf_counter() - t_start

    return {
        'n': n, 'seed': seed, 'skip': False,
        'n_optima': n_opt, 'k': k,
        'shared_frac': float(shared_frac),
        'frac_orc': orc_better / n_opt,
        'frac_mg': mg_better / n_opt,
        'frac_rand': rand_better / n_opt,
        'frac_lon': lon_better / n_opt,
        'mean_orc': float(np.mean(mean_orcs)),
        'otg_rank': otg_m['mean_rank'], 'otg_top5': otg_m['top5'],
        'otg_comp': otg_m['compression'],
        'mg_rank': mg_m['mean_rank'], 'mg_top5': mg_m['top5'],
        'lon_rank': lon_m['mean_rank'], 'lon_top5': lon_m['top5'],
        'lon_comp': lon_m['compression'],
        'time_s': elapsed,
    }


if __name__ == '__main__':
    sizes = [6, 7, 8]
    n_instances = 20
    gamma = 1.0

    configs = [(n, seed, gamma) for n in sizes
               for seed in range(n_instances)]
    total = len(configs)
    print(f"QAP OTG experiment: {total} instances "
          f"(n={sizes}, {n_instances} each)")

    t0 = time.perf_counter()
    results = []
    with Pool(min(total, 60)) as pool:
        done = 0
        for row in pool.imap_unordered(analyze_instance, configs):
            results.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                el = time.perf_counter() - t0
                print(f"  [{done}/{total}] {el:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s")

    with open('results/qap_otg.json', 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (
                      np.floating, np.integer, np.bool_)) else None)

    # --- Summary ---
    valid = [r for r in results if not r.get('skip')]
    grp = defaultdict(list)
    for r in valid:
        grp[r['n']].append(r)

    hdr = (f"{'n':>3} {'#Opt':>6} {'k':>4} {'Shr%':>5} "
           f"{'ORC%':>5} {'MG%':>5} {'Rnd%':>5} {'LON%':>5} "
           f"{'ORC/R':>6} {'ORC/MG':>7} {'ORC/LON':>8} "
           f"{'OTG rk':>7} {'LON rk':>7} {'OTG t5':>7} {'LON t5':>7} "
           f"{'OTG c':>6} {'LON c':>6}")
    print(f"\n{'=' * len(hdr)}")
    print(hdr)
    print(f"{'-' * len(hdr)}")

    for n in sizes:
        rows = grp.get(n, [])
        if not rows:
            continue
        no = np.mean([r['n_optima'] for r in rows])
        kk = rows[0]['k']
        sf = 100 * np.mean([r['shared_frac'] for r in rows])
        ob = 100 * np.mean([r['frac_orc'] for r in rows])
        mb = 100 * np.mean([r['frac_mg'] for r in rows])
        rb = 100 * np.mean([r['frac_rand'] for r in rows])
        lb = 100 * np.mean([r['frac_lon'] for r in rows])
        or_r = ob / max(rb, 0.1)
        or_m = ob / max(mb, 0.1)
        or_l = ob / max(lb, 0.1)
        otg_rk = np.mean([r['otg_rank'] for r in rows])
        lon_rk = np.mean([r['lon_rank'] for r in rows])
        otg_t5 = 100 * np.mean([r['otg_top5'] for r in rows])
        lon_t5 = 100 * np.mean([r['lon_top5'] for r in rows])
        otg_c = 100 * np.mean([r['otg_comp'] for r in rows])
        lon_c = 100 * np.mean([r['lon_comp'] for r in rows])
        print(f"{n:>3} {no:>6.0f} {kk:>4} {sf:>4.1f}% "
              f"{ob:>4.1f}% {mb:>4.1f}% {rb:>4.1f}% {lb:>4.1f}% "
              f"{or_r:>5.2f}x {or_m:>6.2f}x {or_l:>7.2f}x "
              f"{otg_rk:>7.3f} {lon_rk:>7.3f} {otg_t5:>6.1f}% {lon_t5:>6.1f}% "
              f"{otg_c:>5.0f}% {lon_c:>5.0f}%")

    # Key question: OTG vs LON-d1 quality ratio
    print(f"\n--- KEY QUESTION: Does OTG outperform LON-d1 on permutations? ---")
    for n in sizes:
        rows = grp.get(n, [])
        if not rows:
            continue
        otg_rk = np.mean([r['otg_rank'] for r in rows])
        lon_rk = np.mean([r['lon_rank'] for r in rows])
        ratio = lon_rk / max(otg_rk, 0.001)
        shared = 100 * np.mean([r['shared_frac'] for r in rows])
        print(f"  n={n}: OTG rank={otg_rk:.3f}, LON-d1 rank={lon_rk:.3f}, "
              f"LON/OTG={ratio:.1f}x better, shared={shared:.0f}%")

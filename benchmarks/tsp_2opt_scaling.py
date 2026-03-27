"""
Sampling-based TSP 2-opt ORC analysis at scale (n=15..50).

Since 2-opt neighborhoods satisfy the disjoint-neighborhood property
(Corollary 1(iii)), we use the sort-and-match ORC reduction:
per-edge cost O(k log k) instead of O(k^3) Hungarian.
"""

import numpy as np
import json
import os
import time
from multiprocessing import Pool
from collections import defaultdict


def tour_cost(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))


def two_opt_delta(tour, i, j, dist):
    """O(1) cost change of reversing segment [i+1..j]."""
    n = len(tour)
    a, b = tour[i], tour[i + 1]
    c, d = tour[j], tour[(j + 1) % n]
    return dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]


def two_opt_swap(tour, i, j):
    """Return new tour with segment [i+1..j] reversed."""
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


def valid_2opt_pairs(n):
    """All valid (i,j) pairs for 2-opt moves on an n-city tour."""
    pairs = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            pairs.append((i, j))
    return pairs


def hill_climb_2opt(tour, dist):
    """Steepest-descent 2-opt hill climbing."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_ij = None
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                delta = two_opt_delta(tour, i, j, dist)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_ij = (i, j)
        if best_ij is not None:
            tour = two_opt_swap(tour, *best_ij)
            improved = True
    return tuple(tour)


def all_neighbor_fitness(tour, dist, pairs):
    """Compute fitness of all 2-opt neighbors using O(1) deltas."""
    f0 = tour_cost(tour, dist)
    return [f0 + two_opt_delta(tour, i, j, dist) for i, j in pairs]


def compute_all_orc(opt_tour, dist, pairs, gamma=1.0):
    """Compute ORC for all neighbors of opt using sort-and-match.

    The reverse 2-opt move (nbr back to opt) is at the same positional index
    because 2opt(2opt(T,i,j), i,j) = T. This avoids expensive tuple comparisons.
    """
    k = len(pairs)
    f_opt = tour_cost(opt_tour, dist)
    opt_nbr_fitness = all_neighbor_fitness(opt_tour, dist, pairs)

    orc_vals = np.empty(k)
    for nbr_idx in range(k):
        i0, j0 = pairs[nbr_idx]
        nbr_tour = two_opt_swap(list(opt_tour), i0, j0)
        f_nbr = opt_nbr_fitness[nbr_idx]
        nbr_nbr_fitness = all_neighbor_fitness(nbr_tour, dist, pairs)

        opt_excl = sorted(v for idx2, v in enumerate(opt_nbr_fitness) if idx2 != nbr_idx)
        # Reverse move is at same positional index
        nbr_excl = sorted(v for idx2, v in enumerate(nbr_nbr_fitness) if idx2 != nbr_idx)

        C_shared = 2 * (1 + gamma * abs(f_opt - f_nbr))
        excl_cost = sum(2 + gamma * abs(a - b) for a, b in zip(opt_excl, nbr_excl))
        W1 = (C_shared + excl_cost) / (k + 1)
        d_xy = 1 + gamma * abs(f_opt - f_nbr)
        orc_vals[nbr_idx] = 1.0 - W1 / d_xy if d_xy > 0 else 0.0

    return orc_vals


def analyze_instance(n_cities, seed, n_restarts=500, instance_type='random'):
    """Sampling-based ORC analysis for one TSP instance."""
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if instance_type == 'euclidean':
        coords = rng.uniform(0, 100, size=(n_cities, 2))
        dist = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                dist[i, j] = d
                dist[j, i] = d
    else:
        dist = rng.uniform(1, 100, size=(n_cities, n_cities))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)

    pairs = valid_2opt_pairs(n_cities)
    k = len(pairs)

    # Sample local optima via random-restart 2-opt
    optima_set = set()
    optima_list = []
    for _ in range(n_restarts):
        init_tour = list(range(n_cities))
        rng.shuffle(init_tour)
        opt = hill_climb_2opt(init_tour, dist)
        if opt not in optima_set:
            optima_set.add(opt)
            optima_list.append(opt)

    n_opt = len(optima_list)
    if n_opt < 2:
        return None

    fitness = {t: tour_cost(t, dist) for t in optima_list}
    sorted_optima = sorted(optima_list, key=lambda t: fitness[t])
    rank_of = {t: i / n_opt for i, t in enumerate(sorted_optima)}
    global_opt = sorted_optima[0]

    t_sample = time.time() - t0

    orc_esc = 0
    mg_esc = 0
    rand_esc = 0
    total = 0
    orc_better = 0
    mg_better = 0
    agree = 0
    disagree = 0

    for opt_tour in optima_list:
        if opt_tour == global_opt:
            continue
        total += 1
        f_opt = fitness[opt_tour]
        opt_list = list(opt_tour)

        # Fitness of all k neighbors (O(k) via delta)
        nbr_fitness = all_neighbor_fitness(opt_list, dist, pairs)

        # ORC for all neighbors
        orc_vals = compute_all_orc(opt_list, dist, pairs, gamma=1.0)

        # Select min-ORC neighbor
        min_orc_idx = int(np.argmin(orc_vals))
        # Select min-gap neighbor
        min_gap_idx = min(range(k), key=lambda i: abs(nbr_fitness[i] - f_opt))

        # Hill-climb only the selected neighbors
        orc_nbr = two_opt_swap(opt_list, *pairs[min_orc_idx])
        orc_dest = hill_climb_2opt(orc_nbr, dist)
        f_orc_dest = tour_cost(orc_dest, dist)
        if f_orc_dest < f_opt - 1e-10:
            orc_esc += 1

        mg_nbr = two_opt_swap(opt_list, *pairs[min_gap_idx])
        mg_dest = hill_climb_2opt(mg_nbr, dist)
        f_mg_dest = tour_cost(mg_dest, dist)
        if f_mg_dest < f_opt - 1e-10:
            mg_esc += 1

        # Track disagreements
        if min_orc_idx == min_gap_idx:
            agree += 1
        else:
            disagree += 1
            if f_orc_dest < f_mg_dest - 1e-10:
                orc_better += 1
            elif f_mg_dest < f_orc_dest - 1e-10:
                mg_better += 1

        # Random baseline (30 trials, hill-climb each)
        r_esc = 0
        for _ in range(30):
            rand_idx = rng.randint(k)
            rand_nbr = two_opt_swap(opt_list, *pairs[rand_idx])
            rand_dest = hill_climb_2opt(rand_nbr, dist)
            f_rand_dest = tour_cost(rand_dest, dist)
            if f_rand_dest < f_opt - 1e-10:
                r_esc += 1
        rand_esc += r_esc / 30.0

    elapsed = time.time() - t0
    result = {
        'n_cities': n_cities,
        'seed': seed,
        'instance_type': instance_type,
        'n_restarts': n_restarts,
        'n_local_optima': n_opt,
        'k': k,
        'total_tested': total,
        'orc_esc': orc_esc / total if total > 0 else 0,
        'mg_esc': mg_esc / total if total > 0 else 0,
        'rand_esc': rand_esc / total if total > 0 else 0,
        'agree': agree,
        'disagree': disagree,
        'orc_wins_disagree': orc_better,
        'mg_wins_disagree': mg_better,
        'sample_time_s': round(t_sample, 1),
        'elapsed_s': round(elapsed, 1),
    }
    orc_r = result['orc_esc'] / result['rand_esc'] if result['rand_esc'] > 0 else float('inf')
    print(f"  n={n_cities} seed={seed}: {n_opt} opt, "
          f"ORC={result['orc_esc']:.1%} MG={result['mg_esc']:.1%} "
          f"Rand={result['rand_esc']:.1%} ORC/R={orc_r:.1f}x | "
          f"sample={t_sample:.0f}s total={elapsed:.0f}s",
          flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cities', type=int, nargs='+', default=[15, 20, 30])
    parser.add_argument('--n_instances', type=int, default=20)
    parser.add_argument('--n_restarts', type=int, default=500)
    parser.add_argument('--n_cores', type=int, default=80)
    args = parser.parse_args()

    tasks = []
    for n in args.n_cities:
        for seed in range(args.n_instances):
            tasks.append((n, seed, args.n_restarts, 'random'))

    print(f"Running {len(tasks)} TSP 2-opt scaling analyses ({args.n_cores} cores)...")
    with Pool(args.n_cores) as pool:
        results = pool.starmap(analyze_instance, tasks)

    results = [r for r in results if r is not None]

    groups = defaultdict(list)
    for r in results:
        groups[r['n_cities']].append(r)

    print(f"\n{'=' * 90}")
    print(f"TSP 2-opt Scaling Results (Sampling-Based)")
    print(f"{'=' * 90}")
    print(f"{'n':>4} {'#Opt':>6} {'k':>5} | "
          f"{'%ORC':>7} {'%MG':>7} {'%Rand':>7} {'ORC/R':>7} {'ORC/MG':>7} | "
          f"{'Agree':>6} {'ORCwin':>7} {'MGwin':>6}")
    print(f"{'-' * 90}")

    for n in sorted(groups.keys()):
        rows = groups[n]
        n_opt = np.mean([r['n_local_optima'] for r in rows])
        k_val = rows[0]['k']
        orc_e = np.mean([r['orc_esc'] for r in rows])
        mg_e = np.mean([r['mg_esc'] for r in rows])
        rand_e = np.mean([r['rand_esc'] for r in rows])
        orc_r = orc_e / rand_e if rand_e > 0 else float('inf')
        orc_mg = orc_e / mg_e if mg_e > 0 else float('inf')
        agree_total = sum(r['agree'] for r in rows)
        dis_total = sum(r['disagree'] for r in rows)
        orc_w = sum(r['orc_wins_disagree'] for r in rows)
        mg_w = sum(r['mg_wins_disagree'] for r in rows)
        total_opts = agree_total + dis_total
        print(f"{n:>4} {n_opt:>6.0f} {k_val:>5} | "
              f"{100 * orc_e:>6.1f}% {100 * mg_e:>6.1f}% {100 * rand_e:>6.1f}% "
              f"{orc_r:>6.1f}x {orc_mg:>6.2f}x | "
              f"{100 * agree_total / max(total_opts, 1):>5.0f}% "
              f"{100 * orc_w / max(dis_total, 1):>6.0f}% "
              f"{100 * mg_w / max(dis_total, 1):>5.0f}%")

    outpath = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'tsp_2opt_scaling.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()

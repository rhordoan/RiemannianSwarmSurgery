"""
TSP 2-opt ORC experiment: test ORC on non-disjoint neighborhoods.

TSP with 2-opt neighborhood does NOT satisfy the disjoint-neighborhood property
(adjacent tours share common neighbors). This tests whether ORC still provides
useful escape direction information beyond Proposition 1's guarantee.

Usage:
    python tsp_2opt_experiment.py [--n_cities 8 9 10] [--n_instances 20] [--n_cores 160]
"""

import numpy as np
import itertools
import json
import sys
import os
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from multiprocessing import Pool
import time


def generate_tsp_instance(n_cities, seed, instance_type='euclidean'):
    rng = np.random.RandomState(seed)
    if instance_type == 'euclidean':
        coords = rng.uniform(0, 100, size=(n_cities, 2))
        dist = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                dist[i, j] = d
                dist[j, i] = d
    else:
        dist = rng.uniform(1, 100, size=(n_cities, n_cities))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        coords = None
    return coords, dist


def tour_cost(tour, dist):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1) % n]] for i in range(n))


def canonical_tour(tour):
    """Canonical form: start at city 0, second city < last city."""
    n = len(tour)
    idx = list(tour).index(0)
    rotated = tuple(tour[idx:] + tour[:idx])
    if rotated[1] > rotated[-1]:
        rotated = (rotated[0],) + tuple(reversed(rotated[1:]))
    return rotated


def two_opt_neighbors(tour):
    """Generate all 2-opt neighbors of a tour."""
    n = len(tour)
    neighbors = []
    tour = list(tour)
    for i in range(n):
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
            neighbors.append(canonical_tour(tuple(new_tour)))
    return neighbors


def enumerate_all_tours(n_cities):
    """Enumerate all unique tours for n cities."""
    cities = list(range(n_cities))
    tours = set()
    for perm in itertools.permutations(cities[1:]):
        tour = (0,) + perm
        ct = canonical_tour(tour)
        tours.add(ct)
    return sorted(tours)


def compute_orc_edge(x_idx, y_idx, all_neighbors, fitness, tour_to_idx):
    """Compute ORC for edge (x, y) using the standard formulation with
    support overlap (no disjoint-neighborhood assumption)."""
    x_support = [x_idx] + [tour_to_idx[n] for n in all_neighbors[x_idx]]
    y_support = [y_idx] + [tour_to_idx[n] for n in all_neighbors[y_idx]]
    x_support_set = set(x_support)
    y_support_set = set(y_support)

    nx = len(x_support)
    ny = len(y_support)
    assert nx == ny, f"Support sizes differ: {nx} vs {ny}"
    k1 = nx

    gamma = 1.0
    cost_matrix = np.zeros((k1, k1))
    for i, a in enumerate(x_support):
        for j, b in enumerate(y_support):
            if a == b:
                delta = 0.0
            elif a in y_support_set or b in x_support_set:
                delta = 1.0
            else:
                delta = 2.0
            cost_matrix[i, j] = delta + gamma * abs(fitness[a] - fitness[b])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    W1 = cost_matrix[row_ind, col_ind].sum() / k1

    edge_cost = cost_matrix[0, list(y_support).index(x_idx)] if x_idx in y_support_set else 2.0 + gamma * abs(fitness[x_idx] - fitness[y_idx])
    if x_idx in y_support_set and y_idx in x_support_set:
        dx_y = 1.0 + gamma * abs(fitness[x_idx] - fitness[y_idx])
    else:
        dx_y = 2.0 + gamma * abs(fitness[x_idx] - fitness[y_idx])

    kappa = 1.0 - W1 / dx_y if dx_y > 0 else 0.0
    return kappa


def hill_climb(start_idx, fitness, all_neighbors, tour_to_idx):
    current = start_idx
    while True:
        nbrs = [tour_to_idx[n] for n in all_neighbors[current]]
        best = current
        best_fit = fitness[current]
        for nb in nbrs:
            if fitness[nb] < best_fit:
                best = nb
                best_fit = fitness[nb]
        if best == current:
            return current
        current = best


def analyze_instance(n_cities, seed, instance_type='euclidean'):
    """Full analysis of one TSP instance."""
    t0 = time.time()
    coords, dist = generate_tsp_instance(n_cities, seed, instance_type)
    all_tours = enumerate_all_tours(n_cities)
    n_tours = len(all_tours)
    tour_to_idx = {t: i for i, t in enumerate(all_tours)}

    fitness = np.array([tour_cost(t, dist) for t in all_tours])

    all_neighbors_tours = [two_opt_neighbors(t) for t in all_tours]
    all_neighbors_idx = [[tour_to_idx[n] for n in nbrs] for nbrs in all_neighbors_tours]

    nbr_size = len(all_neighbors_tours[0])

    local_optima = []
    for i in range(n_tours):
        is_opt = all(fitness[i] <= fitness[nb] for nb in all_neighbors_idx[i])
        if is_opt:
            local_optima.append(i)
    n_opt = len(local_optima)

    if n_opt < 2:
        return None

    sorted_optima = sorted(local_optima, key=lambda o: fitness[o])
    rank_of = {o: i / n_opt for i, o in enumerate(sorted_optima)}
    global_opt = sorted_optima[0]

    overlap_counts = []
    orc_esc = 0; mg_esc = 0; rand_esc = 0; total = 0
    orc_ranks = []; mg_ranks = []; rand_ranks = []
    orc_better_on_disagree = 0; mg_better_on_disagree = 0; tie_on_disagree = 0; agree = 0

    otg_edges = {}

    for opt in local_optima:
        if opt == global_opt:
            continue
        total += 1
        nbrs_tours = all_neighbors_tours[opt]
        nbrs_idx = all_neighbors_idx[opt]

        overlaps = []
        for nb_tour in nbrs_tours:
            nb_idx = tour_to_idx[nb_tour]
            x_supp = set([opt] + [tour_to_idx[n] for n in nbrs_tours])
            y_supp = set([nb_idx] + [tour_to_idx[n] for n in all_neighbors_tours[nb_idx]])
            overlap = len(x_supp & y_supp) - 2
            overlaps.append(overlap)
        overlap_counts.extend(overlaps)

        orc_vals = {}
        for nb_tour in nbrs_tours:
            nb_idx = tour_to_idx[nb_tour]
            kappa = compute_orc_edge(opt, nb_idx, all_neighbors_tours, fitness, tour_to_idx)
            orc_vals[nb_idx] = kappa

        min_orc_nbr = min(orc_vals, key=orc_vals.get)
        orc_dest = hill_climb(min_orc_nbr, fitness, all_neighbors_tours, tour_to_idx)
        orc_ranks.append(rank_of.get(orc_dest, 1.0))
        if orc_dest != opt and fitness[orc_dest] < fitness[opt]:
            orc_esc += 1
        otg_edges[opt] = orc_dest

        min_gap_nbr = min(nbrs_idx, key=lambda n: abs(fitness[n] - fitness[opt]))
        mg_dest = hill_climb(min_gap_nbr, fitness, all_neighbors_tours, tour_to_idx)
        mg_ranks.append(rank_of.get(mg_dest, 1.0))
        if mg_dest != opt and fitness[mg_dest] < fitness[opt]:
            mg_esc += 1

        if min_orc_nbr == min_gap_nbr:
            agree += 1
        else:
            if fitness[orc_dest] < fitness[mg_dest]:
                orc_better_on_disagree += 1
            elif fitness[mg_dest] < fitness[orc_dest]:
                mg_better_on_disagree += 1
            else:
                tie_on_disagree += 1

        rng = np.random.RandomState(seed * 10000 + opt)
        r_ranks = []
        r_esc = 0
        for _ in range(30):
            rand_nbr = nbrs_idx[rng.randint(len(nbrs_idx))]
            rand_dest = hill_climb(rand_nbr, fitness, all_neighbors_tours, tour_to_idx)
            r_ranks.append(rank_of.get(rand_dest, 1.0))
            if rand_dest != opt and fitness[rand_dest] < fitness[opt]:
                r_esc += 1
        rand_ranks.append(np.mean(r_ranks))
        rand_esc += r_esc / 30.0

    lon_d1_ranks = []
    lon_d1_edges = {}
    for opt in local_optima:
        if opt == global_opt:
            continue
        nbrs_idx = all_neighbors_idx[opt]
        dest_counts = defaultdict(int)
        for nb in nbrs_idx:
            dest = hill_climb(nb, fitness, all_neighbors_tours, tour_to_idx)
            dest_counts[dest] += 1
        mode_dest = max(dest_counts, key=dest_counts.get)
        lon_d1_ranks.append(rank_of.get(mode_dest, 1.0))
        lon_d1_edges[opt] = mode_dest

    def follow_chain(edges, start, max_steps=100):
        visited = set()
        current = start
        for _ in range(max_steps):
            if current in visited or current not in edges:
                return current
            visited.add(current)
            current = edges[current]
        return current

    otg_terminal_ranks = []
    for opt in local_optima:
        if opt == global_opt:
            continue
        terminal = follow_chain(otg_edges, opt)
        otg_terminal_ranks.append(rank_of.get(terminal, 1.0))

    lon_terminal_ranks = []
    for opt in local_optima:
        if opt == global_opt:
            continue
        terminal = follow_chain(lon_d1_edges, opt)
        lon_terminal_ranks.append(rank_of.get(terminal, 1.0))

    disagree = total - agree
    elapsed = time.time() - t0

    result = {
        'n_cities': n_cities,
        'seed': seed,
        'instance_type': instance_type,
        'n_tours': n_tours,
        'n_local_optima': n_opt,
        'nbr_size': nbr_size,
        'mean_overlap': float(np.mean(overlap_counts)) if overlap_counts else 0,
        'max_overlap': int(np.max(overlap_counts)) if overlap_counts else 0,
        'total_optima_tested': total,
        'orc_escape_rate': orc_esc / total if total > 0 else 0,
        'mg_escape_rate': mg_esc / total if total > 0 else 0,
        'rand_escape_rate': rand_esc / total if total > 0 else 0,
        'orc_mean_rank': float(np.mean(orc_ranks)) if orc_ranks else 0,
        'mg_mean_rank': float(np.mean(mg_ranks)) if mg_ranks else 0,
        'rand_mean_rank': float(np.mean(rand_ranks)) if rand_ranks else 0,
        'agree_frac': agree / total if total > 0 else 0,
        'orc_wins_on_disagree': orc_better_on_disagree,
        'mg_wins_on_disagree': mg_better_on_disagree,
        'tie_on_disagree': tie_on_disagree,
        'disagree_total': disagree,
        'otg_mean_terminal_rank': float(np.mean(otg_terminal_ranks)) if otg_terminal_ranks else 0,
        'lon_d1_mean_terminal_rank': float(np.mean(lon_terminal_ranks)) if lon_terminal_ranks else 0,
        'otg_frac_top5': float(np.mean([1 if r <= 0.05 else 0 for r in otg_terminal_ranks])) if otg_terminal_ranks else 0,
        'lon_d1_frac_top5': float(np.mean([1 if r <= 0.05 else 0 for r in lon_terminal_ranks])) if lon_terminal_ranks else 0,
        'elapsed_s': round(elapsed, 1),
    }
    print(f"  n={n_cities} seed={seed}: {n_opt} optima, "
          f"ORC esc={result['orc_escape_rate']:.1%} MG={result['mg_escape_rate']:.1%} "
          f"Rand={result['rand_escape_rate']:.1%} | "
          f"overlap={result['mean_overlap']:.1f} | {elapsed:.1f}s", flush=True)
    return result


def run_experiment(n_cities_list, n_instances, n_cores, instance_types=None):
    if instance_types is None:
        instance_types = ['euclidean', 'random']
    tasks = []
    for itype in instance_types:
        for n in n_cities_list:
            for seed in range(n_instances):
                tasks.append((n, seed, itype))

    print(f"Running {len(tasks)} TSP 2-opt analyses across {n_cores} cores...")

    with Pool(n_cores) as pool:
        results = pool.starmap(analyze_instance, tasks)

    results = [r for r in results if r is not None]

    groups = defaultdict(list)
    for r in results:
        groups[(r['instance_type'], r['n_cities'])].append(r)

    print(f"\n{'='*110}")
    print(f"TSP 2-opt ORC Results")
    print(f"{'='*110}")
    print(f"{'Type':<10} {'n':>3} {'#Opt':>6} {'k':>4} {'Ovlp':>5} | "
          f"{'%ORC':>6} {'%MG':>6} {'%Rand':>6} {'ORC/R':>7} {'ORC/MG':>7} | "
          f"{'Agree':>6} {'ORC wins':>9} {'MG wins':>8}")
    print(f"{'-'*110}")

    for key in sorted(groups.keys()):
        itype, n = key
        rows = groups[key]
        n_opt = np.mean([r['n_local_optima'] for r in rows])
        k = rows[0]['nbr_size']
        overlap = np.mean([r['mean_overlap'] for r in rows])
        orc_e = np.mean([r['orc_escape_rate'] for r in rows])
        mg_e = np.mean([r['mg_escape_rate'] for r in rows])
        rand_e = np.mean([r['rand_escape_rate'] for r in rows])
        orc_r = orc_e / rand_e if rand_e > 0 else float('inf')
        orc_mg = orc_e / mg_e if mg_e > 0 else float('inf')
        agree = np.mean([r['agree_frac'] for r in rows])
        orc_w = sum(r['orc_wins_on_disagree'] for r in rows)
        mg_w = sum(r['mg_wins_on_disagree'] for r in rows)
        tie_w = sum(r['tie_on_disagree'] for r in rows)
        dis_total = sum(r['disagree_total'] for r in rows)

        print(f"{itype:<10} {n:>3} {n_opt:>6.0f} {k:>4} {overlap:>5.1f} | "
              f"{100*orc_e:>5.1f}% {100*mg_e:>5.1f}% {100*rand_e:>5.1f}% "
              f"{orc_r:>6.1f}x {orc_mg:>6.2f}x | "
              f"{100*agree:>5.1f}% "
              f"{100*orc_w/max(dis_total,1):>8.1f}% {100*mg_w/max(dis_total,1):>7.1f}%")

    print(f"\n{'='*100}")
    print(f"OTG vs LON-d1 Terminal Attractor Quality (Multi-hop)")
    print(f"{'='*100}")
    print(f"{'Type':<10} {'n':>3} | {'OTG rank':>9} {'LON rank':>9} {'OTG/LON':>8} | "
          f"{'OTG top5%':>9} {'LON top5%':>9}")
    print(f"{'-'*80}")

    for key in sorted(groups.keys()):
        itype, n = key
        rows = groups[key]
        otg_r = np.mean([r['otg_mean_terminal_rank'] for r in rows])
        lon_r = np.mean([r['lon_d1_mean_terminal_rank'] for r in rows])
        otg_t5 = np.mean([r['otg_frac_top5'] for r in rows])
        lon_t5 = np.mean([r['lon_d1_frac_top5'] for r in rows])
        ratio = lon_r / otg_r if otg_r > 0 else float('inf')
        print(f"{itype:<10} {n:>3} | {otg_r:>9.3f} {lon_r:>9.3f} {ratio:>7.1f}x | "
              f"{100*otg_t5:>8.1f}% {100*lon_t5:>8.1f}%")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'tsp_2opt_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/tsp_2opt_results.json")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cities', type=int, nargs='+', default=[8, 9, 10])
    parser.add_argument('--n_instances', type=int, default=20)
    parser.add_argument('--n_cores', type=int, default=160)
    args = parser.parse_args()
    run_experiment(args.n_cities, args.n_instances, args.n_cores)

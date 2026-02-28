"""
Generic Discrete Ollivier-Ricci Curvature for Fitness Landscape Analysis.

Computes ORC on any discrete search graph defined by:
  - A finite set of solutions indexed 0..N-1
  - A fitness array of shape (N,)
  - A neighbor function: idx -> list of neighbor indices

The fitness-lifted ORC detects transitions between basins of attraction:
negative curvature at a local optimum indicates a saddle direction where
the landscape geometry diverges.

References
----------
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


NeighborFn = Callable[[int], List[int]]


def compute_orc_edge(
    u: int,
    v: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    gamma: float = 1.0,
) -> float:
    """
    Compute Ollivier-Ricci Curvature for edge (u, v).

    Support sets are uniform distributions over {node} union neighbors(node).
    Cost matrix uses fitness-lifted graph distance:
        d_lift(a, b) = 1_{a != b} + gamma * |f(a) - f(b)|

    For neighbors, graph distance is 1 if they share an edge, else 2,
    but we approximate with 1 for simplicity (valid for k-regular graphs
    where support sets largely overlap).
    """
    nbrs_u = neighbor_fn(u)
    nbrs_v = neighbor_fn(v)
    sup_u = [u] + nbrs_u
    sup_v = [v] + nbrs_v

    n_u = len(sup_u)
    n_v = len(sup_v)

    sup_u_set = set(sup_u)
    sup_v_set = set(sup_v)

    fit_u = fitness[sup_u]
    fit_v = fitness[sup_v]

    C = np.empty((n_u, n_v))
    for i, a in enumerate(sup_u):
        for j, b in enumerate(sup_v):
            if a == b:
                C[i, j] = 0.0
            else:
                graph_d = 1.0 if (b in sup_u_set or a in sup_v_set) else 2.0
                C[i, j] = graph_d + gamma * abs(fit_u[i] - fit_v[j])

    d_uv = C[0, 0]
    if d_uv < 1e-12:
        return 0.0

    n = min(n_u, n_v)
    if n_u != n_v:
        C = C[:n, :n]

    row_ind, col_ind = linear_sum_assignment(C)
    W1 = float(np.sum(C[row_ind, col_ind])) / n

    return float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))


def compute_orc_neighborhood(
    center: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    gamma: float = 1.0,
) -> Dict[int, float]:
    """
    Compute ORC for all edges incident to center.
    Returns {neighbor_idx: orc_value}.
    """
    nbrs = neighbor_fn(center)
    sup_u = [center] + nbrs
    n = len(sup_u)
    sup_u_set = set(sup_u)
    fit_u = fitness[sup_u]

    orc_values = {}
    for nbr in nbrs:
        nbr_nbrs = neighbor_fn(nbr)
        sup_v = [nbr] + nbr_nbrs
        sup_v_set = set(sup_v)
        fit_v = fitness[sup_v]

        m = min(n, len(sup_v))
        C = np.empty((m, m))
        for i in range(m):
            a = sup_u[i]
            for j in range(m):
                b = sup_v[j]
                if a == b:
                    C[i, j] = 0.0
                else:
                    graph_d = 1.0 if (b in sup_u_set or a in sup_v_set) else 2.0
                    C[i, j] = graph_d + gamma * abs(fit_u[i] - fit_v[j])

        d_uv = C[0, 0]
        if d_uv < 1e-12:
            orc_values[nbr] = 0.0
            continue

        row_ind, col_ind = linear_sum_assignment(C)
        W1 = float(np.sum(C[row_ind, col_ind])) / m
        orc_values[nbr] = float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))

    return orc_values


def find_saddle_direction(
    center: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    gamma: float = 1.0,
) -> Tuple[Optional[int], float]:
    """
    Find the neighbor with the most negative ORC (strongest saddle).
    Returns (neighbor_idx, min_orc). If all ORC >= 0, returns (None, 0.0).
    """
    orc = compute_orc_neighborhood(center, fitness, neighbor_fn, gamma)
    if not orc:
        return None, 0.0

    min_nbr = min(orc, key=orc.get)
    min_orc = orc[min_nbr]

    if min_orc >= 0:
        return None, 0.0
    return min_nbr, min_orc


# ---------------------------------------------------------------------------
# Landscape analysis utilities
# ---------------------------------------------------------------------------

def is_local_optimum(idx: int, fitness: np.ndarray, neighbor_fn: NeighborFn) -> bool:
    """True iff no neighbor has strictly lower fitness."""
    f = fitness[idx]
    for nbr in neighbor_fn(idx):
        if fitness[nbr] < f:
            return False
    return True


def hill_climb(
    start: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
) -> int:
    """Greedy descent to local optimum. Ties broken by index."""
    current = start
    while True:
        nbrs = neighbor_fn(current)
        f = fitness[current]
        improving = [(n, fitness[n]) for n in nbrs if fitness[n] < f]
        if not improving:
            return current
        current = min(improving, key=lambda x: (x[1], x[0]))[0]


def find_all_local_optima(
    space_size: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
) -> List[int]:
    """Find all local optima by exhaustive scan."""
    return [i for i in range(space_size) if is_local_optimum(i, fitness, neighbor_fn)]


def compute_basin_sizes(
    space_size: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
) -> Dict[int, int]:
    """Map each local optimum to the number of solutions that hill-climb to it."""
    basins: Dict[int, int] = {}
    for i in range(space_size):
        opt = hill_climb(i, fitness, neighbor_fn)
        basins[opt] = basins.get(opt, 0) + 1
    return basins


def full_landscape_analysis(
    space_size: int,
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    gamma: float = 1.0,
    n_random_trials: int = 30,
    seed: int = 0,
) -> dict:
    """
    Complete ORC landscape analysis:
    - Find all local optima
    - Compute basin sizes
    - Compute ORC at each local optimum
    - Check if negative ORC leads to better basins
    - Compare against random-direction and worst-ORC-direction baselines
    """
    rng = np.random.RandomState(seed)
    local_optima = find_all_local_optima(space_size, fitness, neighbor_fn)
    basins = compute_basin_sizes(space_size, fitness, neighbor_fn)

    orc_analyses = []
    for opt in local_optima:
        orc_dict = compute_orc_neighborhood(opt, fitness, neighbor_fn, gamma)
        min_nbr = min(orc_dict, key=orc_dict.get)
        min_orc = orc_dict[min_nbr]
        max_nbr = max(orc_dict, key=orc_dict.get)
        max_orc = orc_dict[max_nbr]

        leads_to_better = False
        dest_opt = None
        if min_orc < 0:
            dest_opt = hill_climb(min_nbr, fitness, neighbor_fn)
            leads_to_better = fitness[dest_opt] < fitness[opt]

        worst_orc_leads_to_better = False
        worst_dest_opt = hill_climb(max_nbr, fitness, neighbor_fn)
        worst_orc_leads_to_better = fitness[worst_dest_opt] < fitness[opt]

        nbrs = list(orc_dict.keys())
        random_better_count = 0
        for _ in range(n_random_trials):
            rand_nbr = nbrs[rng.randint(len(nbrs))]
            rand_dest = hill_climb(rand_nbr, fitness, neighbor_fn)
            if fitness[rand_dest] < fitness[opt]:
                random_better_count += 1
        frac_random_better = random_better_count / n_random_trials

        orc_analyses.append({
            'opt_idx': opt,
            'opt_fitness': float(fitness[opt]),
            'min_orc': float(min_orc),
            'min_orc_neighbor': int(min_nbr),
            'max_orc': float(max_orc),
            'max_orc_neighbor': int(max_nbr),
            'has_negative_orc': min_orc < 0,
            'leads_to_better': leads_to_better,
            'worst_orc_leads_to_better': worst_orc_leads_to_better,
            'frac_random_better': frac_random_better,
            'dest_opt': dest_opt,
            'basin_size': basins.get(opt, 0),
            'orc_values': {int(k): float(v) for k, v in orc_dict.items()},
        })

    n_opt = len(local_optima)
    n_neg = sum(1 for a in orc_analyses if a['has_negative_orc'])
    n_better = sum(1 for a in orc_analyses if a['leads_to_better'])
    n_worst_better = sum(1 for a in orc_analyses if a['worst_orc_leads_to_better'])

    optima_with_neg = [a for a in orc_analyses if a['has_negative_orc']]
    mean_random_better = (
        float(np.mean([a['frac_random_better'] for a in optima_with_neg]))
        if optima_with_neg else 0.0
    )

    return {
        'space_size': space_size,
        'gamma': gamma,
        'n_local_optima': n_opt,
        'frac_with_negative_orc': n_neg / n_opt if n_opt else 0.0,
        'n_with_negative_orc': n_neg,
        'frac_leads_to_better': n_better / n_neg if n_neg else 0.0,
        'n_leads_to_better': n_better,
        'frac_random_leads_to_better': mean_random_better,
        'frac_worst_orc_leads_to_better': n_worst_better / n_neg if n_neg else 0.0,
        'local_optima': local_optima,
        'basin_sizes': basins,
        'orc_analyses': orc_analyses,
    }

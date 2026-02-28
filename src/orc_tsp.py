"""
ORC-TSP: Discrete Ollivier-Ricci Curvature for the Traveling Salesman Problem.

The 2-opt neighborhood defines a natural graph on the space of tours:
two tours are adjacent if one can be obtained from the other by reversing
a single segment. For N cities, each tour has N*(N-1)/2 - N = N*(N-3)/2
2-opt neighbors.

ORC on this graph measures the geometric relationship between basins of
attraction — negative curvature at a local optimum indicates a saddle
direction where the landscape transitions to a different basin.

This module provides:
  - Random TSP instance generation
  - 2-opt neighborhood enumeration
  - Tour distance metrics (edge symmetric difference)
  - ORC computation on the tour graph
  - Search algorithms: ILS, ORC-guided ILS
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# TSP basics
# ---------------------------------------------------------------------------

def generate_instance(n: int, seed: int = 0) -> np.ndarray:
    """Generate random Euclidean TSP instance. Returns (n, 2) city coordinates."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2)


def tour_length(cities: np.ndarray, tour: np.ndarray) -> float:
    """Total Euclidean tour length for a given permutation."""
    coords = cities[tour]
    diffs = np.diff(coords, axis=0, append=coords[:1])
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def tour_edges(tour: np.ndarray) -> FrozenSet[Tuple[int, int]]:
    """Set of undirected edges in the tour."""
    n = len(tour)
    edges = set()
    for k in range(n):
        a, b = int(tour[k]), int(tour[(k + 1) % n])
        edges.add((min(a, b), max(a, b)))
    return frozenset(edges)


def edge_distance(tour_a: np.ndarray, tour_b: np.ndarray) -> int:
    """Number of edges in tour_a not present in tour_b."""
    ea = tour_edges(tour_a)
    eb = tour_edges(tour_b)
    return len(ea - eb)


# ---------------------------------------------------------------------------
# 2-opt neighborhood
# ---------------------------------------------------------------------------

def two_opt_move(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply 2-opt: reverse segment tour[i:j+1]."""
    new = tour.copy()
    new[i:j + 1] = tour[i:j + 1][::-1]
    return new


def two_opt_delta(cities: np.ndarray, tour: np.ndarray, i: int, j: int) -> float:
    """Compute change in tour length from 2-opt(i, j) without building new tour."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    d_ab = np.linalg.norm(cities[a] - cities[b])
    d_cd = np.linalg.norm(cities[c] - cities[d])
    d_ac = np.linalg.norm(cities[a] - cities[c])
    d_bd = np.linalg.norm(cities[b] - cities[d])
    return (d_ac + d_bd) - (d_ab + d_cd)


def hill_climb_2opt(cities: np.ndarray, tour: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Steepest-descent 2-opt hill climbing until local optimum.
    Returns (best_tour, best_length, n_evals).
    """
    n = len(tour)
    current = tour.copy()
    current_len = tour_length(cities, current)
    evals = 0
    improved = True

    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                delta = two_opt_delta(cities, current, i, j)
                evals += 1
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_i >= 0:
            current = two_opt_move(current, best_i, best_j)
            current_len += best_delta
            improved = True

    return current, current_len, evals


def first_improvement_2opt(cities: np.ndarray, tour: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    First-improvement 2-opt hill climbing until local optimum.
    Returns (best_tour, best_length, n_evals).
    """
    n = len(tour)
    current = tour.copy()
    current_len = tour_length(cities, current)
    evals = 0
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                delta = two_opt_delta(cities, current, i, j)
                evals += 1
                if delta < -1e-10:
                    current = two_opt_move(current, i, j)
                    current_len += delta
                    improved = True
                    break
            if improved:
                break

    return current, current_len, evals


# ---------------------------------------------------------------------------
# ORC on the tour graph
# ---------------------------------------------------------------------------

def compute_orc_tsp(
    cities: np.ndarray,
    tour: np.ndarray,
    gamma: float = 1.0,
    k_sample: int = 40,
    seed: int = 0,
) -> List[Tuple[Tuple[int, int], float, float]]:
    """
    Compute ORC for a sample of 2-opt edges at the current tour.

    Samples k_sample 2-opt moves, computes ORC for each using
    fitness-lifted edge distance. Returns list of ((i, j), orc, delta)
    sorted by ORC ascending (most negative first).

    Parameters
    ----------
    cities    : (n, 2) coordinates
    tour      : current tour permutation
    gamma     : weight of fitness difference in lifted distance
    k_sample  : number of 2-opt moves to sample
    seed      : RNG seed for sampling
    """
    n = len(tour)
    rng = np.random.RandomState(seed)
    current_len = tour_length(cities, tour)

    all_moves = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]
    if len(all_moves) <= k_sample:
        sampled = all_moves
    else:
        idx = rng.choice(len(all_moves), size=k_sample, replace=False)
        sampled = [all_moves[k] for k in idx]

    center_nbrs_cache = {}

    def _get_nbr_fitnesses(t: np.ndarray, t_len: float, max_nbrs: int = 20) -> List[float]:
        """Get fitness values of a random subset of 2-opt neighbors."""
        key = t.tobytes()
        if key in center_nbrs_cache:
            return center_nbrs_cache[key]

        nbr_moves = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]
        rng_inner = np.random.RandomState(hash(key) % (2**31))
        if len(nbr_moves) > max_nbrs:
            sel = rng_inner.choice(len(nbr_moves), size=max_nbrs, replace=False)
            nbr_moves = [nbr_moves[k] for k in sel]

        fitnesses = []
        for mi, mj in nbr_moves:
            delta = two_opt_delta(cities, t, mi, mj)
            fitnesses.append(t_len + delta)
        center_nbrs_cache[key] = fitnesses
        return fitnesses

    center_fit = current_len
    center_nbr_fits = _get_nbr_fitnesses(tour, current_len)
    sup_u_fits = np.array([center_fit] + center_nbr_fits)
    n_sup = len(sup_u_fits)

    results = []
    for i, j in sampled:
        delta = two_opt_delta(cities, tour, i, j)
        nbr_tour = two_opt_move(tour, i, j)
        nbr_len = current_len + delta
        nbr_nbr_fits = _get_nbr_fitnesses(nbr_tour, nbr_len)
        sup_v_fits = np.array([nbr_len] + nbr_nbr_fits)

        m = min(len(sup_u_fits), len(sup_v_fits))
        su = sup_u_fits[:m]
        sv = sup_v_fits[:m]

        fit_diff = np.abs(su[:, None] - sv[None, :])
        C = 1.0 + gamma * fit_diff

        d_uv = 1.0 + gamma * abs(center_fit - nbr_len)
        if d_uv < 1e-12:
            results.append(((i, j), 0.0, delta))
            continue

        row_ind, col_ind = linear_sum_assignment(C)
        W1 = float(np.sum(C[row_ind, col_ind])) / m
        orc = float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))
        results.append(((i, j), orc, delta))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def random_restart_ils(
    cities: np.ndarray,
    budget: int,
    seed: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """
    Iterated Local Search with random double-bridge perturbation.
    Budget = total 2-opt evaluations.
    Returns (best_tour, best_length, total_evals).
    """
    n = len(cities)
    rng = np.random.RandomState(seed)
    total_evals = 0

    init_tour = np.arange(n)
    rng.shuffle(init_tour)
    best_tour, best_len, evals = first_improvement_2opt(cities, init_tour)
    total_evals += evals

    while total_evals < budget:
        perturbed = _double_bridge(best_tour, rng)
        local_tour, local_len, evals = first_improvement_2opt(cities, perturbed)
        total_evals += evals

        if local_len < best_len - 1e-10:
            best_tour = local_tour
            best_len = local_len

    return best_tour, best_len, total_evals


def orc_guided_ils(
    cities: np.ndarray,
    budget: int,
    gamma: float = 0.5,
    k_sample: int = 40,
    seed: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """
    ILS with ORC-guided perturbation at local optima.

    Instead of random double-bridge, uses ORC to find the most negative
    curvature direction (saddle crossing) at the current local optimum.
    Falls back to double-bridge if no negative ORC found.

    Budget = total 2-opt evaluations.
    """
    n = len(cities)
    rng = np.random.RandomState(seed)
    total_evals = 0
    orc_step = 0

    init_tour = np.arange(n)
    rng.shuffle(init_tour)
    best_tour, best_len, evals = first_improvement_2opt(cities, init_tour)
    total_evals += evals
    current_tour = best_tour.copy()
    current_len = best_len

    while total_evals < budget:
        orc_results = compute_orc_tsp(
            cities, current_tour, gamma=gamma,
            k_sample=k_sample, seed=seed + orc_step,
        )
        orc_step += 1
        total_evals += k_sample * 20

        negative_orc = [(mv, orc, d) for mv, orc, d in orc_results if orc < 0]

        if negative_orc:
            move, orc_val, delta = negative_orc[0]
            perturbed = two_opt_move(current_tour, move[0], move[1])
        else:
            perturbed = _double_bridge(current_tour, rng)

        local_tour, local_len, evals = first_improvement_2opt(cities, perturbed)
        total_evals += evals

        if local_len < best_len - 1e-10:
            best_tour = local_tour.copy()
            best_len = local_len

        current_tour = local_tour
        current_len = local_len

    return best_tour, best_len, total_evals


def random_restart_ls(
    cities: np.ndarray,
    budget: int,
    seed: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """
    Random restart local search (no ILS, just independent restarts).
    Budget = total 2-opt evaluations.
    """
    n = len(cities)
    rng = np.random.RandomState(seed)
    total_evals = 0
    best_tour = None
    best_len = float('inf')

    while total_evals < budget:
        init_tour = np.arange(n)
        rng.shuffle(init_tour)
        local_tour, local_len, evals = first_improvement_2opt(cities, init_tour)
        total_evals += evals

        if local_len < best_len - 1e-10:
            best_tour = local_tour.copy()
            best_len = local_len

    return best_tour, best_len, total_evals


# ---------------------------------------------------------------------------
# Perturbation operators
# ---------------------------------------------------------------------------

def _double_bridge(tour: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Double-bridge perturbation (standard ILS for TSP)."""
    n = len(tour)
    cuts = sorted(rng.choice(range(1, n), size=3, replace=False))
    a, b, c = cuts
    new_tour = np.concatenate([
        tour[:a], tour[b:c], tour[a:b], tour[c:]
    ])
    return new_tour

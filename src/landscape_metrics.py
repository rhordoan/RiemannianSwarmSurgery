"""
Classical Fitness Landscape Analysis Metrics.

Implements standard FLA metrics for comparison with ORC:
  - Fitness-Distance Correlation (FDC)
  - Autocorrelation length
  - Information content
  - Basin entropy
  - Number of local optima

All metrics operate on discrete landscapes defined by
(fitness array, neighbor function).

References
----------
Jones & Forrest (1995). Fitness Distance Correlation as a Measure of
    Problem Difficulty for Genetic Algorithms. ICGA.
Stadler (1996). Landscapes and their correlation functions.
Vassilev et al. (2000). Information characteristics and the structure
    of landscapes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


NeighborFn = Callable[[int], List[int]]


# ---------------------------------------------------------------------------
# Fitness-Distance Correlation (FDC)
# ---------------------------------------------------------------------------

def fitness_distance_correlation(
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    global_opt: int,
    n_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Pearson correlation between fitness and distance to global optimum.

    Samples n_samples random solutions, computes BFS distance to global_opt.
    For minimization landscapes (lower = better), strong negative FDC means
    fitness decreases as we approach the optimum (easy for hill climbing).
    """
    space_size = len(fitness)
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(space_size, size=min(n_samples, space_size), replace=False)

    distances = _bfs_distances(global_opt, neighbor_fn, space_size)

    f_vals = fitness[sample_idx]
    d_vals = distances[sample_idx].astype(float)

    if np.std(f_vals) < 1e-12 or np.std(d_vals) < 1e-12:
        return 0.0

    return float(np.corrcoef(f_vals, d_vals)[0, 1])


def _bfs_distances(source: int, neighbor_fn: NeighborFn, space_size: int) -> np.ndarray:
    """BFS shortest-path distances from source to all nodes."""
    dist = np.full(space_size, -1, dtype=np.int32)
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in neighbor_fn(u):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    n_walks: int = 100,
    walk_length: int = 200,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Estimate autocorrelation length from random walks.

    Performs n_walks random walks of walk_length steps, computes the
    autocorrelation function, and returns the correlation length
    (number of steps until autocorrelation drops below 1/e).

    Returns (correlation_length, autocorrelation_array).
    """
    space_size = len(fitness)
    rng = np.random.RandomState(seed)
    max_lag = min(50, walk_length // 2)

    all_acf = np.zeros(max_lag)
    count = 0

    for _ in range(n_walks):
        start = rng.randint(0, space_size)
        walk_fitness = np.empty(walk_length)
        current = start
        for step in range(walk_length):
            walk_fitness[step] = fitness[current]
            nbrs = neighbor_fn(current)
            current = nbrs[rng.randint(0, len(nbrs))]

        mean_f = walk_fitness.mean()
        var_f = walk_fitness.var()
        if var_f < 1e-12:
            continue

        centered = walk_fitness - mean_f
        for lag in range(max_lag):
            acf = np.mean(centered[:walk_length - lag] * centered[lag:]) / var_f
            all_acf[lag] += acf
        count += 1

    if count == 0:
        return 0.0, np.zeros(max_lag)

    all_acf /= count

    corr_length = max_lag
    threshold = 1.0 / np.e
    for lag in range(1, max_lag):
        if all_acf[lag] < threshold:
            corr_length = lag
            break

    return float(corr_length), all_acf


# ---------------------------------------------------------------------------
# Information Content
# ---------------------------------------------------------------------------

def information_content(
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    n_walks: int = 100,
    walk_length: int = 200,
    epsilon: float = 0.0,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Information content and partial information content (Vassilev et al. 2000).

    Measures the entropy of fitness change patterns along random walks.
    Returns (H, M):
      H = information content (entropy of sign sequence)
      M = partial information content (number of slope changes / walk length)

    Low H -> regular/predictable landscape
    High H -> random/rugged landscape
    """
    space_size = len(fitness)
    rng = np.random.RandomState(seed)

    pair_counts = np.zeros((3, 3))
    total_slope_changes = 0
    total_steps = 0

    for _ in range(n_walks):
        start = rng.randint(0, space_size)
        current = start
        prev_sign = None

        for step in range(walk_length):
            nbrs = neighbor_fn(current)
            next_node = nbrs[rng.randint(0, len(nbrs))]
            diff = fitness[next_node] - fitness[current]

            if diff > epsilon:
                sign = 1
            elif diff < -epsilon:
                sign = -1
            else:
                sign = 0

            if prev_sign is not None:
                pair_counts[prev_sign + 1, sign + 1] += 1
                if prev_sign != sign and prev_sign != 0 and sign != 0:
                    total_slope_changes += 1
                total_steps += 1

            prev_sign = sign
            current = next_node

    total = pair_counts.sum()
    if total < 1:
        return 0.0, 0.0

    probs = pair_counts / total
    H = 0.0
    for i in range(3):
        row_sum = probs[i].sum()
        if row_sum < 1e-12:
            continue
        for j in range(3):
            if probs[i, j] > 1e-12:
                cond_p = probs[i, j] / row_sum
                H -= probs[i, j] * np.log2(cond_p)

    M = total_slope_changes / total_steps if total_steps > 0 else 0.0

    return float(H), float(M)


# ---------------------------------------------------------------------------
# Basin entropy
# ---------------------------------------------------------------------------

def basin_entropy(basin_sizes: Dict[int, int]) -> float:
    """Shannon entropy of basin size distribution (normalized)."""
    sizes = np.array(list(basin_sizes.values()), dtype=float)
    total = sizes.sum()
    if total < 1:
        return 0.0
    probs = sizes / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Aggregate function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    fitness: np.ndarray,
    neighbor_fn: NeighborFn,
    global_opt: int,
    basin_sizes: Dict[int, int],
    n_local_optima: int,
    seed: int = 0,
) -> dict:
    """Compute all classical landscape metrics for one instance."""
    fdc = fitness_distance_correlation(fitness, neighbor_fn, global_opt, seed=seed)
    corr_len, acf = autocorrelation(fitness, neighbor_fn, seed=seed)
    ic_H, ic_M = information_content(fitness, neighbor_fn, seed=seed)
    b_entropy = basin_entropy(basin_sizes)

    return {
        'fdc': fdc,
        'autocorrelation_length': corr_len,
        'information_content_H': ic_H,
        'partial_information_content_M': ic_M,
        'basin_entropy': b_entropy,
        'n_local_optima': n_local_optima,
    }

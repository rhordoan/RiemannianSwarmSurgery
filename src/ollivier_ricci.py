"""
Ollivier-Ricci Curvature (ORC) for discrete graphs.

Reference:
    Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces.
    Journal of Functional Analysis, 256(3), 810-864.

    Lin, Y., Lu, L., & Yau, S. T. (2011). Ricci curvature of graphs.
    Tohoku Mathematical Journal, 63(4), 605-627.

ORC(u, v) = 1 - W1(mu_u, mu_v) / d(u, v)

where:
  - mu_u = uniform distribution over {u} union N(u)  (the node and its k neighbors)
  - mu_v = uniform distribution over {v} union N(v)
  - W1   = Wasserstein-1 / Earth Mover's Distance
  - d(u,v) = Euclidean distance between agents u and v in search space

Interpretation:
  ORC > 0  : u and v are inside the same dense cluster (intra-basin edge).
              Their neighborhoods overlap -- positive curvature, like a sphere.
  ORC ~ 0  : Neutral -- neither clustered nor separated.
  ORC < 0  : u and v are on opposite sides of a sparse region (inter-basin edge).
              Their neighborhoods diverge -- negative curvature, like a saddle.

Key properties relevant to the paper:
  - Bounded in [-1, 1] by construction (theorem, not a clamp). No blowup possible.
  - Zero extra function evaluations: built from the current swarm positions.
  - O(k^3) per edge for the optimal transport, negligible for k = 7.
  - More reliable on sparse high-dimensional graphs than Forman-Ricci, because
    it measures community structure (neighborhood overlap) rather than local
    edge-weight ratios.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_orc_edge(pos_u: np.ndarray,
                     pos_v: np.ndarray,
                     nbrs_u: np.ndarray,
                     nbrs_v: np.ndarray) -> float:
    """
    Compute Ollivier-Ricci Curvature for a single edge (u, v).

    Args:
        pos_u:  Position of agent u, shape (dim,).
        pos_v:  Position of agent v, shape (dim,).
        nbrs_u: Positions of u's neighbors (excluding v), shape (k_u, dim).
        nbrs_v: Positions of v's neighbors (excluding u), shape (k_v, dim).

    Returns:
        ORC value in [-1.0, 1.0].
    """
    d_uv = float(np.linalg.norm(pos_u - pos_v))
    if d_uv < 1e-12:
        return 0.0

    # Build support sets: include the center node itself (alpha=0 convention)
    # mu_u is uniform over {u} union N(u) -- k_u + 1 points total
    sup_u = np.vstack([pos_u[np.newaxis, :], nbrs_u])  # (k_u+1, dim)
    sup_v = np.vstack([pos_v[np.newaxis, :], nbrs_v])  # (k_v+1, dim)

    n_u = len(sup_u)
    n_v = len(sup_v)

    # Pairwise Euclidean cost matrix C[i,j] = ||sup_u[i] - sup_v[j]||
    diff = sup_u[:, np.newaxis, :] - sup_v[np.newaxis, :, :]  # (n_u, n_v, dim)
    C = np.linalg.norm(diff, axis=-1)                          # (n_u, n_v)

    # Wasserstein-1 between two uniform distributions.
    # When |N(u)+1| == |N(v)+1|, OT reduces to the linear assignment problem:
    #   W1 = (1/n) * min_permutation sum_i C[i, pi(i)]
    #
    # When sizes differ, pad with copies of the respective centroid so we
    # still use the fast O(n^3) Hungarian solver (avoids a full LP).
    if n_u != n_v:
        sup_u, sup_v, C = _pad_to_equal_size(sup_u, sup_v)
        n = len(sup_u)
    else:
        n = n_u

    row_ind, col_ind = linear_sum_assignment(C)
    W1 = float(np.sum(C[row_ind, col_ind])) / n   # divide by n for uniform 1/n weights

    orc = 1.0 - W1 / d_uv
    return float(np.clip(orc, -1.0, 1.0))


def _pad_to_equal_size(sup_u: np.ndarray,
                       sup_v: np.ndarray):
    """
    Pad the smaller support set with copies of its centroid so both have the
    same size, enabling the Hungarian algorithm for the optimal transport.

    Padding with the centroid is equivalent to adding a 'dummy' mass point
    at the mean position, which minimally distorts the transport plan.
    """
    n_u, n_v = len(sup_u), len(sup_v)
    n = max(n_u, n_v)

    if n_u < n:
        centroid = sup_u.mean(axis=0, keepdims=True)
        padding = np.repeat(centroid, n - n_u, axis=0)
        sup_u = np.vstack([sup_u, padding])

    if n_v < n:
        centroid = sup_v.mean(axis=0, keepdims=True)
        padding = np.repeat(centroid, n - n_v, axis=0)
        sup_v = np.vstack([sup_v, padding])

    diff = sup_u[:, np.newaxis, :] - sup_v[np.newaxis, :, :]
    C = np.linalg.norm(diff, axis=-1)
    return sup_u, sup_v, C


def compute_orc_graph(pop: np.ndarray,
                      adjacency: list,
                      k: int = 7) -> np.ndarray:
    """
    Batch-compute ORC for all edges in an adjacency list.

    Args:
        pop:       Agent positions, shape (N, dim).
        adjacency: List of (u, v) integer pairs (directed or undirected).
        k:         Neighbourhood size used when building the graph (for info only).

    Returns:
        Array of ORC values, shape (len(adjacency),), matching adjacency order.

    Complexity: O(|E| * k^3) total, where |E| = len(adjacency).
    """
    # Build neighbour index for fast lookup: nbrs[i] = list of neighbour indices of i
    N = len(pop)
    nbrs = [[] for _ in range(N)]
    for u, v in adjacency:
        nbrs[u].append(v)
        nbrs[v].append(u)

    orc_values = np.zeros(len(adjacency))
    for edge_idx, (u, v) in enumerate(adjacency):
        # Neighbours of u, excluding v
        nu = [w for w in nbrs[u] if w != v]
        # Neighbours of v, excluding u
        nv = [w for w in nbrs[v] if w != u]

        if not nu or not nv:
            orc_values[edge_idx] = 0.0
            continue

        orc_values[edge_idx] = compute_orc_edge(
            pop[u], pop[v],
            pop[np.array(nu, dtype=int)],
            pop[np.array(nv, dtype=int)],
        )

    return orc_values

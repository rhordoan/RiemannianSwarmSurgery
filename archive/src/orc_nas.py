"""
ORC-NAS: Discrete Ollivier-Ricci Curvature for Neural Architecture Search.

The NAS-Bench-201 / NATS-Bench topology search space is a native discrete
graph: 15,625 architectures (5^6), where each architecture is defined by
6 operation choices. Two architectures are neighbors if they differ in
exactly one operation (Hamming distance = 1), giving each node exactly 24
neighbors. This is the ideal domain for ORC.

This module provides:
  - Architecture encoding / decoding for the NAS-Bench-201 cell space
  - Hamming distance and neighbor generation
  - Discrete ORC computation on the architecture graph
  - Fitness-lifted curvature for saddle detection

References
----------
Dong & Yang (2020). NAS-Bench-201: Extending the Scope of Reproducible NAS.
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# NAS-Bench-201 architecture space
# ---------------------------------------------------------------------------

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
N_OPS = len(OPS)
N_EDGES = 6
SPACE_SIZE = N_OPS ** N_EDGES  # 15625

OP_TO_IDX = {op: i for i, op in enumerate(OPS)}
IDX_TO_OP = {i: op for i, op in enumerate(OPS)}


def arch_to_tuple(arch_str: str) -> Tuple[int, ...]:
    """
    Convert a NAS-Bench-201 architecture string to a tuple of operation indices.

    Format: '|op~0|+|op~0|op~1|+|op~0|op~1|op~2|'
    Returns a tuple of 6 integers in [0, 4].
    """
    ops = []
    for token in arch_str.split('|'):
        token = token.strip()
        if '~' in token:
            op_name = token.split('~')[0]
            ops.append(OP_TO_IDX[op_name])
    return tuple(ops)


def tuple_to_arch(t: Tuple[int, ...]) -> str:
    """Convert a tuple of 6 operation indices back to architecture string."""
    ops = [IDX_TO_OP[i] for i in t]
    return (f'|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+'
            f'|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|')


def index_to_tuple(idx: int) -> Tuple[int, ...]:
    """Convert a flat index [0, 15624] to a 6-tuple of op indices."""
    t = []
    for _ in range(N_EDGES):
        t.append(idx % N_OPS)
        idx //= N_OPS
    return tuple(t)


def tuple_to_index(t: Tuple[int, ...]) -> int:
    """Convert a 6-tuple back to a flat index."""
    idx = 0
    for i in range(N_EDGES - 1, -1, -1):
        idx = idx * N_OPS + t[i]
    return idx


# ---------------------------------------------------------------------------
# Hamming distance and neighbors
# ---------------------------------------------------------------------------

def hamming_distance(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """Number of positions where two architecture tuples differ."""
    return sum(x != y for x, y in zip(a, b))


def get_neighbors(t: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    All architectures at Hamming distance 1 from t.
    Each of the 6 positions can be changed to 4 other operations -> 24 neighbors.
    """
    nbrs = []
    for pos in range(N_EDGES):
        for op in range(N_OPS):
            if op != t[pos]:
                nbr = list(t)
                nbr[pos] = op
                nbrs.append(tuple(nbr))
    return nbrs


def get_neighbor_indices(idx: int) -> List[int]:
    """Return flat indices of all 24 neighbors."""
    t = index_to_tuple(idx)
    return [tuple_to_index(n) for n in get_neighbors(t)]


# ---------------------------------------------------------------------------
# Discrete ORC on architecture graph
# ---------------------------------------------------------------------------

def compute_orc_discrete(
    u_idx: int,
    v_idx: int,
    fitness: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """
    Compute Ollivier-Ricci Curvature for edge (u, v) in the NAS architecture
    graph, using fitness-lifted Hamming distance.

    The support sets mu_u, mu_v are uniform distributions over {u} union N(u)
    and {v} union N(v) respectively. The cost matrix uses fitness-lifted
    Hamming distance: d_lift(a, b) = hamming(a, b) + gamma * |f(a) - f(b)|.

    Parameters
    ----------
    u_idx, v_idx : flat architecture indices
    fitness      : array of shape (15625,) with fitness values (lower = better)
    gamma        : weight of fitness difference in the lifted distance

    Returns
    -------
    ORC value in [-1, 1].
    """
    u_t = index_to_tuple(u_idx)
    v_t = index_to_tuple(v_idx)

    # Support sets: {node} + neighbors
    u_nbrs = get_neighbors(u_t)
    v_nbrs = get_neighbors(v_t)
    sup_u = [u_t] + u_nbrs  # 25 points
    sup_v = [v_t] + v_nbrs  # 25 points

    n = len(sup_u)  # always 25

    # Cost matrix: fitness-lifted Hamming distance
    C = np.zeros((n, n))
    for i, a in enumerate(sup_u):
        a_idx = tuple_to_index(a)
        for j, b in enumerate(sup_v):
            b_idx = tuple_to_index(b)
            ham = hamming_distance(a, b)
            fit_diff = abs(fitness[a_idx] - fitness[b_idx])
            C[i, j] = ham + gamma * fit_diff

    # d(u, v) in lifted space
    d_uv = hamming_distance(u_t, v_t) + gamma * abs(fitness[u_idx] - fitness[v_idx])
    if d_uv < 1e-12:
        return 0.0

    # Wasserstein-1 via Hungarian algorithm (uniform weights -> assignment)
    row_ind, col_ind = linear_sum_assignment(C)
    W1 = float(np.sum(C[row_ind, col_ind])) / n

    orc = 1.0 - W1 / d_uv
    return float(np.clip(orc, -1.0, 1.0))


def compute_orc_neighborhood(
    center_idx: int,
    fitness: np.ndarray,
    gamma: float = 1.0,
) -> Dict[int, float]:
    """
    Compute Ollivier-Ricci Curvature for all 24 edges incident to center_idx.

    Uses uniform distributions and the Hungarian algorithm for fast W1 computation.
    Precomputes center's support set for efficiency.
    Returns {neighbor_idx: orc_value}.
    """
    center_t = index_to_tuple(center_idx)
    center_nbrs = get_neighbors(center_t)
    sup_u = [center_t] + center_nbrs
    sup_u_indices = [tuple_to_index(a) for a in sup_u]
    sup_u_fit = np.array([fitness[i] for i in sup_u_indices])
    n = len(sup_u)  # 25
    arr_u = np.array(sup_u, dtype=np.int8)  # (25, 6)

    orc_values = {}
    for nbr_t in center_nbrs:
        nbr_idx = tuple_to_index(nbr_t)
        nbr_nbrs = get_neighbors(nbr_t)
        sup_v = [nbr_t] + nbr_nbrs
        sup_v_indices = [tuple_to_index(b) for b in sup_v]
        sup_v_fit = np.array([fitness[i] for i in sup_v_indices])

        arr_v = np.array(sup_v, dtype=np.int8)  # (25, 6)
        ham_matrix = (arr_u[:, np.newaxis, :] != arr_v[np.newaxis, :, :]).sum(axis=2).astype(np.float64)
        fit_diff_matrix = np.abs(sup_u_fit[:, np.newaxis] - sup_v_fit[np.newaxis, :])
        C = ham_matrix + gamma * fit_diff_matrix

        d_uv = float(C[0, 0])
        if d_uv < 1e-12:
            orc_values[nbr_idx] = 0.0
            continue

        row_ind, col_ind = linear_sum_assignment(C)
        W1 = float(np.sum(C[row_ind, col_ind])) / n
        orc = float(np.clip(1.0 - W1 / d_uv, -1.0, 1.0))
        orc_values[nbr_idx] = orc

    return orc_values


def find_saddle_direction(
    center_idx: int,
    fitness: np.ndarray,
    gamma: float = 1.0,
) -> Tuple[Optional[int], float]:
    """
    Find the neighbor with the most negative ORC (strongest saddle direction).

    Returns (neighbor_idx, min_orc). If all ORC >= 0, returns (None, 0.0).
    """
    orc = compute_orc_neighborhood(center_idx, fitness, gamma)
    if not orc:
        return None, 0.0

    min_nbr = min(orc, key=orc.get)
    min_orc = orc[min_nbr]

    if min_orc >= 0:
        return None, 0.0
    return min_nbr, min_orc


# ---------------------------------------------------------------------------
# NAS-Bench-201 fitness lookup (via NATS-Bench or synthetic)
# ---------------------------------------------------------------------------

class NASBench201:
    """
    Wrapper around the NAS-Bench-201 / NATS-Bench topology search space.

    Provides a fitness array of shape (15625,) where fitness = 100 - accuracy
    (lower is better, minimization problem).
    """

    def __init__(self, dataset: str = 'cifar10', data_path: Optional[str] = None):
        """
        Parameters
        ----------
        dataset   : 'cifar10', 'cifar100', or 'ImageNet16-120'
        data_path : path to NATS-Bench TSS database file. If None, uses
                    synthetic landscape.
        """
        self.dataset = dataset
        self._fitness = np.full(SPACE_SIZE, np.inf)
        self._accuracy = np.zeros(SPACE_SIZE)
        self._loaded = False

        if data_path is not None:
            self._load_nats(data_path, dataset)
        else:
            self._build_synthetic()

    def _load_nats(self, data_path: str, dataset: str):
        """Load real accuracies from NATS-Bench, with .npy caching."""
        import os
        cache_dir = os.path.join(os.path.dirname(data_path), 'cache')
        cache_file = os.path.join(cache_dir, f'nats_accuracy_{dataset}.npy')

        if os.path.exists(cache_file):
            self._accuracy = np.load(cache_file)
            self._fitness = 100.0 - self._accuracy
            self._loaded = True
            return

        try:
            from nats_bench import create
            print(f"Loading NATS-Bench from {data_path} (first time, will cache)...",
                  flush=True)
            api = create(data_path, 'tss', fast_mode=True, verbose=False)
            for idx in range(SPACE_SIZE):
                info = api.get_more_info(idx, dataset, hp='200', is_random=False)
                acc = info.get('test-accuracy', info.get('valid-accuracy', 0.0))
                self._accuracy[idx] = acc
                self._fitness[idx] = 100.0 - acc
                if idx % 5000 == 0 and idx > 0:
                    print(f"  ...loaded {idx}/{SPACE_SIZE}", flush=True)
            self._loaded = True

            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, self._accuracy)
            print(f"Cached to {cache_file}", flush=True)
            del api
        except Exception as e:
            print(f"Warning: could not load NATS-Bench ({e}). "
                  f"Using synthetic landscape.")
            self._build_synthetic()

    def _build_synthetic(self):
        """
        Build a rigorously controlled deceptive landscape for ORC saddle-crossing.

        - Massive Funnel: Base score is 50.0 + 5.0 * (number of 1s).
          This means EVERY random start will hill-climb directly to (1,1,1,1,1,1)
          which has a score of 80.0.
        - Global Optimum: (3,3,3,3,3,3) with score 99.0.
        - Saddle Path: A single, specific 6-step path connecting them.
          Scores: 80 -> 78 -> 76 -> 75 -> 82 -> 90 -> 99.
        """
        rng = np.random.RandomState(42)

        path = [
            (1,1,1,1,1,1),
            (3,1,1,1,1,1),
            (3,3,1,1,1,1),
            (3,3,3,1,1,1),
            (3,3,3,3,1,1),
            (3,3,3,3,3,1),
            (3,3,3,3,3,3)
        ]
        path_scores = [80.0, 78.0, 76.0, 75.0, 82.0, 90.0, 99.0]

        for idx in range(SPACE_SIZE):
            t = index_to_tuple(idx)
            
            if t in path:
                n_3 = sum(1 for x in t if x == 3)
                score = path_scores[n_3]
            else:
                n_1 = sum(1 for x in t if x == 1)
                score = 50.0 + n_1 * 5.0

            noise = rng.normal(0, 0.05)
            self._accuracy[idx] = max(0.0, min(100.0, score + noise))
            self._fitness[idx] = 100.0 - self._accuracy[idx]

    @property
    def fitness(self) -> np.ndarray:
        """Fitness array (lower = better). Shape (15625,)."""
        return self._fitness

    @property
    def accuracy(self) -> np.ndarray:
        """Accuracy array (higher = better). Shape (15625,)."""
        return self._accuracy

    def query(self, idx: int) -> float:
        """Query fitness of architecture by flat index."""
        return float(self._fitness[idx])

    def query_accuracy(self, idx: int) -> float:
        """Query accuracy of architecture by flat index."""
        return float(self._accuracy[idx])

    @property
    def is_real(self) -> bool:
        return self._loaded

    def best_index(self) -> int:
        return int(np.argmin(self._fitness))

    def best_accuracy(self) -> float:
        return float(self._accuracy[self.best_index()])

"""
RiemannianOracle v3: Fitness-Aware Topological Monitor.

v3 changes:
  - FITNESS-WEIGHTED k-NN GRAPH: The graph distance between agents u and v is
    d_combined(u,v) = ||x_u - x_v|| * (1 + beta * |f_norm(u) - f_norm(v)|)
    This makes agents at similar fitness levels more likely to be in the same
    community, and agents with large fitness differences less likely to be
    neighbors. ORC then detects boundaries where fitness changes abruptly
    across spatial communities -- a much stronger signal than pure spatial ORC.

  - ORC still uses raw Euclidean distance for the Wasserstein computation.
    Only the GRAPH STRUCTURE is fitness-weighted; the METRIC in ORC is unchanged.
    This preserves the mathematical properties of ORC (bounded [-1, 1]).

  - EXPLORE-SIDE FITNESS STATS: For each detected saddle, the oracle computes
    fitness statistics of the explore-side neighborhood (mean, variance).
    Saddles where the explore-side has zero fitness variance (flat plateau)
    are filtered out -- they indicate ridge boundaries, not inter-basin saddles.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from src.ollivier_ricci import compute_orc_edge


class RiemannianOracle:

    def __init__(self,
                 dim: int,
                 k: int = 7,
                 orc_threshold: float = -0.1,
                 update_period: int = 5,
                 domain_width: float = 200.0,
                 history_size: int = 80,
                 fitness_weight: float = 0.0):
        self.dim = dim
        self.k = k
        self.orc_threshold = orc_threshold
        self.update_period = update_period
        self.domain_width = domain_width
        self.history_size = history_size
        self.fitness_weight = fitness_weight

        self._adj: list = []
        self._orc: np.ndarray = np.empty(0)
        self._nbrs: list = []
        self._last_pop_size: int = 0
        self._last_update_gen: int = -999

        self._history_pos: list = []
        self._history_fit: list = []

        self.min_orc: float = 0.0
        self.mean_orc: float = 0.0
        self.n_saddle_edges: int = 0

    def step(self,
             pop: np.ndarray,
             fitness: np.ndarray,
             generation: int) -> list:
        N = len(pop)

        if (generation - self._last_update_gen) < self.update_period:
            return []

        if N < self.k + 2:
            return []

        self._last_update_gen = generation
        self._update_history(pop, fitness)

        aug_pop, aug_fit, n_current = self._build_augmented(pop, fitness)

        self._build_knn(aug_pop, aug_fit, len(aug_pop))
        self._compute_orc(aug_pop)
        return self._detect_saddles(aug_pop, aug_fit, n_current)

    # ------------------------------------------------------------------
    # Historical reservoir
    # ------------------------------------------------------------------

    def _update_history(self, pop: np.ndarray, fitness: np.ndarray):
        if len(self._history_pos) < self.history_size:
            for i in range(len(pop)):
                if len(self._history_pos) >= self.history_size:
                    break
                self._history_pos.append(pop[i].copy())
                self._history_fit.append(float(fitness[i]))
            return

        hist_arr = np.array(self._history_pos)
        hist_fit = np.array(self._history_fit)
        worst_fit = hist_fit.max()

        candidates = np.where(fitness < worst_fit)[0]
        if len(candidates) == 0:
            return

        best_cands = candidates[np.argsort(fitness[candidates])[:5]]

        for i in best_cands:
            worst_idx = int(np.argmax(hist_fit))
            if fitness[i] >= hist_fit[worst_idx]:
                continue
            dists = np.linalg.norm(hist_arr - pop[i], axis=1)
            if dists.min() > 1e-8:
                hist_arr[worst_idx] = pop[i]
                hist_fit[worst_idx] = float(fitness[i])
                self._history_pos[worst_idx] = pop[i].copy()
                self._history_fit[worst_idx] = float(fitness[i])

    def _build_augmented(self, pop, fitness):
        n_current = len(pop)
        if not self._history_pos:
            return pop, fitness, n_current

        hist_pos = np.array(self._history_pos)
        hist_fit = np.array(self._history_fit)

        aug_pop = np.vstack([pop, hist_pos])
        aug_fit = np.concatenate([fitness, hist_fit])
        return aug_pop, aug_fit, n_current

    # ------------------------------------------------------------------
    # Fitness-weighted k-NN graph
    # ------------------------------------------------------------------

    def _build_knn(self, pop: np.ndarray, fitness: np.ndarray, N: int):
        """
        Build k-NN graph using fitness-weighted distances.

        d(u,v) = ||x_u - x_v|| * (1 + beta * |f_norm(u) - f_norm(v)|)

        When beta=0, this reduces to standard Euclidean k-NN.
        When beta>0, agents with very different fitness are less likely
        to be neighbors, creating fitness-aware community structure.
        """
        k_actual = min(self.k, N - 1)

        if self.fitness_weight > 0 and N <= 500:
            # Fitness-weighted distance matrix (brute force, fine for N<500)
            spatial_dist = cdist(pop, pop)

            f_min, f_max = float(fitness.min()), float(fitness.max())
            f_range = max(f_max - f_min, 1e-12)
            f_norm = (fitness - f_min) / f_range

            fitness_diff = np.abs(f_norm[:, None] - f_norm[None, :])
            combined_dist = spatial_dist * (1.0 + self.fitness_weight * fitness_diff)
            np.fill_diagonal(combined_dist, np.inf)

            edge_set: set = set()
            for u in range(N):
                nn_indices = np.argpartition(combined_dist[u], k_actual)[:k_actual]
                for v in nn_indices:
                    if u != v:
                        edge_set.add((min(u, v), max(u, v)))
        else:
            # Fallback to standard KDTree for large populations or beta=0
            tree = KDTree(pop)
            _, indices = tree.query(pop, k=k_actual + 1)
            edge_set: set = set()
            for u in range(N):
                for j in range(1, k_actual + 1):
                    v = int(indices[u, j])
                    if u != v:
                        edge_set.add((min(u, v), max(u, v)))

        self._adj = list(edge_set)
        self._last_pop_size = N

        self._nbrs = [[] for _ in range(N)]
        for u, v in self._adj:
            self._nbrs[u].append(v)
            self._nbrs[v].append(u)

    # ------------------------------------------------------------------
    # ORC (always uses raw Euclidean distance)
    # ------------------------------------------------------------------

    def _compute_orc(self, pop: np.ndarray):
        n_edges = len(self._adj)
        self._orc = np.zeros(n_edges)

        for idx, (u, v) in enumerate(self._adj):
            nu_idx = [w for w in self._nbrs[u] if w != v]
            nv_idx = [w for w in self._nbrs[v] if w != u]

            if not nu_idx or not nv_idx:
                self._orc[idx] = 0.0
                continue

            self._orc[idx] = compute_orc_edge(
                pop[u], pop[v],
                pop[np.array(nu_idx, dtype=int)],
                pop[np.array(nv_idx, dtype=int)],
            )

        if len(self._orc) > 0:
            self.min_orc = float(self._orc.min())
            self.mean_orc = float(self._orc.mean())
        else:
            self.min_orc = self.mean_orc = 0.0

    # ------------------------------------------------------------------
    # Saddle detection with fitness-based filtering
    # ------------------------------------------------------------------

    def _detect_saddles(self, pop, fitness, n_current) -> list:
        saddle_mask = self._orc < self.orc_threshold
        saddle_indices = np.where(saddle_mask)[0]

        if len(saddle_indices) == 0:
            self.n_saddle_edges = 0
            return []

        order = np.argsort(self._orc[saddle_indices])
        saddle_indices = saddle_indices[order]

        results = []
        for idx in saddle_indices:
            u, v = self._adj[idx]

            if u >= n_current and v >= n_current:
                continue

            # Determine explore side (worse fitness = unexplored)
            if fitness[u] <= fitness[v]:
                explore_node = v
                known_node = u
            else:
                explore_node = u
                known_node = v

            # Compute explore-side neighborhood fitness statistics
            explore_nbrs = self._nbrs[explore_node]
            if not explore_nbrs:
                continue

            nbr_indices = np.array(explore_nbrs, dtype=int)
            nbr_positions = pop[nbr_indices]
            nbr_fitness = fitness[nbr_indices]
            centroid = nbr_positions.mean(axis=0)

            # PROMISE FILTER: Skip saddles where the explore-side
            # neighborhood has no fitness diversity (flat plateau).
            # On HGBat, going off the ridge leads to uniformly bad fitness.
            # On Rastrigin, the other side has diverse fitness (multiple basins).
            f_explore_std = float(np.std(nbr_fitness))
            f_explore_mean = float(np.mean(nbr_fitness))

            # Also check: does the explore side have ANY agent better than
            # the midpoint fitness? If not, there's nothing promising there.
            midpoint_fitness = (fitness[u] + fitness[v]) * 0.5
            has_promise = np.any(nbr_fitness < midpoint_fitness)

            results.append({
                'u': u,
                'v': v,
                'orc': float(self._orc[idx]),
                'nbr_centroid_explore': centroid,
                'explore_fitness_std': f_explore_std,
                'explore_fitness_mean': f_explore_mean,
                'has_promise': bool(has_promise),
            })

        self.n_saddle_edges = len(results)
        return results

    def get_orc_stats(self) -> dict:
        if len(self._orc) == 0:
            return {'min': None, 'mean': None, 'n_edges': 0, 'n_saddles': 0}
        return {
            'min': self.min_orc,
            'mean': self.mean_orc,
            'n_edges': len(self._adj),
            'n_saddles': self.n_saddle_edges,
        }

"""
RiemannianOracle: Passive Topological Monitor for Population-Based Optimizers.

Uses Ollivier-Ricci Curvature (ORC) on a k-NN graph to identify inter-basin
saddle edges in the current population.  Zero additional function evaluations.

v2 improvements (informed by D=10 CEC 2022 ablation):
  - Historical population buffer: the k-NN graph is built on the union of the
    current population and a reservoir of previously evaluated solutions.  This
    keeps the graph dense even in late generations when LPSR has shrunk the
    population to ~4 agents, producing much more accurate ORC estimates.
  - Neighborhood centroid export: for each detected saddle edge (u, v), the
    oracle returns the centroid of the "unexplored-side" neighborhood.  This
    gives the SaddleArchive a more precise injection target than the crude
    midpoint + direction vector.
"""

import numpy as np
from scipy.spatial import KDTree

from src.ollivier_ricci import compute_orc_edge


class RiemannianOracle:

    def __init__(self,
                 dim: int,
                 k: int = 7,
                 orc_threshold: float = -0.1,
                 update_period: int = 5,
                 domain_width: float = 200.0,
                 history_size: int = 80):
        self.dim = dim
        self.k = k
        self.orc_threshold = orc_threshold
        self.update_period = update_period
        self.domain_width = domain_width
        self.history_size = history_size

        self._adj: list = []
        self._orc: np.ndarray = np.empty(0)
        self._nbrs: list = []
        self._last_pop_size: int = 0
        self._last_update_gen: int = -999

        # Historical buffer: reservoir of (position, fitness) from past generations
        self._history_pos: list = []
        self._history_fit: list = []

        self.min_orc: float = 0.0
        self.mean_orc: float = 0.0
        self.n_saddle_edges: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self,
             pop: np.ndarray,
             fitness: np.ndarray,
             generation: int) -> list:
        """
        Returns list of dicts with keys:
          'u', 'v':             Agent indices in the AUGMENTED population.
          'nbr_centroid_explore': Centroid of the unexplored-side neighborhood.
        Only edges with ORC < threshold are returned.
        """
        N = len(pop)

        if (generation - self._last_update_gen) < self.update_period:
            return []

        if N < self.k + 2:
            return []

        self._last_update_gen = generation
        self._update_history(pop, fitness)

        # Build augmented population: current pop + historical reservoir
        aug_pop, aug_fit, n_current = self._build_augmented(pop, fitness)

        self._build_knn(aug_pop, len(aug_pop))
        self._compute_orc(aug_pop)
        return self._detect_saddles(aug_pop, aug_fit, n_current)

    # ------------------------------------------------------------------
    # Historical reservoir
    # ------------------------------------------------------------------

    def _update_history(self, pop: np.ndarray, fitness: np.ndarray):
        """
        Add best agents from current population to the history reservoir,
        keeping the best unique solutions seen across all generations.
        """
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

        # Only consider agents better than the worst in history
        candidates = np.where(fitness < worst_fit)[0]
        if len(candidates) == 0:
            return

        # Batch: take the best 5 candidates per generation to limit overhead
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
        """Combine current pop with historical buffer."""
        n_current = len(pop)
        if not self._history_pos:
            return pop, fitness, n_current

        hist_pos = np.array(self._history_pos)
        hist_fit = np.array(self._history_fit)

        aug_pop = np.vstack([pop, hist_pos])
        aug_fit = np.concatenate([fitness, hist_fit])
        return aug_pop, aug_fit, n_current

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_knn(self, pop: np.ndarray, N: int):
        k_actual = min(self.k, N - 1)
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

        # Build per-node neighbour lists
        self._nbrs = [[] for _ in range(N)]
        for u, v in self._adj:
            self._nbrs[u].append(v)
            self._nbrs[v].append(u)

    # ------------------------------------------------------------------
    # ORC computation
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
    # Saddle detection with neighborhood centroids
    # ------------------------------------------------------------------

    def _detect_saddles(self, pop, fitness, n_current) -> list:
        """
        Return saddle edges with ORC < threshold, enriched with
        neighborhood centroids for the unexplored side.

        Only returns saddles where at least one endpoint is in the
        CURRENT population (not purely historical).
        """
        saddle_mask = self._orc < self.orc_threshold
        saddle_indices = np.where(saddle_mask)[0]

        if len(saddle_indices) == 0:
            self.n_saddle_edges = 0
            return []

        # Sort by ORC (most negative first)
        order = np.argsort(self._orc[saddle_indices])
        saddle_indices = saddle_indices[order]

        results = []
        for idx in saddle_indices:
            u, v = self._adj[idx]

            # At least one endpoint must be in the current population
            if u >= n_current and v >= n_current:
                continue

            # Determine which side is "unexplored" (worse fitness)
            if fitness[u] <= fitness[v]:
                explore_node = v
            else:
                explore_node = u

            # Centroid of the explore-side neighborhood
            explore_nbrs = self._nbrs[explore_node]
            if explore_nbrs:
                nbr_positions = pop[np.array(explore_nbrs, dtype=int)]
                centroid = nbr_positions.mean(axis=0)
            else:
                centroid = pop[explore_node].copy()

            results.append({
                'u': u,
                'v': v,
                'nbr_centroid_explore': centroid,
            })

        self.n_saddle_edges = len(results)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_orc_stats(self) -> dict:
        if len(self._orc) == 0:
            return {'min': None, 'mean': None, 'n_edges': 0, 'n_saddles': 0}
        return {
            'min': self.min_orc,
            'mean': self.mean_orc,
            'n_edges': len(self._adj),
            'n_saddles': self.n_saddle_edges,
        }

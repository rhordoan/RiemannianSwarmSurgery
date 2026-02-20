"""
RiemannianOracle: Passive Topological Monitor for Population-Based Optimizers.

This module replaces the previous Forman-Ricci / MST-surgery / Voronoi-split
implementation with a clean, mathematically defensible oracle based on
Ollivier-Ricci Curvature (ORC).

Architecture:
  - PASSIVE: The oracle is called after each optimizer generation with the
    current population. It does NOT modify the population, trigger splits,
    or consume function evaluations.
  - ADVISORY: It returns a list of (u, v) agent-index pairs whose connecting
    edge has ORC < orc_threshold. These are inter-basin saddle boundaries.
    The calling optimizer (TMIOptimizer) decides what to do with them.
  - CHEAP: k-NN graph build is O(N log N) via scipy KDTree. ORC computation
    is O(|E| * k^3) with k=7 giving 8^3 = 512 operations per edge -- negligible
    compared to the optimizer's O(N * dim) mutation and evaluation steps.

Why ORC instead of Forman-Ricci?
  - Forman-Ricci on fitness-weight graphs suffered from numerical blowup
    (weights approaching zero cause sqrt(w_e/w_eu) -> inf).
  - ORC is bounded [-1, 1] by a mathematical theorem (not a clamp).
  - ORC measures community structure in the k-NN graph: negative ORC directly
    identifies edges that cross topological community boundaries (basins),
    which is exactly the signal needed for manifold-aware injection.

Reference:
    Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces.
    Journal of Functional Analysis, 256(3), 810-864.
"""

import numpy as np
from scipy.spatial import KDTree

from src.ollivier_ricci import compute_orc_edge


class RiemannianOracle:
    """
    Passive topological monitor using Ollivier-Ricci Curvature on a k-NN graph.

    Identifies inter-basin saddle edges (ORC < threshold) in the current
    population's search-space geometry. Zero additional function evaluations.

    Args:
        dim:            Dimensionality of the search space.
        k:              Number of nearest neighbors for the graph (default 7).
        orc_threshold:  Edges with ORC below this value are flagged as saddles
                        (default -0.1; more negative = stricter).
        update_period:  Oracle runs only every update_period generations
                        (amortises k-NN cost; default 5).
        domain_width:   Search domain diameter, used for scaling (default 200).
    """

    def __init__(self,
                 dim: int,
                 k: int = 7,
                 orc_threshold: float = -0.1,
                 update_period: int = 5,
                 domain_width: float = 200.0):
        self.dim = dim
        self.k = k
        self.orc_threshold = orc_threshold
        self.update_period = update_period
        self.domain_width = domain_width

        # Internal graph state
        self._adj: list = []          # list of (u, v) pairs
        self._orc: np.ndarray = np.empty(0)  # ORC value for each edge
        self._last_pop_size: int = 0
        self._last_update_gen: int = -999

        # Diagnostics exposed for logging/paper tables
        self.min_orc: float = 0.0
        self.mean_orc: float = 0.0
        self.n_saddle_edges: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def step(self,
             pop: np.ndarray,
             fitness: np.ndarray,
             generation: int) -> list:
        """
        Called once per optimizer generation with the current population.

        Args:
            pop:        Agent positions, shape (N, dim).
            fitness:    Fitness values, shape (N,).  Lower = better.
            generation: Current generation index (used for update_period gate).

        Returns:
            List of (u, v) integer index pairs flagging inter-basin saddle
            edges (ORC < orc_threshold), sorted most-negative first.
            Returns [] on non-update generations.
        """
        N = len(pop)

        # Skip this generation if not due for an update
        if (generation - self._last_update_gen) < self.update_period:
            return []

        # Need at least k+1 agents to form a meaningful k-NN graph
        if N < self.k + 2:
            return []

        self._last_update_gen = generation
        self._build_knn(pop, N)
        self._compute_orc(pop)
        return self._detect_saddles()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_knn(self, pop: np.ndarray, N: int):
        """
        Build a symmetric k-NN adjacency list using scipy KDTree.

        Each node u is connected to its k nearest neighbours.  The graph is
        made undirected by adding both (u, v) and (v, u) directions, then
        deduplicating.  Self-loops are excluded.

        Complexity: O(N * k * log N)
        """
        k_actual = min(self.k, N - 1)
        tree = KDTree(pop)
        # query returns (distances, indices); k+1 because the query point is
        # always the first result (distance 0, index = self)
        _, indices = tree.query(pop, k=k_actual + 1)

        edge_set: set = set()
        for u in range(N):
            for j in range(1, k_actual + 1):  # skip column 0 (self)
                v = int(indices[u, j])
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))

        self._adj = list(edge_set)
        self._last_pop_size = N

    # ------------------------------------------------------------------
    # ORC computation
    # ------------------------------------------------------------------

    def _compute_orc(self, pop: np.ndarray):
        """
        Compute ORC for every edge in the adjacency list.

        Uses compute_orc_edge from src.ollivier_ricci, which solves an
        (k+1) x (k+1) optimal transport problem per edge via the Hungarian
        algorithm.

        Complexity: O(|E| * k^3)  -- for k=7, |E|~N*k/2: tiny overhead.
        """
        # Build per-node neighbour list for fast lookup
        N = self._last_pop_size
        nbrs: list = [[] for _ in range(N)]
        for u, v in self._adj:
            nbrs[u].append(v)
            nbrs[v].append(u)

        n_edges = len(self._adj)
        self._orc = np.zeros(n_edges)

        for idx, (u, v) in enumerate(self._adj):
            nu_idx = [w for w in nbrs[u] if w != v]
            nv_idx = [w for w in nbrs[v] if w != u]

            if not nu_idx or not nv_idx:
                self._orc[idx] = 0.0
                continue

            self._orc[idx] = compute_orc_edge(
                pop[u], pop[v],
                pop[np.array(nu_idx, dtype=int)],
                pop[np.array(nv_idx, dtype=int)],
            )

        # Update diagnostics
        if len(self._orc) > 0:
            self.min_orc = float(self._orc.min())
            self.mean_orc = float(self._orc.mean())
        else:
            self.min_orc = self.mean_orc = 0.0

    # ------------------------------------------------------------------
    # Saddle detection
    # ------------------------------------------------------------------

    def _detect_saddles(self) -> list:
        """
        Filter edges by ORC < threshold.

        Returns:
            Sorted list of (u, v) pairs, most negative ORC first.
        """
        saddle_mask = self._orc < self.orc_threshold
        saddle_edges = [self._adj[i] for i in np.where(saddle_mask)[0]]
        saddle_orc = self._orc[saddle_mask]

        # Sort: most negative ORC first (clearest inter-basin boundaries first)
        order = np.argsort(saddle_orc)
        saddle_edges = [saddle_edges[i] for i in order]

        self.n_saddle_edges = len(saddle_edges)
        return saddle_edges

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def get_orc_stats(self) -> dict:
        """
        Return a dict of ORC statistics for logging or paper tables.
        """
        if len(self._orc) == 0:
            return {'min': None, 'mean': None, 'n_edges': 0, 'n_saddles': 0}
        return {
            'min': self.min_orc,
            'mean': self.mean_orc,
            'n_edges': len(self._adj),
            'n_saddles': self.n_saddle_edges,
        }

    def get_edge_orc(self, u: int, v: int) -> float | None:
        """Return ORC for a specific edge (u, v), or None if not in graph."""
        key = (min(u, v), max(u, v))
        for idx, edge in enumerate(self._adj):
            if edge == key:
                return float(self._orc[idx])
        return None

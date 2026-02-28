"""
ORC-BO: Topology-Aware Bayesian Optimization via Ollivier-Ricci Curvature.

The first bridge between discrete Riemannian geometry and surrogate-based
optimization. ORC-BO builds a GP surrogate and computes Ollivier-Ricci
curvature on a DENSE virtual sample of the predicted landscape. This reveals
the topological structure -- basins (positive-ORC components) and saddle
regions (negative-ORC edges) -- enabling a topology-aware acquisition.

Acquisition strategy (hybrid EI + ORC):
  1. Expected Improvement (primary exploitation driver)
  2. ORC saddle exploration (topology-guided, secondary)
  3. Greedy batch selection with distance penalization

References
----------
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
Rasmussen & Williams (2006). Gaussian Processes for Machine Learning.
Jones et al. (1998). Efficient Global Optimization (EGO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from src.ollivier_ricci import compute_orc_edge


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CurvatureField:
    """Dense curvature field computed on the GP surrogate surface."""
    points: np.ndarray       # (N_virtual, dim) -- original-space positions
    points_unit: np.ndarray  # (N_virtual, dim) -- [0,1]^D positions
    means: np.ndarray        # (N_virtual,)     -- GP predicted means
    stds: np.ndarray         # (N_virtual,)     -- GP predicted std devs
    kappa: np.ndarray        # (N_virtual,)     -- per-point min incident ORC
    edges: list              # [(u, v), ...]    -- k-NN edge list
    orc_values: np.ndarray   # (n_edges,)       -- ORC per edge
    nbrs_list: list          # [set(), ...]     -- adjacency lists
    ei: np.ndarray           # (N_virtual,)     -- expected improvement


@dataclass
class Basin:
    """A connected component of positive-ORC edges."""
    members: np.ndarray
    best_idx: int
    best_point: np.ndarray
    best_predicted: float
    mean_uncertainty: float


@dataclass
class AcquisitionCandidate:
    """Scored candidate point for evaluation."""
    point: np.ndarray       # original-space position
    score: float
    source: str             # 'ei', 'saddle', 'basin'


# ---------------------------------------------------------------------------
# ORC-BO optimizer
# ---------------------------------------------------------------------------

class ORCBO:
    """
    Ollivier-Ricci Curvature-guided Bayesian Optimization.

    All inputs/outputs are in the original problem space. Internally, the GP
    operates in [0,1]^D normalized space for stable kernel hyperparameters.
    """

    def __init__(self, problem, dim: int, budget: int, *,
                 batch_size: Optional[int] = None,
                 n_init: Optional[int] = None,
                 n_virtual: int = 500,
                 k_orc: Optional[int] = None,
                 orc_period: int = 3,
                 orc_start_frac: float = 0.15,
                 seed: Optional[int] = None):
        self.problem = problem
        self.dim = dim
        self.budget = budget
        self.batch_size = batch_size if batch_size is not None else max(dim, 5)
        self.n_init = n_init if n_init is not None else max(5 * dim, 20)
        self.n_virtual = n_virtual
        self.k_orc = k_orc if k_orc is not None else min(dim + 1, 20)
        self.orc_period = orc_period
        self.orc_start_frac = orc_start_frac

        self.lb = float(problem.bounds[0])
        self.ub = float(problem.bounds[1])
        self.rng = np.random.RandomState(seed)

        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=0.2 * np.ones(dim),
                          length_scale_bounds=(1e-3, 1e2))
                   + WhiteKernel(noise_level=1e-6,
                                 noise_level_bounds=(1e-10, 1e-1)),
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed,
        )

        self.X_eval: List[np.ndarray] = []
        self.y_eval: List[float] = []
        self.fe_count = 0
        self.best_fitness = np.inf
        self.best_solution: Optional[np.ndarray] = None
        self.convergence_log: List[Tuple[int, float]] = []

        self._n_basins_history: List[int] = []
        self._n_saddles_history: List[int] = []
        self._step_count = 0
        self._cached_field = None
        self._cached_basins = None
        self._cached_saddles = None

    # ------------------------------------------------------------------
    # Space normalization
    # ------------------------------------------------------------------

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        """Map from original [lb, ub]^D to [0, 1]^D."""
        return (X - self.lb) / (self.ub - self.lb)

    def _from_unit(self, X_unit: np.ndarray) -> np.ndarray:
        """Map from [0, 1]^D back to [lb, ub]^D."""
        return self.lb + (self.ub - self.lb) * X_unit

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):
        """LHS initialization: space-filling initial sample."""
        sampler = LatinHypercube(d=self.dim, seed=self.rng)
        X_unit = sampler.random(n=self.n_init)
        X_init = self._from_unit(X_unit)
        for x in X_init:
            y = float(self.problem.evaluate(x))
            self.X_eval.append(x.copy())
            self.y_eval.append(y)
            self.fe_count += 1
            if y < self.best_fitness:
                self.best_fitness = y
                self.best_solution = x.copy()
        self.convergence_log.append((self.fe_count, self.best_fitness))

    # ------------------------------------------------------------------
    # GP surrogate (operates in [0,1]^D)
    # ------------------------------------------------------------------

    def _fit_surrogate(self):
        """Fit GP on normalized inputs."""
        X_unit = self._to_unit(np.array(self.X_eval))
        y = np.array(self.y_eval)
        n = len(X_unit)
        if n > 300:
            self.gp.n_restarts_optimizer = 1
        elif n > 100:
            self.gp.n_restarts_optimizer = 3
        self.gp.fit(X_unit, y)

    def _predict(self, X: np.ndarray):
        """Predict in original space (normalizes internally)."""
        X_unit = self._to_unit(X)
        return self.gp.predict(X_unit, return_std=True)

    # ------------------------------------------------------------------
    # Expected Improvement
    # ------------------------------------------------------------------

    def _compute_ei(self, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Compute EI given GP predictions."""
        f_best = min(self.y_eval)
        stds = np.maximum(stds, 1e-10)
        z = (f_best - means) / stds
        ei = (f_best - means) * norm.cdf(z) + stds * norm.pdf(z)
        return np.maximum(ei, 0.0)

    # ------------------------------------------------------------------
    # Dense curvature field on GP surface
    # ------------------------------------------------------------------

    def _compute_curvature_field(self) -> CurvatureField:
        """Sample dense virtual population, predict with GP, compute ORC."""
        sampler = LatinHypercube(d=self.dim, seed=self.rng)
        X_virtual_unit = sampler.random(n=self.n_virtual)
        X_virtual = self._from_unit(X_virtual_unit)

        means, stds = self.gp.predict(X_virtual_unit, return_std=True)
        stds = np.maximum(stds, 1e-10)
        ei = self._compute_ei(means, stds)

        # Fitness-lifted space in normalized coordinates
        log_fit = np.log1p(np.maximum(means - means.min() + 1e-10, 0.0))
        log_fit_std = max(float(log_fit.std()), 1e-10)
        gamma = np.sqrt(self.dim)
        fit_col = (gamma * log_fit / log_fit_std)[:, np.newaxis]
        lifted = np.hstack([X_virtual_unit, fit_col])

        # k-NN graph on lifted positions
        k_actual = min(self.k_orc, self.n_virtual - 1)
        if k_actual < 2:
            return CurvatureField(
                points=X_virtual, points_unit=X_virtual_unit,
                means=means, stds=stds,
                kappa=np.zeros(self.n_virtual), edges=[],
                orc_values=np.array([]),
                nbrs_list=[set() for _ in range(self.n_virtual)],
                ei=ei,
            )

        tree = KDTree(lifted)
        _, indices = tree.query(lifted, k=k_actual + 1)

        nbrs_list = [set() for _ in range(self.n_virtual)]
        edge_set = set()
        for u in range(self.n_virtual):
            for j in range(1, k_actual + 1):
                v = int(indices[u, j])
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))
                    nbrs_list[u].add(v)
                    nbrs_list[v].add(u)

        edges = list(edge_set)
        nbrs_limit = max(1, k_actual - 1)

        orc_values = np.zeros(len(edges))
        for ei_idx, (u, v) in enumerate(edges):
            nu = [w for w in nbrs_list[u] if w != v][:nbrs_limit]
            nv = [w for w in nbrs_list[v] if w != u][:nbrs_limit]
            if not nu or not nv:
                continue
            orc_values[ei_idx] = compute_orc_edge(
                lifted[u], lifted[v],
                lifted[np.array(nu, dtype=int)],
                lifted[np.array(nv, dtype=int)],
            )

        kappa = np.zeros(self.n_virtual)
        for ei_idx, (u, v) in enumerate(edges):
            if orc_values[ei_idx] < kappa[u]:
                kappa[u] = orc_values[ei_idx]
            if orc_values[ei_idx] < kappa[v]:
                kappa[v] = orc_values[ei_idx]

        return CurvatureField(
            points=X_virtual, points_unit=X_virtual_unit,
            means=means, stds=stds, kappa=kappa,
            edges=edges, orc_values=orc_values,
            nbrs_list=nbrs_list, ei=ei,
        )

    # ------------------------------------------------------------------
    # Topology detection
    # ------------------------------------------------------------------

    def _detect_topology(self, field: CurvatureField
                         ) -> Tuple[List[Basin], list]:
        """Identify basins (positive-ORC components) and saddle edges."""
        N = len(field.points)

        adj = lil_matrix((N, N), dtype=bool)
        for ei_idx, (u, v) in enumerate(field.edges):
            if field.orc_values[ei_idx] > 0:
                adj[u, v] = True
                adj[v, u] = True

        n_comp, labels = connected_components(adj.tocsr(), directed=False)

        basins: List[Basin] = []
        for c in range(n_comp):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            best_local = members[np.argmin(field.means[members])]
            basins.append(Basin(
                members=members,
                best_idx=int(best_local),
                best_point=field.points[best_local].copy(),
                best_predicted=float(field.means[best_local]),
                mean_uncertainty=float(field.stds[members].mean()),
            ))
        basins.sort(key=lambda b: b.best_predicted)

        saddle_edges = []
        for ei_idx, (u, v) in enumerate(field.edges):
            if field.orc_values[ei_idx] < 0 and labels[u] != labels[v]:
                midpoint = (field.points[u] + field.points[v]) / 2.0
                mid_std = (field.stds[u] + field.stds[v]) / 2.0
                saddle_edges.append({
                    'u': u, 'v': v,
                    'orc': field.orc_values[ei_idx],
                    'midpoint': midpoint,
                    'uncertainty': float(mid_std),
                    'basin_u': int(labels[u]),
                    'basin_v': int(labels[v]),
                })
        saddle_edges.sort(
            key=lambda s: abs(s['orc']) * s['uncertainty'], reverse=True)

        self._n_basins_history.append(len(basins))
        self._n_saddles_history.append(len(saddle_edges))
        return basins, saddle_edges

    # ------------------------------------------------------------------
    # Topology-aware acquisition (hybrid EI + ORC)
    # ------------------------------------------------------------------

    def _build_candidates(self, field: CurvatureField,
                          basins: List[Basin],
                          saddle_edges: list) -> List[AcquisitionCandidate]:
        """
        Build candidates from two sources:
        1. EI (primary) -- top Expected Improvement points from virtual pop
        2. ORC saddle exploration (secondary) -- only after enough data
        """
        candidates: List[AcquisitionCandidate] = []
        ei_max = max(float(field.ei.max()), 1e-20)

        # --- 1. EI exploitation (always active, primary driver) ---
        n_ei = max(self.batch_size * 5, 30)
        top_ei_idx = np.argsort(field.ei)[-n_ei:]
        for idx in top_ei_idx:
            score = 2.0 * (field.ei[idx] / ei_max)
            candidates.append(AcquisitionCandidate(
                point=field.points[idx].copy(),
                score=score,
                source='ei',
            ))

        # --- 2. ORC saddle exploration (activated after orc_start_frac) ---
        progress = self.fe_count / self.budget
        if progress > self.orc_start_frac and saddle_edges:
            orc_weight = min(1.0, (progress - self.orc_start_frac) * 3.0)
            n_saddle = max(self.batch_size, 5)
            seen_pairs = set()
            for s in saddle_edges:
                pair = (min(s['basin_u'], s['basin_v']),
                        max(s['basin_u'], s['basin_v']))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                score = orc_weight * abs(s['orc']) * (s['uncertainty'] / ei_max)
                candidates.append(AcquisitionCandidate(
                    point=s['midpoint'].copy(),
                    score=score,
                    source='saddle',
                ))
                if len(seen_pairs) >= n_saddle:
                    break

        return candidates

    def _select_batch(self, candidates: List[AcquisitionCandidate]
                      ) -> np.ndarray:
        """Greedy batch selection with distance penalization."""
        if not candidates:
            sampler = LatinHypercube(d=self.dim, seed=self.rng)
            return self._from_unit(sampler.random(n=self.batch_size))

        X_existing = np.array(self.X_eval)
        # Length scale in original space from fitted GP kernel
        try:
            ls_unit = np.mean(self.gp.kernel_.get_params().get(
                'k1__length_scale', 0.2))
            ls = ls_unit * (self.ub - self.lb)
        except Exception:
            ls = (self.ub - self.lb) * 0.2

        selected = []
        remaining = list(candidates)

        for _ in range(self.batch_size):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = 0

            all_existing = (np.vstack([X_existing] + [np.array(selected)])
                            if selected else X_existing)

            for ci, cand in enumerate(remaining):
                dists = np.linalg.norm(all_existing - cand.point, axis=1)
                min_dist = float(dists.min())
                penalty = 1.0 - np.exp(-min_dist / max(ls, 1e-6))
                penalized = cand.score * penalty

                if penalized > best_score:
                    best_score = penalized
                    best_idx = ci

            selected.append(remaining[best_idx].point.copy())
            remaining.pop(best_idx)

        return np.array(selected)

    # ------------------------------------------------------------------
    # Acquisition step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """One iteration of ORC-BO."""
        self._step_count += 1
        self._fit_surrogate()

        progress = self.fe_count / self.budget
        use_orc = progress > self.orc_start_frac

        if use_orc:
            recompute = (self._cached_field is None
                         or self._step_count % self.orc_period == 1)
            if recompute:
                field = self._compute_curvature_field()
                basins, saddle_edges = self._detect_topology(field)
                self._cached_field = field
                self._cached_basins = basins
                self._cached_saddles = saddle_edges
            else:
                field = self._cached_field
                basins = self._cached_basins
                saddle_edges = self._cached_saddles
        else:
            # EI-only phase: still need predictions on virtual points
            sampler = LatinHypercube(d=self.dim, seed=self.rng)
            X_virtual_unit = sampler.random(n=self.n_virtual)
            X_virtual = self._from_unit(X_virtual_unit)
            means, stds = self.gp.predict(X_virtual_unit, return_std=True)
            stds = np.maximum(stds, 1e-10)
            ei = self._compute_ei(means, stds)
            field = CurvatureField(
                points=X_virtual, points_unit=X_virtual_unit,
                means=means, stds=stds,
                kappa=np.zeros(len(X_virtual)), edges=[],
                orc_values=np.array([]),
                nbrs_list=[], ei=ei,
            )
            basins, saddle_edges = [], []

        batch = self._select_batch(
            self._build_candidates(field, basins, saddle_edges))
        batch = np.clip(batch, self.lb, self.ub)

        for x in batch:
            if self.fe_count >= self.budget:
                break
            y = float(self.problem.evaluate(x))
            self.X_eval.append(x.copy())
            self.y_eval.append(y)
            self.fe_count += 1
            if y < self.best_fitness:
                self.best_fitness = y
                self.best_solution = x.copy()

        self.convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_fitness

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> Tuple[np.ndarray, float]:
        """Run until budget exhausted."""
        self._initialize()
        while self.fe_count < self.budget:
            self.step()
        return self.best_solution, self.best_fitness

    def get_run_stats(self) -> dict:
        return {
            'best_fitness': self.best_fitness,
            'fe_count': self.fe_count,
            'n_iterations': len(self.convergence_log),
            'n_basins_final': (self._n_basins_history[-1]
                               if self._n_basins_history else 0),
            'n_saddles_final': (self._n_saddles_history[-1]
                                if self._n_saddles_history else 0),
            'avg_basins': (float(np.mean(self._n_basins_history))
                           if self._n_basins_history else 0),
        }

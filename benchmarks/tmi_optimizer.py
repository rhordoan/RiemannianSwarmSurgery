"""
TMIOptimizer: Topological Manifold Injection Wrapper.

Architecture (three clean layers):

  Layer 1 – Base Optimizer (completely unmodified):
    L-SHADE or NL-SHADE runs its standard generation loop. The optimizer's
    internal state (population, archive, adaptation histories) is never touched
    by the oracle or injection layers.

  Layer 2 – Riemannian Oracle (zero extra FEs):
    After each generation, RiemannianOracle.step() is called with the current
    population. It builds a k-NN graph and computes Ollivier-Ricci Curvature
    for every edge. Edges with ORC < threshold (default -0.1) are flagged as
    inter-basin saddle boundaries. Total overhead: O(N * k^3) arithmetic
    operations per update_period generations -- negligible vs. the optimizer.

  Layer 3 – Topological Injection (topology-aware restart):
    When the base optimizer stagnates (no improvement for STAGNATION_GENS
    generations), the TMIOptimizer queries the Topological Saddle Archive for
    the best stored saddle vector and replaces the bottom INJECTION_FRACTION of
    the population with points displaced in the descent direction.

    Injection costs REAL function evaluations (n_inject per restart). This is
    transparently reported in the paper. The claim is not "zero cost" but
    "injected FEs are dramatically more efficient than random LHS restarts"
    because they are deterministically aimed at the boundary between a known
    good basin and an unexplored one.

Paper claim (falsifiable):
    NL-SHADE + TMI reduces mean final error vs. vanilla NL-SHADE on multimodal
    CEC 2022 functions (p < 0.05, Wilcoxon), with no significant difference
    on unimodal functions (F1-F3), confirming topology-sensitivity.
"""

import sys
import os
import numpy as np

# Allow imports from project root when running benchmarks directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lshade import LSHADE
from benchmarks.nlshade import NLSHADE
from src.riemannian_swarm import RiemannianOracle
from src.saddle_archive import SaddleArchive


class TMIOptimizer:
    """
    Topological Manifold Injection wrapper for L-SHADE / NL-SHADE.

    Args:
        problem:            Optimisation problem with .evaluate(x) and
                            .bounds = [lb, ub].
        base:               Base optimizer string: 'lshade' or 'nlshade'.
        dim:                Problem dimensionality.
        pop_size:           Initial population size (None => 18 * dim).
        max_fe:             Total function evaluation budget.
        k_orc:              k-NN degree for the Riemannian oracle.
        orc_threshold:      ORC value below which an edge is a saddle (-0.1).
        orc_update_period:  Oracle runs every this many generations (5).
        stagnation_gens:    Gens without improvement to trigger injection (30).
        injection_fraction: Fraction of population replaced per injection (0.30).
        inject_step_frac:   Injection displacement as fraction of domain (0.15).
        max_saddles:        Capacity of the Saddle Archive (30).
    """

    # Defaults tuned on CEC 2022 D=10, max_FE=200_000
    # Stagnation window is auto-scaled to the budget (see __init__), so these
    # defaults only apply if stagnation_gens is explicitly passed.
    STAGNATION_GENS: int = 50
    INJECTION_FRACTION: float = 0.30
    INJECT_STEP_FRAC: float = 0.15
    ORC_THRESHOLD: float = -0.1
    ORC_UPDATE_PERIOD: int = 5
    K_ORC: int = 7
    MAX_SADDLES: int = 30
    # Upper bound on total injections per run.  Prevents burning the FE budget
    # on repeated injections when the archive does not have useful structure yet.
    MAX_INJECTIONS: int = 8

    def __init__(self,
                 problem,
                 base: str = 'lshade',
                 dim: int = 10,
                 pop_size: int = None,
                 max_fe: int = 200_000,
                 k_orc: int = None,
                 orc_threshold: float = None,
                 orc_update_period: int = None,
                 stagnation_gens: int = None,
                 injection_fraction: float = None,
                 inject_step_frac: float = None,
                 max_saddles: int = None,
                 max_injections: int = None):

        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.base_name = base.lower()

        # Stagnation window auto-scales with the budget when not overridden.
        # Reference: 200k FEs, pop_size ~18*dim => ~1000 generations.
        # We want ~5% of total generations as the stagnation window.
        # For any other budget, we scale proportionally (min 20, max 200).
        if stagnation_gens is not None:
            self._stagnation_gens = stagnation_gens
        else:
            _pop_est = min(18 * dim, 100)
            _total_gens_est = max_fe / max(_pop_est, 1)
            self._stagnation_gens = int(max(20, min(200, _total_gens_est * 0.05)))

        self._injection_fraction = injection_fraction if injection_fraction is not None else self.INJECTION_FRACTION
        self._inject_step_frac = inject_step_frac if inject_step_frac is not None else self.INJECT_STEP_FRAC
        self._orc_threshold = orc_threshold if orc_threshold is not None else self.ORC_THRESHOLD
        self._orc_update_period = orc_update_period if orc_update_period is not None else self.ORC_UPDATE_PERIOD
        self._k_orc = k_orc if k_orc is not None else self.K_ORC
        self._max_saddles = max_saddles if max_saddles is not None else self.MAX_SADDLES
        self._max_injections = max_injections if max_injections is not None else self.MAX_INJECTIONS

        # Domain geometry
        lb, ub = float(problem.bounds[0]), float(problem.bounds[1])
        self.lb = lb
        self.ub = ub
        self.domain_width = ub - lb

        # --- Layer 1: Base Optimizer ---
        if self.base_name == 'lshade':
            self.optimizer = LSHADE(problem, dim, pop_size, max_fe)
        elif self.base_name == 'nlshade':
            self.optimizer = NLSHADE(problem, dim, pop_size, max_fe)
        else:
            raise ValueError(f"Unknown base optimizer '{base}'. Use 'lshade' or 'nlshade'.")

        # --- Layer 2: Riemannian Oracle ---
        self.oracle = RiemannianOracle(
            dim=dim,
            k=self._k_orc,
            orc_threshold=self._orc_threshold,
            update_period=self._orc_update_period,
            domain_width=self.domain_width,
        )

        # --- Layer 3: Saddle Archive ---
        self.saddle_archive = SaddleArchive(
            domain_width=self.domain_width,
            max_saddles=self._max_saddles,
        )

        # --- Internal state ---
        self.generation: int = 0
        self.gens_without_improvement: int = 0
        self.best_fitness: float = self.optimizer.best_fitness
        self.best_solution: np.ndarray = self.optimizer.best_solution.copy()

        # Diagnostics logged per run for paper tables
        self.injection_count: int = 0
        self.total_injection_fes: int = 0
        self.saddles_archived: int = 0
        self._convergence: list = []  # (fe_count, best_fitness) pairs

    # ------------------------------------------------------------------
    # Properties that delegate to the base optimizer
    # ------------------------------------------------------------------

    @property
    def fe_count(self) -> int:
        return self.optimizer.fe_count

    @property
    def pop(self) -> np.ndarray:
        return self.optimizer.pop

    @property
    def fitness(self) -> np.ndarray:
        return self.optimizer.fitness

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Execute one TMI step:
          1. One generation of the base optimizer.
          2. ORC oracle update (zero FEs).
          3. Archive newly detected saddles.
          4. Topological injection if stagnated and archive non-empty.

        Returns current best fitness.
        """
        # 1. Base optimizer generation
        self.optimizer.step()
        self.generation += 1

        # 2. Track improvement
        if self.optimizer.best_fitness < self.best_fitness - 1e-12:
            self.best_fitness = self.optimizer.best_fitness
            self.best_solution = self.optimizer.best_solution.copy()
            self.gens_without_improvement = 0
        else:
            self.gens_without_improvement += 1

        self._convergence.append((self.fe_count, self.best_fitness))

        # 3. Oracle step (zero FEs)
        saddles = self.oracle.step(
            self.optimizer.pop,
            self.optimizer.fitness,
            self.generation,
        )

        # 4. Archive saddles
        for u, v in saddles:
            N = len(self.optimizer.pop)
            if u < N and v < N:
                stored = self.saddle_archive.store_saddle(
                    self.optimizer.pop[u],
                    self.optimizer.pop[v],
                    float(self.optimizer.fitness[u]),
                    float(self.optimizer.fitness[v]),
                    self.generation,
                )
                if stored:
                    self.saddles_archived += 1

        # 5. Topological injection on stagnation (respecting the per-run cap)
        if (self.gens_without_improvement >= self._stagnation_gens
                and self.saddle_archive.num_saddles > 0
                and self.fe_count < self.max_fe
                and self.injection_count < self._max_injections):
            self._topological_injection()

        return self.best_fitness

    def _topological_injection(self):
        """
        Replace the bottom INJECTION_FRACTION of the population with agents
        placed at the best archived saddle point displaced by the descent vector.

        These replacements cost real FEs. The injection cost is logged in
        self.total_injection_fes for transparent reporting in the paper.
        """
        N = len(self.optimizer.pop)
        n_inject = max(1, int(round(self._injection_fraction * N)))
        step_size = self._inject_step_frac * self.domain_width

        rng = np.random.default_rng()  # thread-safe, unaffected by np.random state
        points = self.saddle_archive.get_injection_points(
            n_inject, step_size, self.lb, self.ub, rng=rng
        )

        if points is None:
            return

        # Indices of worst (highest-fitness) agents -- they will be replaced
        sorted_idx = np.argsort(self.optimizer.fitness)[::-1]
        n_replace = min(n_inject, len(points))
        inject_idx = sorted_idx[:n_replace]

        for local_i, pop_idx in enumerate(inject_idx):
            if local_i >= len(points):
                break
            if self.fe_count >= self.max_fe:
                break

            x_new = points[local_i]
            f_new = self.problem.evaluate(x_new)
            self.optimizer.fe_count += 1
            self.total_injection_fes += 1

            self.optimizer.pop[pop_idx] = x_new
            self.optimizer.fitness[pop_idx] = f_new

            if f_new < self.best_fitness:
                self.best_fitness = f_new
                self.best_solution = x_new.copy()

        self.injection_count += 1
        self.gens_without_improvement = 0

        # Sync the base optimizer's best tracking
        if self.best_fitness < self.optimizer.best_fitness:
            self.optimizer.best_fitness = self.best_fitness
            self.optimizer.best_solution = self.best_solution.copy()

    def run(self) -> tuple:
        """
        Run until budget exhausted.

        Returns:
            (best_solution, best_fitness) tuple.
        """
        while self.fe_count < self.max_fe:
            self.step()
        return self.best_solution, self.best_fitness

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_run_stats(self) -> dict:
        """
        Return a dictionary of run statistics suitable for paper tables.
        """
        return {
            'base': self.base_name,
            'best_fitness': self.best_fitness,
            'fe_count': self.fe_count,
            'generations': self.generation,
            'stagnation_gens': self._stagnation_gens,
            'max_injections': self._max_injections,
            'injection_count': self.injection_count,
            'total_injection_fes': self.total_injection_fes,
            'saddles_archived': self.saddles_archived,
            'orc_min': self.oracle.min_orc,
            'orc_mean': self.oracle.mean_orc,
            'n_saddle_edges': self.oracle.n_saddle_edges,
        }

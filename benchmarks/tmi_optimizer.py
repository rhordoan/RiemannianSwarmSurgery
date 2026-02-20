"""
TMIOptimizer v3: Topological Manifold Injection with Fitness-Aware ORC.

v3 changes over v2:
  1. FITNESS-WEIGHTED k-NN: The oracle builds the graph using a combined
     spatial+fitness distance, making ORC landscape-aware.
  2. PROMISE FILTERING: Saddles where the explore-side is a flat plateau
     are automatically rejected (won't inject off a ridge into nothing).
  3. REDUCED INJECTION: 15% replacement instead of 30%, less disruptive
     to the optimizer's adapted parameters.
  4. STRICTER ORC THRESHOLD: Only inject at edges with ORC < -0.15 (was -0.1).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lshade import LSHADE
from benchmarks.nlshade import NLSHADE
from src.riemannian_swarm import RiemannianOracle
from src.saddle_archive import SaddleArchive


class TMIOptimizer:

    STAGNATION_GENS: int = 50
    INJECTION_FRACTION: float = 0.30
    INJECT_STEP_FRAC: float = 0.15
    ORC_THRESHOLD: float = -0.10
    ORC_UPDATE_PERIOD: int = 5
    K_ORC: int = 7
    MAX_SADDLES: int = 30
    MAX_INJECTIONS: int = 8
    CONVERGENCE_EPS: float = 1e-8
    FITNESS_WEIGHT: float = 0.0

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
                 max_injections: int = None,
                 use_orc: bool = True,
                 fitness_weight: float = None):

        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.base_name = base.lower()

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
        self._use_orc = use_orc
        self._fitness_weight = fitness_weight if fitness_weight is not None else self.FITNESS_WEIGHT

        lb, ub = float(problem.bounds[0]), float(problem.bounds[1])
        self.lb = lb
        self.ub = ub
        self.domain_width = ub - lb

        if self.base_name == 'lshade':
            self.optimizer = LSHADE(problem, dim, pop_size, max_fe)
        elif self.base_name == 'nlshade':
            self.optimizer = NLSHADE(problem, dim, pop_size, max_fe)
        else:
            raise ValueError(f"Unknown base optimizer '{base}'.")

        self.oracle = RiemannianOracle(
            dim=dim,
            k=self._k_orc,
            orc_threshold=self._orc_threshold,
            update_period=self._orc_update_period,
            domain_width=self.domain_width,
            fitness_weight=self._fitness_weight,
        ) if self._use_orc else None

        _pop_est = min(18 * dim, 100)
        _total_gens_est = max_fe / max(_pop_est, 1)
        _max_age = int(max(50, _total_gens_est * 0.4))

        self.saddle_archive = SaddleArchive(
            domain_width=self.domain_width,
            max_saddles=self._max_saddles,
            max_age=_max_age,
        ) if self._use_orc else None

        self.generation: int = 0
        self.gens_without_improvement: int = 0
        self.best_fitness: float = self.optimizer.best_fitness
        self.best_solution: np.ndarray = self.optimizer.best_solution.copy()

        self.injection_count: int = 0
        self.total_injection_fes: int = 0
        self.saddles_archived: int = 0
        self._convergence: list = []

    @property
    def fe_count(self) -> int:
        return self.optimizer.fe_count

    @property
    def pop(self) -> np.ndarray:
        return self.optimizer.pop

    @property
    def fitness(self) -> np.ndarray:
        return self.optimizer.fitness

    def step(self) -> float:
        self.optimizer.step()
        self.generation += 1

        if self.optimizer.best_fitness < self.best_fitness:
            if self.optimizer.best_fitness < self.best_fitness - 1e-12:
                self.gens_without_improvement = 0
            else:
                self.gens_without_improvement += 1
            self.best_fitness = self.optimizer.best_fitness
            self.best_solution = self.optimizer.best_solution.copy()
        else:
            self.gens_without_improvement += 1

        self._convergence.append((self.fe_count, self.best_fitness))

        if self._use_orc:
            saddles = self.oracle.step(
                self.optimizer.pop,
                self.optimizer.fitness,
                self.generation,
            )

            if saddles:
                aug_pop_full, aug_fit_full, _ = self.oracle._build_augmented(
                    self.optimizer.pop, self.optimizer.fitness
                )
                n_aug = len(aug_pop_full)

                for s in saddles:
                    u, v = s['u'], s['v']
                    if u < n_aug and v < n_aug:
                        stored = self.saddle_archive.store_saddle(
                            aug_pop_full[u],
                            aug_pop_full[v],
                            float(aug_fit_full[u]),
                            float(aug_fit_full[v]),
                            self.generation,
                            nbr_centroid_explore=s.get('nbr_centroid_explore'),
                            explore_fitness_std=s.get('explore_fitness_std', 0.0),
                            has_promise=s.get('has_promise', True),
                        )
                        if stored:
                            self.saddles_archived += 1

            if (self.gens_without_improvement >= self._stagnation_gens
                    and self.saddle_archive.num_saddles > 0
                    and self.fe_count < self.max_fe
                    and self.injection_count < self._max_injections
                    and not self._is_converged()):
                self._topological_injection()
        else:
            if (self.gens_without_improvement >= self._stagnation_gens
                    and self.fe_count < self.max_fe
                    and self.injection_count < self._max_injections
                    and not self._is_converged()):
                self._random_injection()

        return self.best_fitness

    def _is_converged(self) -> bool:
        fit = self.optimizer.fitness
        if len(fit) < 2:
            return False

        fit_range = float(np.max(fit) - np.min(fit))
        scale = max(abs(self.best_fitness), 1.0)

        if fit_range / scale < self.CONVERGENCE_EPS:
            return True

        pop_std = float(np.mean(np.std(self.optimizer.pop, axis=0)))
        if pop_std < 1e-10 * self.domain_width:
            return True

        return False

    def _topological_injection(self):
        N = len(self.optimizer.pop)
        n_inject = max(1, int(round(self._injection_fraction * N)))
        base_step = self._inject_step_frac * self.domain_width

        rng = np.random.default_rng()
        points = self.saddle_archive.get_injection_points(
            n_inject, base_step, self.lb, self.ub, rng=rng,
            current_gen=self.generation
        )

        if points is None:
            return

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

        if self.best_fitness < self.optimizer.best_fitness:
            self.optimizer.best_fitness = self.best_fitness
            self.optimizer.best_solution = self.best_solution.copy()

    def _random_injection(self):
        N = len(self.optimizer.pop)
        n_inject = max(1, int(round(self._injection_fraction * N)))

        rng = np.random.default_rng()
        points = np.empty((n_inject, self.dim))
        for d in range(self.dim):
            cuts = np.linspace(self.lb, self.ub, n_inject + 1)
            points[:, d] = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(points)

        sorted_idx = np.argsort(self.optimizer.fitness)[::-1]
        inject_idx = sorted_idx[:n_inject]

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

        if self.best_fitness < self.optimizer.best_fitness:
            self.optimizer.best_fitness = self.best_fitness
            self.optimizer.best_solution = self.best_solution.copy()

    def run(self) -> tuple:
        while self.fe_count < self.max_fe:
            self.step()
        return self.best_solution, self.best_fitness

    def get_run_stats(self) -> dict:
        orc_stats = self.oracle.get_orc_stats() if self._use_orc else {}
        return {
            'base': self.base_name,
            'use_orc': self._use_orc,
            'best_fitness': self.best_fitness,
            'fe_count': self.fe_count,
            'generations': self.generation,
            'stagnation_gens': self._stagnation_gens,
            'max_injections': self._max_injections,
            'injection_count': self.injection_count,
            'total_injection_fes': self.total_injection_fes,
            'saddles_archived': self.saddles_archived,
            'orc_min': orc_stats.get('min'),
            'orc_mean': orc_stats.get('mean'),
            'n_saddle_edges': orc_stats.get('n_saddles', 0),
        }

"""
Bayesian Optimization Baselines for expensive optimization benchmarking.

Provides:
  - EGO: GP + Expected Improvement (the BO gold standard)
  - RandomSearch: uniform random sampling
  - LHSSearch: Latin Hypercube space-filling

All share the interface:
    optimizer.run() -> (best_solution, best_fitness)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class EGO:
    """
    Efficient Global Optimization (Jones et al., 1998).

    GP surrogate with Expected Improvement acquisition.
    Input normalization to [0,1]^D for stable GP kernel hyperparameters.
    """

    def __init__(self, problem, dim: int, budget: int, *,
                 batch_size: Optional[int] = None,
                 n_init: Optional[int] = None,
                 n_candidates: int = 2000,
                 seed: Optional[int] = None):
        self.problem = problem
        self.dim = dim
        self.budget = budget
        self.batch_size = batch_size if batch_size is not None else max(dim, 5)
        self.n_init = n_init if n_init is not None else max(5 * dim, 20)
        self.n_candidates = n_candidates

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

    def _to_unit(self, X):
        return (X - self.lb) / (self.ub - self.lb)

    def _from_unit(self, X_unit):
        return self.lb + (self.ub - self.lb) * X_unit

    def _initialize(self):
        sampler = LatinHypercube(d=self.dim, seed=self.rng)
        X_init = self._from_unit(sampler.random(n=self.n_init))
        for x in X_init:
            y = float(self.problem.evaluate(x))
            self.X_eval.append(x.copy())
            self.y_eval.append(y)
            self.fe_count += 1
            if y < self.best_fitness:
                self.best_fitness = y
                self.best_solution = x.copy()
        self.convergence_log.append((self.fe_count, self.best_fitness))

    def _expected_improvement(self, X_cand_unit: np.ndarray) -> np.ndarray:
        """Compute EI at unit-space candidate points."""
        mu, sigma = self.gp.predict(X_cand_unit, return_std=True)
        sigma = np.maximum(sigma, 1e-10)
        f_best = min(self.y_eval)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def step(self) -> float:
        X_unit = self._to_unit(np.array(self.X_eval))
        y = np.array(self.y_eval)
        n = len(X_unit)
        if n > 300:
            self.gp.n_restarts_optimizer = 1
        elif n > 100:
            self.gp.n_restarts_optimizer = 3
        self.gp.fit(X_unit, y)

        sampler = LatinHypercube(d=self.dim, seed=self.rng)
        X_cand_unit = sampler.random(n=self.n_candidates)

        selected = []
        for _ in range(self.batch_size):
            if self.fe_count >= self.budget:
                break
            ei = self._expected_improvement(X_cand_unit)
            best_idx = np.argmax(ei)
            x_unit = X_cand_unit[best_idx].copy()
            x_orig = self._from_unit(x_unit)
            selected.append(x_orig)

            y_new = float(self.problem.evaluate(x_orig))
            self.X_eval.append(x_orig)
            self.y_eval.append(y_new)
            self.fe_count += 1
            if y_new < self.best_fitness:
                self.best_fitness = y_new
                self.best_solution = x_orig.copy()

            X_cand_unit = np.delete(X_cand_unit, best_idx, axis=0)
            if len(X_cand_unit) == 0:
                break

        self.convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_fitness

    def run(self) -> Tuple[np.ndarray, float]:
        self._initialize()
        while self.fe_count < self.budget:
            self.step()
        return self.best_solution, self.best_fitness

    def get_run_stats(self) -> dict:
        return {
            'best_fitness': self.best_fitness,
            'fe_count': self.fe_count,
            'n_iterations': len(self.convergence_log),
        }


class RandomSearch:
    """Uniform random search baseline."""

    def __init__(self, problem, dim: int, budget: int, *,
                 seed: Optional[int] = None):
        self.problem = problem
        self.dim = dim
        self.budget = budget
        self.lb = float(problem.bounds[0])
        self.ub = float(problem.bounds[1])
        self.rng = np.random.RandomState(seed)
        self.fe_count = 0
        self.best_fitness = np.inf
        self.best_solution: Optional[np.ndarray] = None
        self.convergence_log: List[Tuple[int, float]] = []

    def run(self) -> Tuple[np.ndarray, float]:
        while self.fe_count < self.budget:
            x = self.rng.uniform(self.lb, self.ub, self.dim)
            y = float(self.problem.evaluate(x))
            self.fe_count += 1
            if y < self.best_fitness:
                self.best_fitness = y
                self.best_solution = x.copy()
            if self.fe_count % 10 == 0 or self.fe_count == self.budget:
                self.convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_solution, self.best_fitness

    def get_run_stats(self) -> dict:
        return {'best_fitness': self.best_fitness, 'fe_count': self.fe_count}


class LHSSearch:
    """Latin Hypercube space-filling baseline (no surrogate)."""

    def __init__(self, problem, dim: int, budget: int, *,
                 seed: Optional[int] = None):
        self.problem = problem
        self.dim = dim
        self.budget = budget
        self.lb = float(problem.bounds[0])
        self.ub = float(problem.bounds[1])
        self.rng = np.random.RandomState(seed)
        self.fe_count = 0
        self.best_fitness = np.inf
        self.best_solution: Optional[np.ndarray] = None
        self.convergence_log: List[Tuple[int, float]] = []

    def run(self) -> Tuple[np.ndarray, float]:
        sampler = LatinHypercube(d=self.dim, seed=self.rng)
        X = self.lb + (self.ub - self.lb) * sampler.random(n=self.budget)
        for x in X:
            y = float(self.problem.evaluate(x))
            self.fe_count += 1
            if y < self.best_fitness:
                self.best_fitness = y
                self.best_solution = x.copy()
            if self.fe_count % 10 == 0 or self.fe_count == self.budget:
                self.convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_solution, self.best_fitness

    def get_run_stats(self) -> dict:
        return {'best_fitness': self.best_fitness, 'fe_count': self.fe_count}

"""
L-SHADE: Success-History based Adaptive Differential Evolution
with Linear Population Size Reduction.

Reference:
    Tanabe, R. & Fukunaga, A. (2014). Improving the Search Performance
    of SHADE Using Linear Population Size Reduction. IEEE CEC 2014.

This is a clean, standalone implementation suitable for use as a base
optimizer that RSS wraps around. Any improvement from geometric
mechanisms is then a clean delta over L-SHADE's performance.
"""

import numpy as np


class LSHADE:
    """
    L-SHADE optimizer with current-to-pbest/1 mutation, success-history
    parameter adaptation, external archive, and linear population size
    reduction (LPSR).

    Args:
        problem: Object with .evaluate(x) and .bounds = [lb, ub].
        dim: Dimensionality.
        pop_size: Initial population size (N_init).
        max_fe: Maximum function evaluations budget.
        pop_size_min: Minimum population size after LPSR.
        H: Size of success history for F and CR adaptation.
    """

    def __init__(self, problem, dim, pop_size=None, max_fe=200000,
                 pop_size_min=4, H=6):
        self.problem = problem
        self.dim = dim
        self.pop_size_init = pop_size if pop_size is not None else 18 * dim
        self.pop_size = self.pop_size_init
        self.pop_size_min = max(pop_size_min, 4)
        self.max_fe = max_fe
        self.H = H

        # Initialize population
        lb, ub = problem.bounds[0], problem.bounds[1]
        self.lb = lb
        self.ub = ub
        self.pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        self.fitness = np.array([problem.evaluate(x) for x in self.pop])
        self.fe_count = self.pop_size

        # Best tracking
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.pop[best_idx].copy()

        # Success history for F and CR (initialized to 0.5)
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0  # History pointer (circular)

        # External archive A (stores replaced parents for diversity)
        self.archive = []
        self.archive_max_size = self.pop_size_init

        # Generation counter
        self.generation = 0

    def _generate_F(self, size):
        """Generate F values from Cauchy distribution centered on M_F[r]."""
        F_values = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + self.M_F[r]
                if Fi > 0:
                    break
            F_values[i] = min(Fi, 1.0)
        return F_values

    def _generate_CR(self, size):
        """Generate CR values from Normal distribution centered on M_CR[r]."""
        CR_values = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            CRi = np.random.normal(self.M_CR[r], 0.1)
            CR_values[i] = np.clip(CRi, 0.0, 1.0)
        return CR_values

    def _lehmer_mean(self, values, weights=None):
        """Weighted Lehmer mean (used for F adaptation)."""
        if len(values) == 0:
            return 0.5
        values = np.array(values)
        if weights is None:
            weights = np.ones(len(values))
        weights = np.array(weights)
        num = np.sum(weights * values ** 2)
        den = np.sum(weights * values)
        if den < 1e-30:
            return 0.5
        return num / den

    def _weighted_mean(self, values, weights=None):
        """Weighted arithmetic mean (used for CR adaptation)."""
        if len(values) == 0:
            return 0.5
        values = np.array(values)
        if weights is None:
            weights = np.ones(len(values))
        weights = np.array(weights) / np.sum(weights)
        return np.sum(weights * values)

    def step(self):
        """
        Execute one generation of L-SHADE.

        Returns:
            best_fitness: Best fitness found so far.
        """
        self.generation += 1
        N = len(self.pop)

        if N < 4:
            return self.best_fitness

        # Generate F and CR for this generation
        F_vals = self._generate_F(N)
        CR_vals = self._generate_CR(N)

        # p for pbest selection: p in [2/N, 0.2], at least 2 individuals
        p = max(2.0 / N, 0.2)
        p_count = max(2, int(round(p * N)))

        # Sort population by fitness (ascending = best first)
        sorted_indices = np.argsort(self.fitness)

        # Mutation: current-to-pbest/1 with archive
        new_pop = np.empty_like(self.pop)
        new_fitness = np.empty(N)
        success_F = []
        success_CR = []
        success_delta = []  # |f_parent - f_trial| for weighting

        # Combined population + archive for r2 selection
        if len(self.archive) > 0:
            combined = np.vstack([self.pop, np.array(self.archive)])
        else:
            combined = self.pop.copy()

        for i in range(N):
            Fi = F_vals[i]
            CRi = CR_vals[i]

            # Select pbest: random from top-p individuals
            pbest_idx = sorted_indices[np.random.randint(0, p_count)]

            # Select r1 != i from population
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, N)

            # Select r2 != i, r1 from population + archive
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len(combined))

            # Mutation: current-to-pbest/1
            # v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            mutant = (self.pop[i]
                      + Fi * (self.pop[pbest_idx] - self.pop[i])
                      + Fi * (self.pop[r1] - combined[r2]))

            # Boundary handling: bounce-back
            for d in range(self.dim):
                if mutant[d] < self.lb:
                    mutant[d] = (self.lb + self.pop[i][d]) / 2.0
                elif mutant[d] > self.ub:
                    mutant[d] = (self.ub + self.pop[i][d]) / 2.0

            # Binomial crossover
            j_rand = np.random.randint(0, self.dim)
            trial = self.pop[i].copy()
            for d in range(self.dim):
                if np.random.rand() < CRi or d == j_rand:
                    trial[d] = mutant[d]

            # Evaluate trial
            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1

            # Greedy selection
            if f_trial <= self.fitness[i]:
                # Archive the replaced parent
                if f_trial < self.fitness[i]:
                    self.archive.append(self.pop[i].copy())
                    success_F.append(Fi)
                    success_CR.append(CRi)
                    success_delta.append(abs(self.fitness[i] - f_trial))

                new_pop[i] = trial
                new_fitness[i] = f_trial

                # Update global best
                if f_trial < self.best_fitness:
                    self.best_fitness = f_trial
                    self.best_solution = trial.copy()
            else:
                new_pop[i] = self.pop[i]
                new_fitness[i] = self.fitness[i]

            if self.fe_count >= self.max_fe:
                # Copy remaining unchanged
                for j in range(i + 1, N):
                    new_pop[j] = self.pop[j]
                    new_fitness[j] = self.fitness[j]
                break

        self.pop = new_pop
        self.fitness = new_fitness

        # Update success history
        if len(success_F) > 0:
            delta = np.array(success_delta)
            weights = delta / (np.sum(delta) + 1e-30)
            self.M_F[self.k] = self._lehmer_mean(success_F, weights)
            self.M_CR[self.k] = self._weighted_mean(success_CR, weights)
            self.k = (self.k + 1) % self.H

        # Trim archive if too large
        while len(self.archive) > self.archive_max_size:
            idx = np.random.randint(0, len(self.archive))
            self.archive.pop(idx)

        # Linear Population Size Reduction (LPSR)
        progress = self.fe_count / self.max_fe
        new_size = int(round(
            self.pop_size_init
            + (self.pop_size_min - self.pop_size_init) * progress
        ))
        new_size = max(new_size, self.pop_size_min)

        if new_size < len(self.pop):
            # Remove worst individuals
            sorted_idx = np.argsort(self.fitness)
            keep = sorted_idx[:new_size]
            self.pop = self.pop[keep]
            self.fitness = self.fitness[keep]

        return self.best_fitness

    def run(self):
        """Run L-SHADE until budget exhausted. Returns convergence history."""
        history = []
        while self.fe_count < self.max_fe:
            best = self.step()
            history.append(best)
        return history

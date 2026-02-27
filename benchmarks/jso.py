"""
jSO: Improved iL-SHADE with weighted mutation strategy.

Reference:
    Brest, J., Maucec, M. S., & Boskovic, B. (2017).
    Single Objective Real-Parameter Optimization: Algorithm jSO.
    IEEE Congress on Evolutionary Computation (CEC), 1311-1318.

    Winner of the CEC 2017 competition (best DE, 2nd overall).

Key innovations over L-SHADE:
  1. Weighted mutation: v = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)
     where Fw = 0.7F / 0.8F / 1.2F depending on FES progress.
  2. CR floor: CR >= 0.7 early, CR >= 0.6 mid-search.
  3. F cap: F <= 0.7 in the first 60% of budget.
  4. Memory init: M_F=0.3, M_CR=0.8 (more conservative F, aggressive CR).
  5. Memory slot H is overridden to 0.9 at runtime.
  6. Memory update averages new Lehmer mean with old value (damping).
  7. Adaptive p-best rate: decays multiplicatively during PSR.

Implementation faithfully follows the C++ reference code by Brest et al.
"""

import math
import numpy as np


def _lhs_init(dim, n, lb, ub):
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=dim)
        unit = sampler.random(n=n)
        return lb + (ub - lb) * unit
    except Exception:
        return np.random.uniform(lb, ub, (n, dim))


class JSO:
    """
    jSO optimizer: the CEC 2017 competition-winning DE variant.

    Args:
        problem:   Object with .evaluate(x)->float, .bounds=[lb, ub].
        dim:       Dimensionality.
        max_fe:    Maximum function evaluations.
        pop_size:  Initial pop size (default: round(25 * log(D+1) * sqrt(D))).
    """

    def __init__(self, problem, dim, max_fe=100_000, pop_size=None):
        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe

        lb, ub = problem.bounds[0], problem.bounds[1]
        self.lb = float(lb)
        self.ub = float(ub)

        self.H = 5
        self.pop_size_init = pop_size or round(
            25 * math.log(dim + 1) * math.sqrt(dim)
        )
        self.pop_size_min = 4
        self.pop_size = self.pop_size_init
        self.p_best_rate = 0.25
        self.arc_rate = 1.0

        self.pop = _lhs_init(dim, self.pop_size, self.lb, self.ub)
        self.fitness = np.array([problem.evaluate(x) for x in self.pop])
        self.fe_count = self.pop_size

        best_idx = int(np.argmin(self.fitness))
        self.best_fitness = float(self.fitness[best_idx])
        self.best_solution = self.pop[best_idx].copy()

        self.M_F = np.full(self.H, 0.3)
        self.M_CR = np.full(self.H, 0.8)
        self.k = 0

        self.archive = []
        self.arc_size = int(round(self.pop_size * self.arc_rate))

        self.generation = 0

    def step(self):
        self.generation += 1
        N = len(self.pop)
        if N < self.pop_size_min:
            return self.best_fitness

        progress = self.fe_count / self.max_fe

        sorted_idx = np.argsort(self.fitness)
        p_num = max(2, int(round(N * self.p_best_rate)))

        if self.archive:
            combined = np.vstack([self.pop, np.array(self.archive)])
        else:
            combined = self.pop.copy()
        n_combined = len(combined)

        new_pop = np.empty_like(self.pop)
        new_fitness = np.empty(N)
        success_F = []
        success_CR = []
        success_delta = []

        for i in range(N):
            r_mem = np.random.randint(0, self.H)
            if r_mem == self.H - 1:
                mu_f, mu_cr = 0.9, 0.9
            else:
                mu_f = self.M_F[r_mem]
                mu_cr = self.M_CR[r_mem]

            if mu_cr < 0:
                CRi = 0.0
            else:
                CRi = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
            if progress < 0.25 and CRi < 0.7:
                CRi = 0.7
            elif progress < 0.50 and CRi < 0.6:
                CRi = 0.6

            while True:
                Fi = np.random.standard_cauchy() * 0.1 + mu_f
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            if progress < 0.6 and Fi > 0.7:
                Fi = 0.7

            if progress < 0.2:
                jF = Fi * 0.7
            elif progress < 0.4:
                jF = Fi * 0.8
            else:
                jF = Fi * 1.2

            p_best_ind = sorted_idx[np.random.randint(0, p_num)]
            if progress < 0.5:
                attempts = 0
                while p_best_ind == i and attempts < 20:
                    p_best_ind = sorted_idx[np.random.randint(0, p_num)]
                    attempts += 1

            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, N)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, n_combined)

            mutant = (self.pop[i]
                      + jF * (self.pop[p_best_ind] - self.pop[i])
                      + Fi * (self.pop[r1] - combined[r2]))

            for d in range(self.dim):
                if mutant[d] < self.lb:
                    mutant[d] = (self.lb + self.pop[i][d]) * 0.5
                elif mutant[d] > self.ub:
                    mutant[d] = (self.ub + self.pop[i][d]) * 0.5

            j_rand = np.random.randint(0, self.dim)
            trial = self.pop[i].copy()
            mask = np.random.rand(self.dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1

            if f_trial <= self.fitness[i]:
                if f_trial < self.fitness[i]:
                    success_delta.append(abs(self.fitness[i] - f_trial))
                    success_F.append(Fi)
                    success_CR.append(CRi)
                    if len(self.archive) < self.arc_size:
                        self.archive.append(self.pop[i].copy())
                    elif self.arc_size > 0:
                        self.archive[np.random.randint(0, self.arc_size)] = \
                            self.pop[i].copy()

                new_pop[i] = trial
                new_fitness[i] = f_trial

                if f_trial < self.best_fitness:
                    self.best_fitness = f_trial
                    self.best_solution = trial.copy()
            else:
                new_pop[i] = self.pop[i]
                new_fitness[i] = self.fitness[i]

            if self.fe_count >= self.max_fe:
                for j in range(i + 1, N):
                    new_pop[j] = self.pop[j]
                    new_fitness[j] = self.fitness[j]
                break

        self.pop = new_pop
        self.fitness = new_fitness

        if success_F:
            delta = np.array(success_delta)
            weights = delta / (delta.sum() + 1e-30)

            old_f = self.M_F[self.k]
            old_cr = self.M_CR[self.k]

            sf = np.array(success_F)
            scr = np.array(success_CR)

            num_f = np.sum(weights * sf * sf)
            den_f = np.sum(weights * sf)
            new_f = num_f / den_f if den_f > 1e-30 else old_f

            num_cr = np.sum(weights * scr * scr)
            den_cr = np.sum(weights * scr)
            if den_cr == 0 or old_cr < 0:
                new_cr = -1.0
            else:
                new_cr = num_cr / den_cr if den_cr > 1e-30 else old_cr

            self.M_F[self.k] = (new_f + old_f) / 2.0
            self.M_CR[self.k] = (new_cr + old_cr) / 2.0

            self.k = (self.k + 1) % self.H

        plan_size = int(round(
            self.pop_size_init
            + (self.pop_size_min - self.pop_size_init)
            * (self.fe_count / self.max_fe)
        ))
        plan_size = max(plan_size, self.pop_size_min)

        if plan_size < len(self.pop):
            keep = np.argsort(self.fitness)[:plan_size]
            self.pop = self.pop[keep]
            self.fitness = self.fitness[keep]

            self.arc_size = int(round(len(self.pop) * self.arc_rate))
            if len(self.archive) > self.arc_size:
                self.archive = self.archive[:self.arc_size]

            self.p_best_rate *= (1.0 - 0.5 * self.fe_count / self.max_fe)

        return self.best_fitness

    def run(self):
        history = []
        while self.fe_count < self.max_fe:
            history.append(self.step())
        return history

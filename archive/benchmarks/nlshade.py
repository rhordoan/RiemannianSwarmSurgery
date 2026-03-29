"""
NL-SHADE: Non-Linear population size reduction SHADE.

Reference:
    Stanovov, V., Akhmedova, S., & Sopov, E. (2021).
    NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021.
    IEEE Congress on Evolutionary Computation (CEC), 809-816.

    Stanovov, V., Akhmedova, S., & Sopov, E. (2022).
    NL-SHADE-LBC Algorithm with Linear population size reduction and Bound
    Constraints handling for CEC 2022.

The sole algorithmic difference from L-SHADE is the population size reduction
schedule.

  L-SHADE  (linear):     N(t) = N_max + (N_min - N_max) * t / max_FE
  NL-SHADE (non-linear): N(t) = round(N_max + (N_min - N_max) * (t / max_FE)^(1/p))

With p = 4, the decay is concave: the population stays large for the first ~50%
of the budget (maintaining diversity) and then collapses rapidly in the final
quarter (aggressive exploitation). This schedule consistently outperforms linear
PSR on the CEC 2022 benchmark suite.

The rest of the algorithm (current-to-pbest/1 mutation, CR/F success-history
adaptation, external archive, bounce-back boundary handling) is identical to
the L-SHADE in src/lshade.py.
"""

import numpy as np


def _lhs_init(dim, n, lb, ub, seed=None):
    """Latin Hypercube Sampling initialization for better space coverage."""
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=dim, seed=seed)
        unit = sampler.random(n=n)
        return lb + (ub - lb) * unit
    except Exception:
        return np.random.uniform(lb, ub, (n, dim))


class NLSHADE:
    """
    NL-SHADE optimizer: SHADE with non-linear population size reduction.

    API is intentionally identical to src.lshade.LSHADE so TMIOptimizer
    can swap them without any other changes.

    Args:
        problem:       Object with .evaluate(x) -> float and .bounds = [lb, ub].
        dim:           Dimensionality.
        pop_size:      Initial population size N_max (default: 18 * dim).
        max_fe:        Maximum function evaluations budget.
        pop_size_min:  Minimum population size N_min (default: 4).
        H:             Success-history size for F and CR adaptation.
        psr_exponent:  Non-linear PSR exponent p (default: 4).
    """

    def __init__(self,
                 problem,
                 dim: int,
                 pop_size: int = None,
                 max_fe: int = 200_000,
                 pop_size_min: int = 4,
                 H: int = 6,
                 psr_exponent: float = 4.0):

        self.problem = problem
        self.dim = dim
        self.pop_size_init = pop_size if pop_size is not None else 18 * dim
        self.pop_size = self.pop_size_init
        self.pop_size_min = max(pop_size_min, 4)
        self.max_fe = max_fe
        self.H = H
        self.psr_exponent = psr_exponent

        lb, ub = float(problem.bounds[0]), float(problem.bounds[1])
        self.lb = np.full(dim, lb)
        self.ub = np.full(dim, ub)
        self._orig_lb = self.lb.copy()
        self._orig_ub = self.ub.copy()
        self._orig_pop_size_init = self.pop_size_init

        # --- Initialise population (LHS for better coverage) ---
        self.pop = _lhs_init(dim, self.pop_size, self.lb, self.ub)
        self.fitness = np.array([problem.evaluate(x) for x in self.pop])
        self.fe_count = self.pop_size

        # --- Best tracking ---
        best_idx = np.argmin(self.fitness)
        self.best_fitness = float(self.fitness[best_idx])
        self.best_solution = self.pop[best_idx].copy()

        # --- Success history (SHADE) ---
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0  # circular pointer

        # --- External archive ---
        self.archive: list = []
        self.archive_max_size = self.pop_size_init

        self.generation = 0
        self._segment_start_fe = 0
        self._segment_max_fe = self.max_fe

    # ------------------------------------------------------------------
    # Parameter adaptation helpers
    # ------------------------------------------------------------------

    def _generate_F(self, size: int) -> np.ndarray:
        """Cauchy-distributed F values, clipped to (0, 1]."""
        F = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + self.M_F[r]
                if Fi > 0:
                    break
            F[i] = min(Fi, 1.0)
        return F

    def _generate_CR(self, size: int) -> np.ndarray:
        """Normal-distributed CR values, clipped to [0, 1]."""
        CR = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            CR[i] = np.clip(np.random.normal(self.M_CR[r], 0.1), 0.0, 1.0)
        return CR

    @staticmethod
    def _lehmer_mean(values: list, weights: np.ndarray) -> float:
        """Weighted Lehmer mean for F adaptation."""
        if not values:
            return 0.5
        v = np.array(values)
        num = np.sum(weights * v ** 2)
        den = np.sum(weights * v)
        return float(num / den) if den > 1e-30 else 0.5

    @staticmethod
    def _weighted_mean(values: list, weights: np.ndarray) -> float:
        """Weighted arithmetic mean for CR adaptation."""
        if not values:
            return 0.5
        v = np.array(values)
        w = weights / (weights.sum() + 1e-30)
        return float(np.dot(w, v))

    # ------------------------------------------------------------------
    # Non-linear PSR
    # ------------------------------------------------------------------

    def _nlpsr_target_size(self) -> int:
        """
        Non-linear population size using segment-local progress.

        After a reinitialize(), PSR resets so each restart segment gets
        a full exploration-to-exploitation cycle.

        N(t) = round(N_max + (N_min - N_max) * (t_seg / budget_seg)^(1/p))
        """
        seg_elapsed = self.fe_count - self._segment_start_fe
        seg_budget = self._segment_max_fe - self._segment_start_fe
        progress = min(seg_elapsed / max(seg_budget, 1), 1.0)
        exponent = 1.0 / self.psr_exponent
        new_size = round(
            self.pop_size_init
            + (self.pop_size_min - self.pop_size_init) * (progress ** exponent)
        )
        return max(int(new_size), self.pop_size_min)

    # ------------------------------------------------------------------
    # Core generation step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Execute one generation of NL-SHADE.

        Returns:
            Current best fitness.
        """
        self.generation += 1
        N = len(self.pop)

        if N < 4:
            return self.best_fitness

        F_vals = self._generate_F(N)
        CR_vals = self._generate_CR(N)

        # p_best fraction: at least 2 individuals, at most 20 % of pop
        p = max(2.0 / N, 0.2)
        p_count = max(2, int(round(p * N)))
        sorted_indices = np.argsort(self.fitness)

        # Combined population + archive for r2 selection
        if len(self.archive) > 0:
            combined = np.vstack([self.pop, np.array(self.archive)])
        else:
            combined = self.pop.copy()

        new_pop = np.empty_like(self.pop)
        new_fitness = np.empty(N)
        success_F: list = []
        success_CR: list = []
        success_delta: list = []

        for i in range(N):
            Fi = F_vals[i]
            CRi = CR_vals[i]

            # current-to-pbest/1 mutation
            pbest_idx = sorted_indices[np.random.randint(0, p_count)]
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, N)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len(combined))

            mutant = (self.pop[i]
                      + Fi * (self.pop[pbest_idx] - self.pop[i])
                      + Fi * (self.pop[r1] - combined[r2]))

            # Bounce-back boundary handling (per-dimension)
            below = mutant < self.lb
            above = mutant > self.ub
            mutant[below] = (self.lb[below] + self.pop[i][below]) * 0.5
            mutant[above] = (self.ub[above] + self.pop[i][above]) * 0.5

            # Binomial crossover
            j_rand = np.random.randint(0, self.dim)
            trial = self.pop[i].copy()
            mask = np.random.rand(self.dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            # Evaluate
            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1

            # Greedy selection
            if f_trial <= self.fitness[i]:
                if f_trial < self.fitness[i]:
                    self.archive.append(self.pop[i].copy())
                    success_F.append(Fi)
                    success_CR.append(CRi)
                    success_delta.append(abs(self.fitness[i] - f_trial))

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

        # Success-history update
        if success_F:
            delta = np.array(success_delta)
            weights = delta / (delta.sum() + 1e-30)
            self.M_F[self.k] = self._lehmer_mean(success_F, weights)
            self.M_CR[self.k] = self._weighted_mean(success_CR, weights)
            self.k = (self.k + 1) % self.H

        # Archive trimming
        while len(self.archive) > self.archive_max_size:
            self.archive.pop(np.random.randint(0, len(self.archive)))

        # Non-linear PSR
        new_size = self._nlpsr_target_size()
        if new_size < len(self.pop):
            keep = np.argsort(self.fitness)[:new_size]
            self.pop = self.pop[keep]
            self.fitness = self.fitness[keep]

        return self.best_fitness

    # ------------------------------------------------------------------
    # Restart support (used by CARS)
    # ------------------------------------------------------------------

    def reinitialize(self, keep_memory=True, new_pop_size=None,
                     lb_override=None, ub_override=None,
                     seed_pop=None):
        """
        Reinitialize the population for a new search segment.

        Per-dimension bounds can be overridden to restrict the search
        to a sub-region (used by CARS dimensional exclusion).  Bounds
        persist for the entire segment so that NL-SHADE's boundary
        handling keeps the population in the target region.

        Args:
            keep_memory:  if True, preserve SHADE memory across restart.
            new_pop_size: override population size (for IPOP scaling).
            lb_override:  ndarray (D,), per-dim lower bounds for this segment.
            ub_override:  ndarray (D,), per-dim upper bounds for this segment.
            seed_pop:     ndarray (ps, D), pre-generated population to use
                          instead of LHS sampling.
        """
        ps = new_pop_size or self._orig_pop_size_init

        if lb_override is not None:
            self.lb = np.asarray(lb_override, dtype=float)
        else:
            self.lb = self._orig_lb.copy()

        if ub_override is not None:
            self.ub = np.asarray(ub_override, dtype=float)
        else:
            self.ub = self._orig_ub.copy()

        if seed_pop is not None:
            new_pop = np.asarray(seed_pop, dtype=float)
            ps = len(new_pop)
        else:
            new_pop = _lhs_init(self.dim, ps, self.lb, self.ub)

        self.pop = new_pop
        self.fitness = np.array(
            [self.problem.evaluate(x) for x in self.pop]
        )
        self.fe_count += len(self.pop)
        self.pop_size = ps
        self.pop_size_init = ps

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.pop[best_idx].copy()

        self.archive.clear()
        self.archive_max_size = ps
        self.generation = 0
        self._segment_start_fe = self.fe_count
        self._segment_max_fe = self.max_fe

        if not keep_memory:
            self.M_F = np.full(self.H, 0.5)
            self.M_CR = np.full(self.H, 0.5)
            self.k = 0

    # ------------------------------------------------------------------
    # Full-run interface
    # ------------------------------------------------------------------

    def run(self) -> list:
        """Run until budget exhausted. Returns per-generation best-fitness history."""
        history = []
        while self.fe_count < self.max_fe:
            history.append(self.step())
        return history

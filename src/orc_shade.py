"""
ORC-SHADE: Curvature-Modulated Differential Evolution.

A native algorithm (not a wrapper). Ollivier-Ricci Curvature (ORC) is
embedded directly into the Differential Evolution mutation operator,
continuously steering each agent based on its local topological position
on the fitness landscape manifold.

Core Mechanism
--------------
At each generation, ORC is computed on a k-NN graph built over the active
population PLUS a ghost reservoir of the best historically visited solutions
(keeping the graph dense despite LPSR).

For every agent i, the algorithm finds its most negatively curved incident
edge and uses the curvature kappa_i to gate the mutation strategy:

  kappa_i >= tau  (positive/neutral -- agent inside a basin)
      EXPLOIT: current-to-pbest/1
      v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)

  kappa_i < tau   (negative -- agent on a topological saddle boundary)
      EXPLORE: current-to-explore/1 with boosted step
      v_i = x_i + F_exp*(x_explore - x_i) + F_i*(x_r1 - x_r2)
      F_exp = min(1, F_i * (1 + lambda * |kappa_i|))

  x_explore = centroid of the unexplored neighboring community across saddle.

Self-Regulation
---------------
  Unimodal landscapes: all agents quickly enter positive curvature ->
  ORC-SHADE degrades gracefully to pure NL-SHADE. No gate needed.

  Multimodal landscapes: boundary agents get a continuous deterministic
  nudge toward unexplored basins every update_period generations.
  No stagnation counter, no population replacement, no heuristic trigger.

References
----------
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
    Journal of Functional Analysis, 256(3), 810-864.
Tanabe & Fukunaga (2014). L-SHADE. IEEE CEC 2014.
Stanovov et al. (2022). NL-SHADE-RSP. IEEE CEC 2022.
"""

import numpy as np
from scipy.spatial import KDTree

from src.ollivier_ricci import compute_orc_edge


# ---------------------------------------------------------------------------
# Embedded curvature field
# ---------------------------------------------------------------------------

class _CurvatureField:
    """
    Lightweight continuous curvature monitor used natively inside ORC-SHADE.

    Maintains a ghost reservoir of the best historically seen solutions and
    computes per-agent ORC at configurable intervals. The reservoir keeps the
    k-NN graph dense even after LPSR shrinks the active population to 4 agents.
    """

    def __init__(self, dim, k=5, orc_threshold=-0.30,
                 update_period=3, ghost_size=None,
                 adaptive_threshold=False, max_fe=1):
        self.dim = dim
        self.k = k
        self.orc_threshold = orc_threshold
        self.update_period = update_period
        self.ghost_size = ghost_size if ghost_size is not None else 18 * dim
        self.adaptive_threshold = adaptive_threshold
        self.max_fe = max_fe
        self._fe_count_ref = None
        self._ghost_pos = []
        self._ghost_fit = []
        self._kappa = np.array([])
        self._x_explore = np.zeros((0, dim))
        self._last_update = -999
        self.n_explore_agents = 0
        self.mean_kappa = 0.0

    # ------------------------------------------------------------------
    # Ghost reservoir management
    # ------------------------------------------------------------------

    def update_ghosts(self, pop, fitness):
        """Maintain a best-unique historical solution pool."""
        if len(self._ghost_pos) < self.ghost_size:
            for i in range(len(pop)):
                if len(self._ghost_pos) >= self.ghost_size:
                    break
                self._ghost_pos.append(pop[i].copy())
                self._ghost_fit.append(float(fitness[i]))
            return
        ghost_arr = np.array(self._ghost_pos)
        ghost_fit = np.array(self._ghost_fit)
        worst_fit = ghost_fit.max()
        candidates = np.where(fitness < worst_fit)[0]
        if not len(candidates):
            return
        for i in candidates[np.argsort(fitness[candidates])[:5]]:
            worst_idx = int(np.argmax(ghost_fit))
            if fitness[i] >= ghost_fit[worst_idx]:
                continue
            dists = np.linalg.norm(ghost_arr - pop[i], axis=1)
            if dists.min() > 1e-8:
                ghost_arr[worst_idx] = pop[i]
                ghost_fit[worst_idx] = float(fitness[i])
                self._ghost_pos[worst_idx] = pop[i].copy()
                self._ghost_fit[worst_idx] = float(fitness[i])

    def _augmented(self, pop, fitness):
        n_active = len(pop)
        if not self._ghost_pos:
            return pop, fitness, n_active
        aug_pop = np.vstack([pop, np.array(self._ghost_pos)])
        aug_fit = np.concatenate([fitness, np.array(self._ghost_fit)])
        return aug_pop, aug_fit, n_active

    # ------------------------------------------------------------------
    # Per-agent ORC computation
    # ------------------------------------------------------------------

    def compute(self, pop, fitness, generation):
        """
        Compute per-agent curvature and explore targets.

        Returns
        -------
        kappa     : ndarray (N,)   minimum incident ORC per active agent
        x_explore : ndarray (N, d) explore target per active agent
        """
        N_active = len(pop)
        self.update_ghosts(pop, fitness)

        if (generation - self._last_update) < self.update_period:
            k = self._kappa
            xe = self._x_explore
            if len(k) != N_active:
                k = np.zeros(N_active)
                xe = np.tile(pop.mean(axis=0), (N_active, 1))
            return k, xe

        self._last_update = generation
        aug_pop, aug_fit, n_active = self._augmented(pop, fitness)
        N_aug = len(aug_pop)
        k_actual = min(self.k, N_aug - 1)

        if k_actual < 2:
            zk = np.zeros(N_active)
            zx = np.tile(pop.mean(axis=0), (N_active, 1))
            self._kappa, self._x_explore = zk, zx
            return zk, zx

        tree = KDTree(aug_pop)
        _, indices = tree.query(aug_pop, k=k_actual + 1)

        nbrs_list = [set() for _ in range(N_aug)]
        edge_set = set()
        for u in range(N_aug):
            for j in range(1, k_actual + 1):
                v = int(indices[u, j])
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))
                    nbrs_list[u].add(v)
                    nbrs_list[v].add(u)

        nbrs_list = [list(s) for s in nbrs_list]
        edges = list(edge_set)

        orc_edge = np.zeros(len(edges))
        for ei, (u, v) in enumerate(edges):
            nu = [w for w in nbrs_list[u] if w != v]
            nv = [w for w in nbrs_list[v] if w != u]
            if not nu or not nv:
                continue
            orc_edge[ei] = compute_orc_edge(
                aug_pop[u], aug_pop[v],
                aug_pop[np.array(nu, dtype=int)],
                aug_pop[np.array(nv, dtype=int)],
            )

        kappa = np.zeros(N_active)
        x_explore = np.array([pop[i].copy() for i in range(N_active)])

        for ei, (u, v) in enumerate(edges):
            for agent_idx, other_idx in [(u, v), (v, u)]:
                if agent_idx >= N_active:
                    continue
                if orc_edge[ei] < kappa[agent_idx]:
                    kappa[agent_idx] = orc_edge[ei]
                    other_nbrs = nbrs_list[other_idx]
                    # Fitness-gated: only explore toward a community that is
                    # genuinely better than the current agent.  This prevents
                    # spurious saddle crossings on narrow-valley / ridge
                    # functions where off-ridge communities are *worse*.
                    if other_nbrs:
                        nbr_arr = np.array(other_nbrs, dtype=int)
                        nbr_fits = aug_fit[nbr_arr]
                        # Only cross the saddle if the other community is better
                        if nbr_fits.min() < aug_fit[agent_idx]:  # any better neighbour is enough
                            # Softmin-weighted centroid: weight by exp(-f/sigma)
                            # so the best agents in the other basin pull hardest
                            shifted = nbr_fits - nbr_fits.min()
                            sigma = max(shifted.std(), 1e-10)
                            w = np.exp(-shifted / sigma)
                            w /= w.sum()
                            x_explore[agent_idx] = (aug_pop[nbr_arr] * w[:, None]).sum(axis=0)
                    else:
                        if aug_fit[other_idx] < aug_fit[agent_idx]:
                            x_explore[agent_idx] = aug_pop[other_idx].copy()

        self._kappa = kappa
        self._x_explore = x_explore
        # Effective threshold (static or annealed)
        eff_thresh = self.orc_threshold
        if self.adaptive_threshold and self._fe_count_ref is not None:
            progress = min(1.0, self._fe_count_ref[0] / self.max_fe)
            eff_thresh = -0.10 - 0.50 * progress
            self.orc_threshold = eff_thresh
        self.n_explore_agents = int((kappa < eff_thresh).sum())
        self.mean_kappa = float(kappa.mean()) if N_active else 0.0
        return kappa, x_explore


# ---------------------------------------------------------------------------
# ORC-SHADE
# ---------------------------------------------------------------------------

class ORCSHADE:
    """
    ORC-SHADE: Curvature-Modulated Differential Evolution.

    All L-SHADE mechanics are preserved (success-history parameter adaptation,
    external archive, LPSR / NL-SHADE nonlinear schedule). The only structural
    change is in the mutation operator: agents on topological saddles use
    current-to-explore/1 instead of current-to-pbest/1.

    Parameters
    ----------
    problem           : object with .evaluate(x)->float and .bounds=[lb,ub]
    dim               : problem dimensionality
    pop_size          : initial population (default 18*dim)
    max_fe            : function evaluation budget
    pop_size_min      : minimum population after LPSR (default 4)
    H                 : success-history length (default 6)
    orc_k             : k-NN degree for ORC computation (default 5)
    orc_threshold     : kappa below which agent explores (default -0.30)
    orc_update_period : recompute curvature every N generations (default 3)
    orc_lambda        : F-boost multiplier for saddle agents (default 0.5)
    ghost_size        : historical reservoir capacity (default 18*dim)
    pop_schedule      : 'nonlinear' (NL-SHADE, default) or 'linear' (L-SHADE)
    max_explore_frac  : max fraction of agents that can use explore mutation (default 0.25)
    adaptive_threshold: if True, anneal threshold from -0.10 to -0.60 over the run (default False)
    """

    def __init__(self, problem, dim, pop_size=None, max_fe=200_000,
                 pop_size_min=4, H=6, orc_k=5, orc_threshold=-0.30,
                 orc_update_period=3, orc_lambda=0.5, ghost_size=None,
                 pop_schedule='nonlinear',
                 max_explore_frac=0.25,
                 adaptive_threshold=False):
        self.problem = problem
        self.dim = dim
        self.pop_size_init = pop_size if pop_size is not None else 18 * dim
        self.pop_size_min = max(pop_size_min, 4)
        self.max_fe = max_fe
        self.H = H
        self.orc_lambda = orc_lambda
        self.pop_schedule = pop_schedule
        self.max_explore_frac = max_explore_frac
        self.adaptive_threshold = adaptive_threshold

        lb, ub = float(problem.bounds[0]), float(problem.bounds[1])
        self.lb = lb
        self.ub = ub

        try:
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=dim, seed=None)
            lhs = sampler.random(n=self.pop_size_init)
            self.pop = lb + (ub - lb) * lhs
        except Exception:
            self.pop = np.random.uniform(lb, ub, (self.pop_size_init, dim))
        self.fitness = np.array([problem.evaluate(x) for x in self.pop])
        self.fe_count = self.pop_size_init

        best_idx = np.argmin(self.fitness)
        self.best_fitness = float(self.fitness[best_idx])
        self.best_solution = self.pop[best_idx].copy()

        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self._hist_ptr = 0
        self.archive = []
        self.archive_max_size = self.pop_size_init

        self.generation = 0
        self.total_explore_mutations = 0
        self.total_exploit_mutations = 0
        # Convergence log: [(fe_count, best_fitness), ...] at each step
        self.convergence_log: list = [(self.fe_count, self.best_fitness)]

        self._curv = _CurvatureField(
            dim=dim, k=orc_k, orc_threshold=orc_threshold,
            update_period=orc_update_period, ghost_size=ghost_size,
            adaptive_threshold=adaptive_threshold, max_fe=max_fe,
        )
        self._curv._fe_count_ref = [self.fe_count]

    # ------------------------------------------------------------------
    # L-SHADE parameter generation
    # ------------------------------------------------------------------

    def _gen_F(self, size):
        F = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + self.M_F[r]
                if Fi > 0:
                    break
            F[i] = min(Fi, 1.0)
        return F

    def _gen_CR(self, size):
        CR = np.empty(size)
        for i in range(size):
            r = np.random.randint(0, self.H)
            CR[i] = np.clip(np.random.normal(self.M_CR[r], 0.1), 0.0, 1.0)
        return CR

    def _lehmer(self, vals, w):
        v, w = np.array(vals), np.array(w)
        d = np.dot(w, v)
        return float(np.dot(w, v**2) / d) if d > 1e-30 else 0.5

    def _wmean(self, vals, w):
        v, w = np.array(vals), np.array(w)
        w = w / (w.sum() + 1e-30)
        return float(np.dot(w, v))

    def _bounce(self, mutant, parent):
        """Bounce-back boundary repair."""
        lo = mutant < self.lb
        hi = mutant > self.ub
        mutant[lo] = (self.lb + parent[lo]) / 2.0
        mutant[hi] = (self.ub + parent[hi]) / 2.0
        return mutant

    # ------------------------------------------------------------------
    # Core generation step
    # ------------------------------------------------------------------

    def step(self):
        self.generation += 1
        N = len(self.pop)
        if N < 4:
            return self.best_fitness

        self._curv._fe_count_ref[0] = self.fe_count
        kappa, x_explore = self._curv.compute(self.pop, self.fitness, self.generation)
        F_vals = self._gen_F(N)
        CR_vals = self._gen_CR(N)

        p = max(2.0 / N, 0.2)
        p_count = max(2, int(round(p * N)))
        sorted_idx = np.argsort(self.fitness)
        combined = (np.vstack([self.pop, np.array(self.archive)])
                    if self.archive else self.pop.copy())

        new_pop = np.empty_like(self.pop)
        new_fit = np.empty(N)
        suc_F, suc_CR, suc_delta = [], [], []
        threshold = self._curv.orc_threshold

        # Build explore mask: only the most negatively curved agents, capped
        # at max_explore_frac * N to preserve enough exploitation bandwidth.
        explore_mask = np.zeros(N, dtype=bool)
        saddle_idx = np.where(kappa < threshold)[0]
        if len(saddle_idx) > 0:
            max_explorers = max(1, int(round(self.max_explore_frac * N)))
            if len(saddle_idx) <= max_explorers:
                explore_mask[saddle_idx] = True
            else:
                # Pick the most negatively curved (genuine saddles first)
                most_neg = saddle_idx[np.argsort(kappa[saddle_idx])[:max_explorers]]
                explore_mask[most_neg] = True

        for i in range(N):
            Fi = F_vals[i]
            CRi = CR_vals[i]

            if explore_mask[i] and not np.allclose(x_explore[i], self.pop[i]):
                # EXPLORE: current-to-explore/1 with curvature-boosted F
                F_exp = min(1.0, Fi * (1.0 + self.orc_lambda * abs(float(kappa[i]))))
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(0, N)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, len(combined))
                mutant = (self.pop[i]
                          + F_exp * (x_explore[i] - self.pop[i])
                          + Fi * (self.pop[r1] - combined[r2]))
                self.total_explore_mutations += 1
            else:
                # EXPLOIT: current-to-pbest/1
                pbest_idx = sorted_idx[np.random.randint(0, p_count)]
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(0, N)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, len(combined))
                mutant = (self.pop[i]
                          + Fi * (self.pop[pbest_idx] - self.pop[i])
                          + Fi * (self.pop[r1] - combined[r2]))
                self.total_exploit_mutations += 1

            mutant = self._bounce(mutant.copy(), self.pop[i])
            j_rand = np.random.randint(0, self.dim)
            trial = self.pop[i].copy()
            for d in range(self.dim):
                if np.random.rand() < CRi or d == j_rand:
                    trial[d] = mutant[d]

            f_trial = float(self.problem.evaluate(trial))
            self.fe_count += 1

            if f_trial <= self.fitness[i]:
                if f_trial < self.fitness[i]:
                    self.archive.append(self.pop[i].copy())
                    suc_F.append(Fi)
                    suc_CR.append(CRi)
                    suc_delta.append(self.fitness[i] - f_trial)
                new_pop[i] = trial
                new_fit[i] = f_trial
                if f_trial < self.best_fitness:
                    self.best_fitness = f_trial
                    self.best_solution = trial.copy()
            else:
                new_pop[i] = self.pop[i]
                new_fit[i] = self.fitness[i]

            if self.fe_count >= self.max_fe:
                for j in range(i + 1, N):
                    new_pop[j] = self.pop[j]
                    new_fit[j] = self.fitness[j]
                break

        self.pop = new_pop
        self.fitness = new_fit

        if suc_F:
            w = np.array(suc_delta)
            w = w / (w.sum() + 1e-30)
            self.M_F[self._hist_ptr] = self._lehmer(suc_F, w)
            self.M_CR[self._hist_ptr] = self._wmean(suc_CR, w)
            self._hist_ptr = (self._hist_ptr + 1) % self.H

        while len(self.archive) > self.archive_max_size:
            self.archive.pop(np.random.randint(0, len(self.archive)))

        progress = self.fe_count / self.max_fe
        if self.pop_schedule == 'nonlinear':
            new_size = int(round(
                self.pop_size_min
                + (self.pop_size_init - self.pop_size_min) * (1.0 - progress ** 4)
            ))
        else:
            new_size = int(round(
                self.pop_size_init
                + (self.pop_size_min - self.pop_size_init) * progress
            ))
        new_size = max(new_size, self.pop_size_min)
        if new_size < len(self.pop):
            keep = np.argsort(self.fitness)[:new_size]
            self.pop = self.pop[keep]
            self.fitness = self.fitness[keep]

        self.convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_fitness

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def run(self):
        """Run until budget exhausted. Returns (best_solution, best_fitness)."""
        while self.fe_count < self.max_fe:
            self.step()
        return self.best_solution, self.best_fitness

    def get_run_stats(self):
        total = self.total_explore_mutations + self.total_exploit_mutations
        return {
            'best_fitness': self.best_fitness,
            'fe_count': self.fe_count,
            'generations': self.generation,
            'explore_mutations': self.total_explore_mutations,
            'exploit_mutations': self.total_exploit_mutations,
            'explore_pct': 100.0 * self.total_explore_mutations / max(total, 1),
            'mean_kappa': self._curv.mean_kappa,
            'n_explore_agents': self._curv.n_explore_agents,
            'effective_threshold': float(self._curv.orc_threshold),
        }

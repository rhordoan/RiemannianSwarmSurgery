"""
ORC-SHADE v2: Curvature-Modulated Differential Evolution.

A native algorithm (not a wrapper). Ollivier-Ricci Curvature (ORC) is
embedded directly into the Differential Evolution mutation operator,
continuously steering each agent based on its local topological position
on the fitness landscape manifold.

Architecture v2
---------------
Improvements over v1:

1. PCA-Projected Curvature:
   The k-NN graph and ORC values are computed in the top d_eff = min(D, 10)
   principal components of the swarm+ghost ensemble, eliminating distance
   concentration that makes curvature meaningless in high D. Explore targets
   x_explore remain in the original D-dimensional space.

2. Adaptive k-NN:
   k = min(2*d_eff + 1, N_aug // 4, 10), bounded for computational cost.

3. Continuous Alpha Blending:
   Replaces v1 binary explore/exploit switch.
   alpha_i = clip(|kappa_i| / kappa_scale, 0, 1) if kappa_i < 0 else 0
   target_i = alpha_i * x_explore[i] + (1 - alpha_i) * x_pbest
   v_i = x_i + F_i*(target_i - x_i) + F_i*(x_r1 - x_r2)

4. Fitness-Gated Softmin-Centroid Explore Target:
   For each agent on a saddle (kappa < 0), the explore target is the
   softmin-weighted centroid of the community on the other side of the
   most-negative-curvature edge. Fitness gate: only set explore target
   when the other community contains at least one agent with better
   fitness. This prevents off-ridge exploration on ridge functions and
   spurious exploration on unimodal functions.

Parameters removed vs v1: orc_threshold, max_explore_frac, orc_lambda,
  adaptive_threshold, orc_k (derived from d_eff).
Parameter added vs v1: kappa_scale (default 1.0, ORC is in [-1,1]).

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
    Curvature oracle for ORC-SHADE v2.

    Computes per-agent Ollivier-Ricci curvature on the PCA-projected
    intrinsic manifold of the swarm+ghost ensemble, then sets a
    fitness-gated softmin-centroid explore target for agents on saddles.
    """

    def __init__(self, dim, update_period=5, ghost_size=None, k_max=10):
        self.dim = dim
        self.update_period = update_period
        self.ghost_size = ghost_size if ghost_size is not None else 18 * dim
        self.k_max = k_max
        self._ghost_pos = []
        self._ghost_fit = []
        self._kappa = np.array([])
        self._x_explore = np.zeros((0, dim))
        self._explore_valid = np.zeros(0, dtype=bool)
        self._last_update = -999
        self.mean_kappa = 0.0
        self.n_clusters = 1

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
        ghost_arr = np.array(self._ghost_pos)
        ghost_fit = np.array(self._ghost_fit)
        # Use only the best orc_aug_size ghosts for ORC augmentation.
        # Full ghost buffer includes early-run random scatter which creates
        # artificial bimodal structure in the augmented population, causing
        # spurious strongly-negative ORC on nearly every agent.
        orc_aug_size = min(len(ghost_arr), 4 * self.dim)
        if len(ghost_arr) > orc_aug_size:
            top_idx = np.argpartition(ghost_fit, orc_aug_size)[:orc_aug_size]
            ghost_arr = ghost_arr[top_idx]
            ghost_fit = ghost_fit[top_idx]
        aug_pop = np.vstack([pop, ghost_arr])
        aug_fit = np.concatenate([fitness, ghost_fit])
        return aug_pop, aug_fit, n_active

    # ------------------------------------------------------------------
    # Per-agent ORC computation
    # ------------------------------------------------------------------

    def compute(self, pop, fitness, generation):
        """
        Compute per-agent curvature and fitness-gated explore targets.

        Returns
        -------
        kappa     : ndarray (N,)   minimum incident ORC per active agent
        x_explore : ndarray (N, d) softmin-centroid explore target
                                   (only set where fitness gate passes)
        """
        N_active = len(pop)
        self.update_ghosts(pop, fitness)

        if (generation - self._last_update) < self.update_period:
            k = self._kappa
            xe = self._x_explore
            if len(k) != N_active:
                k = np.zeros(N_active)
                xe = np.tile(pop.mean(axis=0), (N_active, 1))
                self._explore_valid = np.zeros(N_active, dtype=bool)
            return k, xe

        self._last_update = generation
        aug_pop, aug_fit, n_active = self._augmented(pop, fitness)
        N_aug = len(aug_pop)

        # --- Fitness-lifted positions (replaces PCA projection) ---
        # Append log-fitness as an extra coordinate scaled by sqrt(D).
        # Euclidean distance in lifted space reflects both spatial proximity
        # AND fitness similarity: agents in the same basin stay close, agents
        # separated by a fitness ridge become far apart. This is the pullback
        # metric on the fitness graph surface (x, f(x)).
        # Log-fitness compresses the dynamic range (CEC functions span 1e-14
        # to 1e+10) while preserving relative basin structure.
        spatial_std = max(aug_pop.std(), 1e-10)
        log_fit = np.log1p(np.maximum(aug_fit, 0.0))
        log_fit_std = max(log_fit.std(), 1e-10)
        gamma = np.sqrt(self.dim)
        fit_col = (gamma * log_fit / log_fit_std)[:, np.newaxis]
        lifted = np.hstack([aug_pop / spatial_std, fit_col])

        # --- Adaptive k ---
        k_target = min(2 * min(self.dim, 15) + 1, N_aug // 4, self.k_max)
        k_actual = min(k_target, N_aug - 1)

        if k_actual < 2:
            zk = np.zeros(N_active)
            zx = np.tile(pop.mean(axis=0), (N_active, 1))
            self._kappa, self._x_explore = zk, zx
            self._explore_valid = np.zeros(N_active, dtype=bool)
            return zk, zx

        # --- k-NN graph on fitness-lifted positions ---
        tree = KDTree(lifted)
        _, indices = tree.query(lifted, k=k_actual + 1)

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

        # --- ORC on fitness-lifted positions ---
        # Truncate nu/nv to k_actual-1 to guarantee equal support-set sizes.
        # Every node has >=k_actual undirected neighbors; after excluding one
        # endpoint, >=k_actual-1 remain. Fixed-size sets make cost matrices
        # square, eliminating _pad_to_equal_size calls (was 86% of edge calls
        # and the dominant runtime bottleneck at 0.12s / 0.22s total).
        nbrs_limit = max(1, k_actual - 1)
        orc_edge = np.zeros(len(edges))
        for ei, (u, v) in enumerate(edges):
            nu = [w for w in nbrs_list[u] if w != v][:nbrs_limit]
            nv = [w for w in nbrs_list[v] if w != u][:nbrs_limit]
            if not nu or not nv:
                continue
            orc_edge[ei] = compute_orc_edge(
                lifted[u], lifted[v],
                lifted[np.array(nu, dtype=int)],
                lifted[np.array(nv, dtype=int)],
            )

        # --- Per-agent kappa: minimum incident ORC ---
        # The explore target is coupled to the most-negative edge: only
        # the strongest saddle signal determines where to explore.  This
        # prevents weak/noisy negative edges from triggering exploration
        # toward marginally-better but spatially-distant solutions.
        kappa = np.zeros(N_active)
        x_explore = pop.copy()
        explore_valid = np.zeros(N_active, dtype=bool)

        for ei, (u, v) in enumerate(edges):
            for agent_idx, other_idx in [(u, v), (v, u)]:
                if agent_idx >= N_active:
                    continue
                if orc_edge[ei] < kappa[agent_idx]:
                    kappa[agent_idx] = orc_edge[ei]
                    other_nbrs = nbrs_list[other_idx]
                    if other_nbrs:
                        nbr_arr = np.array(other_nbrs, dtype=int)
                        nbr_fits = aug_fit[nbr_arr]
                        best_nbr = nbr_arr[np.argmin(nbr_fits)]
                        if aug_fit[best_nbr] < aug_fit[agent_idx]:
                            x_explore[agent_idx] = aug_pop[best_nbr].copy()
                            explore_valid[agent_idx] = True
                    else:
                        if aug_fit[other_idx] < aug_fit[agent_idx]:
                            x_explore[agent_idx] = aug_pop[other_idx].copy()
                            explore_valid[agent_idx] = True

        self._kappa = kappa
        self._x_explore = x_explore
        self._explore_valid = explore_valid
        self.mean_kappa = float(kappa.mean()) if N_active else 0.0
        self.n_clusters = int(explore_valid.sum())
        return kappa, x_explore


# ---------------------------------------------------------------------------
# ORC-SHADE v2
# ---------------------------------------------------------------------------

class ORCSHADE:
    """
    ORC-SHADE v2: Curvature-Modulated Differential Evolution.

    All L-SHADE mechanics preserved (success-history adaptation, external
    archive, nonlinear population size reduction). The mutation operator uses
    a continuous curvature-modulated blend: agents on topological saddles
    (kappa < 0) with a fitness-gated explore target get alpha > 0, steering
    them toward the better neighboring community. All others use pure
    current-to-pbest/1, identical to NL-SHADE.

    Parameters
    ----------
    problem          : object with .evaluate(x)->float and .bounds=[lb, ub]
    dim              : problem dimensionality
    pop_size         : initial population (default 18*dim)
    max_fe           : function evaluation budget
    pop_size_min     : minimum population after LPSR (default 4)
    H                : success-history length (default 6)
    orc_update_period: recompute curvature every N generations (default 5)
    ghost_size       : historical reservoir capacity (default 18*dim)
    pop_schedule     : "nonlinear" (NL-SHADE, default) or "linear"
    kappa_scale      : curvature normalization. alpha = clip(|kappa|/kappa_scale, 0, 1).
                       ORC bounded in [-1, 1], so kappa_scale=1.0 spans the full range.
    """

    def __init__(self, problem, dim, pop_size=None, max_fe=200_000,
                 pop_size_min=4, H=6, orc_update_period=5, ghost_size=None,
                 pop_schedule="nonlinear", kappa_scale=1.0,
                 kappa_min=0.0, p_elite=0.05, k_max=10):
        self.problem = problem
        self.dim = dim
        self.pop_size_init = pop_size if pop_size is not None else 18 * dim
        self.pop_size_min = max(pop_size_min, 4)
        self.max_fe = max_fe
        self.H = H
        self.pop_schedule = pop_schedule
        self.kappa_scale = max(kappa_scale, kappa_min + 1e-6)
        self.kappa_min = kappa_min
        self.p_elite = p_elite

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

        # Exploit history prior: M_CR=0.5 (NL-SHADE standard)
        # Explore history prior: M_CR=0.8 (cross-saddle needs high CR)
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.M_F_explore = np.full(H, 0.5)
        self.M_CR_explore = np.full(H, 0.8)
        self._hist_ptr = 0
        self._hist_ptr_explore = 0
        self.archive = []
        self.archive_max_size = self.pop_size_init

        self.generation = 0
        self._total_alpha = 0.0
        self._total_mutations = 0
        self._last_mean_alpha = 0.0
        self.convergence_log: list = [(self.fe_count, self.best_fitness)]

        self._curv = _CurvatureField(
            dim=dim, update_period=orc_update_period, ghost_size=ghost_size,
            k_max=k_max,
        )
        self._base_update_period = orc_update_period
        self._idle_streak = 0

    # ------------------------------------------------------------------
    # L-SHADE parameter generation
    # ------------------------------------------------------------------

    def _gen_F_one(self, mem):
        """Draw one F from Cauchy distribution centred on mem[r]."""
        while True:
            r = np.random.randint(0, self.H)
            Fi = np.random.standard_cauchy() * 0.1 + mem[r]
            if Fi > 0:
                break
        return min(Fi, 1.0)

    def _gen_CR_one(self, mem):
        """Draw one CR from Gaussian distribution centred on mem[r]."""
        r = np.random.randint(0, self.H)
        return float(np.clip(np.random.normal(mem[r], 0.1), 0.0, 1.0))

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

        kappa, x_explore = self._curv.compute(self.pop, self.fitness, self.generation)
        ev = self._curv._explore_valid

        sorted_idx = np.argsort(self.fitness)

        # Fix 2 -- elite mask: top p_elite fraction always exploits.
        # These agents are the pbest anchors; exploring from there
        # destabilises their role as gradient attractors for the swarm.
        n_elite = max(1, int(round(self.p_elite * N)))
        elite_set = set(int(x) for x in sorted_idx[:n_elite])

        p = max(2.0 / N, 0.2)
        p_count = max(2, int(round(p * N)))
        combined = (np.vstack([self.pop, np.array(self.archive)])
                    if self.archive else self.pop.copy())

        new_pop = np.empty_like(self.pop)
        new_fit = np.empty(N)
        suc_F_ex,  suc_CR_ex,  suc_d_ex  = [], [], []   # exploitation
        suc_F_xp,  suc_CR_xp,  suc_d_xp  = [], [], []   # exploration
        total_alpha = 0.0

        # Adaptive kappa normalization: use 90th percentile of |kappa|
        # among active agents so the strongest saddle agents get alpha~1.
        neg_kappas = np.abs(kappa[(kappa < -self.kappa_min)
                                  & (np.arange(len(kappa)) < len(ev))
                                  & ev[:len(kappa)]])
        if len(neg_kappas) >= 3:
            kappa_range = max(np.percentile(neg_kappas, 90) - self.kappa_min, 0.05)
        else:
            kappa_range = max(self.kappa_scale - self.kappa_min, 1e-6)

        for i in range(N):
            ki = float(kappa[i])
            explore_active = (i < len(ev) and bool(ev[i]))

            if ki < -self.kappa_min and explore_active and i not in elite_set:
                alpha_i = float(np.clip(
                    (abs(ki) - self.kappa_min) / kappa_range, 0.0, 1.0))
            else:
                alpha_i = 0.0
            total_alpha += alpha_i

            # Draw F/CR from the mode-appropriate SHADE history.
            if alpha_i > 0.0:
                Fi   = self._gen_F_one(self.M_F_explore)
                CR_i = self._gen_CR_one(self.M_CR_explore)
            else:
                Fi   = self._gen_F_one(self.M_F)
                CR_i = self._gen_CR_one(self.M_CR)

            pbest_idx = sorted_idx[np.random.randint(0, p_count)]
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, N)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len(combined))

            # Continuous blend: alpha=0 is pure current-to-pbest/1 (NL-SHADE).
            # alpha>0 smoothly steers toward verified best across saddle.
            target = alpha_i * x_explore[i] + (1.0 - alpha_i) * self.pop[pbest_idx]
            mutant = (self.pop[i]
                      + Fi * (target - self.pop[i])
                      + Fi * (self.pop[r1] - combined[r2]))

            mutant = self._bounce(mutant.copy(), self.pop[i])
            j_rand = np.random.randint(0, self.dim)
            trial = self.pop[i].copy()
            mask = np.random.rand(self.dim) < CR_i
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            f_trial = float(self.problem.evaluate(trial))
            self.fe_count += 1

            if f_trial <= self.fitness[i]:
                if f_trial < self.fitness[i]:
                    self.archive.append(self.pop[i].copy())
                    delta = self.fitness[i] - f_trial
                    if alpha_i == 0.0:
                        suc_F_ex.append(Fi);  suc_CR_ex.append(CR_i);  suc_d_ex.append(delta)
                    else:
                        suc_F_xp.append(Fi);  suc_CR_xp.append(CR_i);  suc_d_xp.append(delta)
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
        self._last_mean_alpha = total_alpha / max(N, 1)
        self._total_alpha += total_alpha
        self._total_mutations += N

        # Adaptive ORC frequency: if exploration has been idle for many
        # consecutive generations, double the update period (up to 40).
        # Resets immediately when exploration fires again.
        if self._last_mean_alpha < 0.001:
            self._idle_streak += 1
            if self._idle_streak >= 10:
                self._curv.update_period = min(
                    self._base_update_period * (2 ** (self._idle_streak // 10)), 40)
        else:
            self._idle_streak = 0
            self._curv.update_period = self._base_update_period

        # Update exploit SHADE history (alpha=0 successes only)
        if suc_F_ex:
            w = np.array(suc_d_ex); w /= (w.sum() + 1e-30)
            self.M_F[self._hist_ptr]  = self._lehmer(suc_F_ex, w)
            self.M_CR[self._hist_ptr] = self._wmean(suc_CR_ex, w)
            self._hist_ptr = (self._hist_ptr + 1) % self.H

        # Update explore SHADE history (alpha>0 successes only)
        if suc_F_xp:
            w = np.array(suc_d_xp); w /= (w.sum() + 1e-30)
            self.M_F_explore[self._hist_ptr_explore]  = self._lehmer(suc_F_xp, w)
            self.M_CR_explore[self._hist_ptr_explore] = self._wmean(suc_CR_xp, w)
            self._hist_ptr_explore = (self._hist_ptr_explore + 1) % self.H

        while len(self.archive) > self.archive_max_size:
            self.archive.pop(np.random.randint(0, len(self.archive)))

        progress = self.fe_count / self.max_fe
        if self.pop_schedule == "nonlinear":
            # Concave schedule matching NL-SHADE exactly:
            # N(t) = N_max + (N_min - N_max) * t^(1/4)
            # At t=0.5 -> ~84% of pop already removed; fast early collapse
            # preserves many exploitation generations at the end.
            new_size = int(round(
                self.pop_size_init
                + (self.pop_size_min - self.pop_size_init) * (progress ** 0.25)
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
        total = max(self._total_mutations, 1)
        mean_alpha = self._total_alpha / total
        return {
            "best_fitness":     self.best_fitness,
            "fe_count":         self.fe_count,
            "generations":      self.generation,
            "explore_pct":      100.0 * mean_alpha,
            "mean_alpha":       mean_alpha,
            "last_mean_alpha":  self._last_mean_alpha,
            "n_explore_agents": int(self._curv._explore_valid.sum()),
            "mean_kappa":       float(self._curv.mean_kappa),
            "kappa_min":        self.kappa_min,
            "p_elite":          self.p_elite,
            "M_CR_exploit":     float(self.M_CR.mean()),
            "M_CR_explore":     float(self.M_CR_explore.mean()),
        }

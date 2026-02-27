"""
CARS: Curvature-Aware Restart Strategy for Differential Evolution.

Uses Ollivier-Ricci Curvature (ORC) both as a topology monitor on the
population's fitness-lifted k-NN graph AND as a directional landscape
scanner via ORC Landscape Probing (OLP).

Key mechanisms:

  1. ORC monitoring: classifies population topological state to
     decide WHEN to restart (stagnation in a single basin).

  2. ORC Landscape Probing: when a restart is triggered, creates
     axis-aligned directional probes at multiple radii around the
     converged best, computes ORC on the probe graph, and identifies
     dimensions where the fitness landscape has basin transitions
     (negative curvature = ridge/saddle).

  3. Curvature-directed dimensional exclusion: restricts the top-K
     most-negative-curvature dimensions to the opposite half-space,
     directing the restart toward detected basin transitions.
     K scales with dimensionality: K = max(3, ceil(D * 0.15)).

  4. Opposition-based restart seeding: 50% of the restart population
     is generated via opposition-based learning, 50% via LHS.

  5. Maximin diversification: restart individuals that land too close
     to archived basin centers are regenerated.

References
----------
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
Stanovov et al. (2022). NL-SHADE-LBC. IEEE CEC 2022.
Brest et al. (2017). Algorithm jSO. IEEE CEC 2017.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import logging
import numpy as np
from scipy.spatial import KDTree

_log = logging.getLogger("CARS")

from src.ollivier_ricci import compute_orc_edge
from benchmarks.nlshade import NLSHADE


# ---------------------------------------------------------------------------
# Curvature Monitor
# ---------------------------------------------------------------------------

class CurvatureMonitor:
    """
    Computes ORC on the population's fitness-lifted k-NN graph and
    classifies the topological state of the swarm.

    Also provides ORC Landscape Probing for directional basin-transition
    detection around a converged point.
    """

    def __init__(self, dim, k=7, domain_width=200.0):
        self.dim = dim
        self.k = k
        self.gamma = np.sqrt(dim)
        self.domain_width = domain_width
        self.convergence_ratio = 0.02

    def classify(self, pop, fitness, stagnating):
        N = len(pop)
        if N < 6:
            return ("exploiting" if not stagnating else "trapped"), 0.0, 0.0

        pop_spread = float(np.mean(np.std(pop, axis=0)))
        converged = pop_spread < self.convergence_ratio * self.domain_width

        if converged and stagnating:
            return "trapped", 0.0, 0.0
        if converged:
            return "exploiting", 0.0, 0.0

        lifted = self._lift(pop, fitness)
        k_actual = min(self.k, N - 1)
        if k_actual < 2:
            return ("exploiting" if not stagnating else "trapped"), 0.0, 0.0

        tree = KDTree(lifted)
        _, indices = tree.query(lifted, k=k_actual + 1)

        nbrs_list = [set() for _ in range(N)]
        edge_set = set()
        for u in range(N):
            for j in range(1, k_actual + 1):
                v = int(indices[u, j])
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))
                    nbrs_list[u].add(v)
                    nbrs_list[v].add(u)
        nbrs_list = [list(s) for s in nbrs_list]
        edges = list(edge_set)

        if not edges:
            return ("exploiting" if not stagnating else "trapped"), 0.0, 0.0

        nbrs_limit = max(1, k_actual - 1)
        orc_vals = np.zeros(len(edges))
        for ei, (u, v) in enumerate(edges):
            nu = [w for w in nbrs_list[u] if w != v][:nbrs_limit]
            nv = [w for w in nbrs_list[v] if w != u][:nbrs_limit]
            if not nu or not nv:
                continue
            orc_vals[ei] = compute_orc_edge(
                lifted[u], lifted[v],
                lifted[np.array(nu, dtype=int)],
                lifted[np.array(nv, dtype=int)],
            )

        neg_fraction = float(np.mean(orc_vals < -0.1))
        mean_kappa = float(np.mean(orc_vals))

        tau = 0.08
        if neg_fraction > tau:
            return "exploring", neg_fraction, mean_kappa
        elif stagnating:
            return "trapped", neg_fraction, mean_kappa
        else:
            return "exploiting", neg_fraction, mean_kappa

    def _lift(self, pop, fitness):
        spatial_std = max(pop.std(), 1e-10)
        log_fit = np.log1p(np.maximum(fitness, 0.0))
        log_fit_std = max(log_fit.std(), 1e-10)
        fit_col = (self.gamma * log_fit / log_fit_std)[:, np.newaxis]
        return np.hstack([pop / spatial_std, fit_col])

    def _lift_probes(self, probes, fitness):
        """Per-dimension normalization for probe populations."""
        per_dim_std = np.maximum(np.std(probes, axis=0), 1e-10)
        spatial = probes / per_dim_std
        log_fit = np.log1p(np.maximum(fitness, 0.0))
        fit_std = max(float(np.std(log_fit)), 1e-10)
        fit_col = (self.gamma * log_fit / fit_std)[:, np.newaxis]
        return np.hstack([spatial, fit_col])

    def probe_curvature_profile(self, x_star, lb, ub, problem,
                                radii_frac=(0.05, 0.15, 0.30)):
        """
        ORC Landscape Probing: directional probes around x_star at
        multiple radii, with ORC computed on the probe graph.

        Returns
        -------
        curvature_profile : ndarray(dim,)
            Average ORC for edges involving each dimension's probes.
        n_evals : int
            Number of function evaluations consumed.
        probe_info : dict
            Raw diagnostics.
        """
        dim = len(x_star)
        domain_half = (ub - lb) / 2.0

        probes = [x_star.copy()]
        probe_dim_map = [-1]

        for frac in radii_frac:
            delta = frac * domain_half
            for d in range(dim):
                p_plus = x_star.copy()
                p_plus[d] = min(p_plus[d] + delta, ub)
                p_minus = x_star.copy()
                p_minus[d] = max(p_minus[d] - delta, lb)
                probes.append(p_plus)
                probes.append(p_minus)
                probe_dim_map.extend([d, d])

        probes = np.array(probes)
        fitness = np.array([problem.evaluate(p) for p in probes])
        n_evals = len(probes)

        center_fitness = float(fitness[0])
        n_radii = len(radii_frac)
        last_radius_offset = 1 + (n_radii - 1) * 2 * dim
        fitness_asymmetry = np.zeros(dim)
        for d in range(dim):
            idx_plus = last_radius_offset + 2 * d
            idx_minus = last_radius_offset + 2 * d + 1
            fitness_asymmetry[d] = fitness[idx_plus] - fitness[idx_minus]

        probe_info = {
            "center_fitness": center_fitness,
            "fitness_asymmetry": fitness_asymmetry,
            "probe_fitness_range": (float(fitness.min()), float(fitness.max())),
        }

        lifted = self._lift_probes(probes, fitness)
        k = min(7, len(probes) - 1)
        if k < 2:
            return np.zeros(dim), n_evals, probe_info

        tree = KDTree(lifted)
        _, indices = tree.query(lifted, k=k + 1)

        N = len(probes)
        nbrs_list = [set() for _ in range(N)]
        edge_set = set()
        for u in range(N):
            for j in range(1, k + 1):
                v = int(indices[u, j])
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))
                    nbrs_list[u].add(v)
                    nbrs_list[v].add(u)
        nbrs_list = [list(s) for s in nbrs_list]
        edges = list(edge_set)

        if not edges:
            return np.zeros(dim), n_evals, probe_info

        nbrs_limit = max(1, k - 1)
        orc_vals = np.zeros(len(edges))
        for ei, (u, v) in enumerate(edges):
            nu = [w for w in nbrs_list[u] if w != v][:nbrs_limit]
            nv = [w for w in nbrs_list[v] if w != u][:nbrs_limit]
            if not nu or not nv:
                continue
            orc_vals[ei] = compute_orc_edge(
                lifted[u], lifted[v],
                lifted[np.array(nu, dtype=int)],
                lifted[np.array(nv, dtype=int)],
            )

        curvature_sum = np.zeros(dim)
        curvature_count = np.zeros(dim)
        for ei, (u, v) in enumerate(edges):
            orc = orc_vals[ei]
            du = probe_dim_map[u]
            dv = probe_dim_map[v]
            if du >= 0:
                curvature_sum[du] += orc
                curvature_count[du] += 1
            if dv >= 0:
                curvature_sum[dv] += orc
                curvature_count[dv] += 1

        curvature_profile = np.where(
            curvature_count > 0,
            curvature_sum / curvature_count,
            0.0,
        )
        return curvature_profile, n_evals, probe_info


# ---------------------------------------------------------------------------
# CARS: Curvature-Aware Restart Strategy
# ---------------------------------------------------------------------------

class CARS:
    """
    Wraps NL-SHADE with ORC-based restart decisions and ORC Landscape
    Probing for curvature-directed dimensional exclusion.

    Parameters
    ----------
    problem        : object with .evaluate(x)->float and .bounds=[lb, ub]
    dim            : problem dimensionality
    max_fe         : total function evaluation budget
    pop_size       : initial population size (default 18*dim)
    orc_period     : check ORC every N generations (default 25)
    stag_gens      : generations without improvement to declare stagnation
    n_exclude_dims : dimensions to restrict per restart (default adaptive)
    """

    def __init__(self, problem, dim, max_fe=300_000,
                 pop_size=None, orc_period=25, stag_gens=None,
                 n_exclude_dims=None, verbose=False):
        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.orc_period = orc_period
        self.verbose = verbose

        lb, ub = problem.bounds
        self.lb = float(lb)
        self.ub = float(ub)
        self.domain_center = (self.lb + self.ub) / 2.0

        ps = pop_size or 18 * dim
        self.initial_pop_size = ps
        self.stag_gens = stag_gens or max(50, 3 * dim)
        self._min_segment_fes = ps * 20
        n_excl = n_exclude_dims or max(3, int(np.ceil(dim * 0.15)))
        self.n_exclude_dims = min(n_excl, dim // 2)

        self.monitor = CurvatureMonitor(dim, k=min(7, ps // 4))
        self.archived_bests = []
        self.restart_log = []

        self._nlshade = NLSHADE(
            problem, dim,
            pop_size=pop_size,
            max_fe=max_fe,
        )

        self.best_fitness = self._nlshade.best_fitness
        self.best_solution = self._nlshade.best_solution.copy()

        self._best_at_last_check = self.best_fitness
        self._gens_without_improvement = 0
        self._base_stag_gens = self.stag_gens
        self._current_stag_gens = self.stag_gens
        self._best_before_restart = self.best_fitness
        self._n_restarts = 0
        self._heartbeat_counter = 0

        if self.verbose:
            _log.info(
                "INIT  D=%d  max_fe=%d  pop=%d  stag=%d  "
                "n_excl=%d  init_best=%.4e",
                dim, max_fe, ps, self.stag_gens,
                self.n_exclude_dims, self.best_fitness,
            )

    @property
    def fe_count(self):
        return self._nlshade.fe_count

    def _basin_distances(self, x):
        """Euclidean distances from x to every archived basin center."""
        return [float(np.linalg.norm(x - c)) for c in self.archived_bests]

    def run(self):
        convergence_log = [(self.fe_count, self.best_fitness)]

        while self.fe_count < self.max_fe:
            for _ in range(self.orc_period):
                if self.fe_count >= self.max_fe:
                    break
                self._nlshade.step()
                self._sync_best()

            convergence_log.append((self.fe_count, self.best_fitness))

            if self.fe_count >= self.max_fe:
                break

            if self.best_fitness < self._best_at_last_check - 1e-12:
                self._gens_without_improvement = 0
                self._best_at_last_check = self.best_fitness
            else:
                self._gens_without_improvement += self.orc_period

            self._heartbeat_counter += 1
            if self.verbose and self._heartbeat_counter % 10 == 0:
                pop = self._nlshade.pop
                fit = self._nlshade.fitness
                spread = float(np.mean(np.std(pop, axis=0)))
                _log.info(
                    "HEART  fe=%d/%d  best=%.4e  pop=%d  "
                    "spread=%.2f  fit=[%.4e, %.4e]  stag=%d/%d",
                    self.fe_count, self.max_fe, self.best_fitness,
                    len(pop), spread, float(fit.min()), float(fit.max()),
                    self._gens_without_improvement,
                    self._current_stag_gens,
                )

            stagnating = (self._gens_without_improvement
                          >= self._current_stag_gens)
            pop = self._nlshade.pop
            fit = self._nlshade.fitness

            if len(pop) < 8 or not stagnating:
                continue

            state, neg_frac, mean_k = self.monitor.classify(
                pop, fit, stagnating
            )

            if self.verbose:
                spread = float(np.mean(np.std(pop, axis=0)))
                _log.info(
                    "ORC    fe=%d  state=%s  neg_frac=%.3f  "
                    "mean_kappa=%.4f  pop=%d  spread=%.2f",
                    self.fe_count, state, neg_frac, mean_k,
                    len(pop), spread,
                )

            progress = self.fe_count / self.max_fe
            remaining = self.max_fe - self.fe_count
            if (state == "trapped"
                    and remaining > self._min_segment_fes
                    and progress < 0.85):
                self._do_restart()
                convergence_log.append((self.fe_count, self.best_fitness))

        if self.verbose:
            _log.info(
                "DONE   fe=%d  best=%.4e  restarts=%d",
                self.fe_count, self.best_fitness, self._n_restarts,
            )

        return self.best_solution, self.best_fitness, convergence_log

    def _sync_best(self):
        if self._nlshade.best_fitness < self.best_fitness:
            old = self.best_fitness
            self.best_fitness = self._nlshade.best_fitness
            self.best_solution = self._nlshade.best_solution.copy()
            if self.verbose and self.archived_bests and (old - self.best_fitness) > 1.0:
                dists = self._basin_distances(self.best_solution)
                dist_str = ", ".join(f"b{i}={d:.1f}" for i, d in enumerate(dists))
                _log.info(
                    "IMPROV fe=%d  %.4e -> %.4e  dist_to_basins=[%s]",
                    self.fe_count, old, self.best_fitness, dist_str,
                )

    def _select_exclusion_dims(self):
        """
        Use ORC Landscape Probing to identify dimensions where the
        fitness landscape has basin transitions (negative curvature).
        Falls back to heuristic if probing finds no negative curvature.
        """
        curvature_profile, n_evals, probe_info = (
            self.monitor.probe_curvature_profile(
                self.best_solution, self.lb, self.ub, self.problem,
            )
        )
        self._nlshade.fe_count += n_evals

        if self.verbose:
            asym = probe_info["fitness_asymmetry"]
            fmin, fmax = probe_info["probe_fitness_range"]
            _log.info(
                "PROBE  center_f=%.4e  probe_range=[%.4e, %.4e]  "
                "n_evals=%d",
                probe_info["center_fitness"], fmin, fmax, n_evals,
            )
            ranked = np.argsort(curvature_profile)
            top5 = ranked[:min(5, len(ranked))]
            parts = [f"d{d}={curvature_profile[d]:+.4f}" for d in top5]
            _log.info("PROBE  curvature (most negative): %s", ", ".join(parts))
            asym_ranked = np.argsort(-np.abs(asym))
            top5a = asym_ranked[:min(5, len(asym_ranked))]
            parts_a = [f"d{d}={asym[d]:+.1f}" for d in top5a]
            _log.info(
                "PROBE  fitness asymmetry f(+)-f(-) (largest |asym|): %s",
                ", ".join(parts_a),
            )

        n = self.n_exclude_dims
        method = "orc"

        if np.any(curvature_profile < -0.01):
            dims = np.argsort(curvature_profile)[:n]
        else:
            method = "fallback"
            all_centers = self.archived_bests + [self.best_solution]
            qualifying = []
            for d in range(self.dim):
                sides = [1 if c[d] > self.domain_center else -1
                         for c in all_centers]
                if len(set(sides)) == 1:
                    qualifying.append(d)

            if len(qualifying) >= n:
                dims = np.random.choice(qualifying, size=n, replace=False)
            else:
                extremeness = np.abs(self.best_solution - self.domain_center)
                dims = np.argsort(-extremeness)[:n]

        if self.verbose:
            excl_parts = []
            for d in dims:
                side = "upper" if self.best_solution[d] > self.domain_center else "lower"
                kept = "lower" if side == "upper" else "upper"
                excl_parts.append(
                    f"d{d}(x*={self.best_solution[d]:+.1f}, keep={kept}, "
                    f"kappa={curvature_profile[d]:+.4f})"
                )
            _log.info(
                "EXCL   method=%s  dims=[%s]",
                method, ", ".join(excl_parts),
            )

        return dims

    def _generate_obl_population(self, lb_arr, ub_arr, n_obl):
        """Generate opposition-based restart individuals."""
        obl_pop = np.empty((n_obl, self.dim))
        for i in range(n_obl):
            center = (lb_arr + ub_arr) / 2.0
            noise = np.random.uniform(-0.1, 0.1, self.dim) * (ub_arr - lb_arr)
            obl_pop[i] = np.clip(center + noise + (center - self.best_solution),
                                 lb_arr, ub_arr)
        return obl_pop

    def _enforce_basin_distance(self, pop, lb_arr, ub_arr, max_retries=5):
        """Regenerate individuals too close to any archived basin center."""
        if not self.archived_bests:
            return pop
        domain_diag = float(np.linalg.norm(ub_arr - lb_arr))
        min_dist = 0.1 * domain_diag / np.sqrt(self.dim)
        centers = np.array(self.archived_bests)
        for i in range(len(pop)):
            for _ in range(max_retries):
                dists = np.linalg.norm(centers - pop[i], axis=1)
                if dists.min() >= min_dist:
                    break
                pop[i] = np.random.uniform(lb_arr, ub_arr)
        return pop

    def _do_restart(self):
        """ORC-probed dimensional exclusion restart with OBL seeding."""
        self._n_restarts += 1

        improved = self.best_fitness < self._best_before_restart - 1e-12
        if not improved:
            self._current_stag_gens = min(
                self._current_stag_gens * 2,
                self._base_stag_gens * 8,
            )
        else:
            self._current_stag_gens = self._base_stag_gens

        if self.verbose:
            dists = self._basin_distances(self.best_solution)
            dist_str = (
                ", ".join(f"b{i}={d:.1f}" for i, d in enumerate(dists))
                if dists else "none"
            )
            _log.info(
                "RESTART #%d  fe=%d  error=%.4e  improved=%s  "
                "patience=%d  dist_prev=[%s]",
                self._n_restarts, self.fe_count, self.best_fitness,
                improved, self._current_stag_gens, dist_str,
            )

        self.archived_bests.append(self.best_solution.copy())
        self.restart_log.append(
            (self.fe_count, self.best_fitness, self._n_restarts)
        )

        dims = self._select_exclusion_dims()
        lb_new = np.full(self.dim, self.lb)
        ub_new = np.full(self.dim, self.ub)

        for d in dims:
            if self.best_solution[d] > self.domain_center:
                ub_new[d] = self.domain_center
            else:
                lb_new[d] = self.domain_center

        ps = self.initial_pop_size
        n_obl = ps // 2
        n_lhs = ps - n_obl

        obl_pop = self._generate_obl_population(lb_new, ub_new, n_obl)
        try:
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=self.dim)
            lhs_pop = lb_new + (ub_new - lb_new) * sampler.random(n=n_lhs)
        except Exception:
            lhs_pop = np.random.uniform(lb_new, ub_new, (n_lhs, self.dim))

        seed_pop = np.vstack([obl_pop, lhs_pop])
        seed_pop = self._enforce_basin_distance(seed_pop, lb_new, ub_new)

        pre_restart_best = self.best_fitness
        self._best_before_restart = self.best_fitness
        self._nlshade.reinitialize(
            keep_memory=True,
            lb_override=lb_new,
            ub_override=ub_new,
            seed_pop=seed_pop,
        )
        self._sync_best()
        self._gens_without_improvement = 0
        self._best_at_last_check = self.best_fitness

        if self.verbose:
            dists = self._basin_distances(self._nlshade.best_solution)
            dist_str = ", ".join(f"b{i}={d:.1f}" for i, d in enumerate(dists))
            _log.info(
                "SEED   post_init_best=%.4e  (was %.4e)  "
                "dist_to_basins=[%s]  n_obl=%d  n_lhs=%d",
                self.best_fitness, pre_restart_best, dist_str,
                n_obl, n_lhs,
            )


# ---------------------------------------------------------------------------
# Ablation baselines
# ---------------------------------------------------------------------------

class NLSHADEStagnationRestart:
    """
    NL-SHADE with stagnation-based restarts (no ORC, no directed seeding).
    Full-domain random LHS on every restart.
    """

    def __init__(self, problem, dim, max_fe=300_000,
                 pop_size=None, check_period=25, stag_gens=None):
        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.check_period = check_period
        self.stag_gens = stag_gens or max(50, 3 * dim)
        self.tabu_zones = []

        self._nlshade = NLSHADE(problem, dim, pop_size=pop_size, max_fe=max_fe)
        self.best_fitness = self._nlshade.best_fitness
        self.best_solution = self._nlshade.best_solution.copy()
        self._best_at_last_check = self.best_fitness
        self._gens_no_improv = 0

    @property
    def fe_count(self):
        return self._nlshade.fe_count

    def run(self):
        convergence_log = [(self.fe_count, self.best_fitness)]
        while self.fe_count < self.max_fe:
            for _ in range(self.check_period):
                if self.fe_count >= self.max_fe:
                    break
                self._nlshade.step()
                self._sync()
            convergence_log.append((self.fe_count, self.best_fitness))
            if self.fe_count >= self.max_fe:
                break

            if self.best_fitness < self._best_at_last_check - 1e-12:
                self._gens_no_improv = 0
                self._best_at_last_check = self.best_fitness
            else:
                self._gens_no_improv += self.check_period

            progress = self.fe_count / self.max_fe
            if self._gens_no_improv >= self.stag_gens and progress < 0.85:
                self.tabu_zones.append(self.best_solution.copy())
                self._nlshade.reinitialize(keep_memory=True)
                self._sync()
                self._gens_no_improv = 0
                self._best_at_last_check = self.best_fitness
                convergence_log.append((self.fe_count, self.best_fitness))

        return self.best_solution, self.best_fitness, convergence_log

    def _sync(self):
        if self._nlshade.best_fitness < self.best_fitness:
            self.best_fitness = self._nlshade.best_fitness
            self.best_solution = self._nlshade.best_solution.copy()


class NLSHADEPeriodicRestart:
    """NL-SHADE with periodic restarts (no ORC, no stagnation check)."""

    def __init__(self, problem, dim, max_fe=300_000,
                 pop_size=None, n_restarts=3):
        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.n_restarts = n_restarts
        self.restart_interval = max_fe // (n_restarts + 1)
        self.tabu_zones = []

        self._nlshade = NLSHADE(problem, dim, pop_size=pop_size, max_fe=max_fe)
        self.best_fitness = self._nlshade.best_fitness
        self.best_solution = self._nlshade.best_solution.copy()
        self._next_restart_fe = self.restart_interval

    @property
    def fe_count(self):
        return self._nlshade.fe_count

    def run(self):
        convergence_log = [(self.fe_count, self.best_fitness)]
        while self.fe_count < self.max_fe:
            self._nlshade.step()
            self._sync()

            if (self.fe_count >= self._next_restart_fe
                    and self.fe_count < self.max_fe * 0.85):
                self.tabu_zones.append(self.best_solution.copy())
                self._nlshade.reinitialize(keep_memory=True)
                self._sync()
                self._next_restart_fe += self.restart_interval
                convergence_log.append((self.fe_count, self.best_fitness))

        convergence_log.append((self.fe_count, self.best_fitness))
        return self.best_solution, self.best_fitness, convergence_log

    def _sync(self):
        if self._nlshade.best_fitness < self.best_fitness:
            self.best_fitness = self._nlshade.best_fitness
            self.best_solution = self._nlshade.best_solution.copy()

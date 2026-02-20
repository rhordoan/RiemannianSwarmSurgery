"""
RSS Optimizer: Geometric Landscape Decomposition (GLD).

Integrates L-SHADE (base optimizer) with Perelman-style Ricci flow
surgery and per-basin algorithm portfolio selection.

Key Mechanisms:
1. Fitness-informed persistent metric: edge weights encode landscape
   barriers and accumulate Ricci flow evolution across generations.
2. Curvature-aware mutation: F boosted on bottleneck edges (negative
   curvature / developing neck), reduced in basins (positive curvature).
3. Dual-signal surgery: population splits at neck pinch points
   detected by BOTH weight blowup (Ricci flow) AND topological
   confirmation (PH H0 component count increase).
4. Per-basin algorithm portfolio: after surgery, each basin's
   curvature profile selects the optimal optimizer (CMA-ES for smooth,
   L-SHADE for rugged, Nelder-Mead for tiny).
5. Saddle-directed exploration: developing necks trigger scout
   injection at saddle points and directional mutation bias across
   the saddle -- geometry replaces random stagnation restart.
6. Convergence detection: curvature uniformization triggers exploitation.
7. Stagnation escape: archive-guided reinitialization as fallback.
"""

import numpy as np
import networkx as nx
import logging
import math

from src.riemannian_swarm import RiemannianSwarm

logger = logging.getLogger(__name__)


class RSSOptimizer:
    """
    RSS = L-SHADE + Perelman Ricci Flow Surgery + Portfolio + Saddle Injection.

    Args:
        problem: Object with .evaluate(x) and .bounds = [lb, ub].
        pop_size: Initial population size.
        dim: Dimensionality.
        max_fe: Maximum function evaluations.
        archive_type: 'sheaf', 'tabu', or 'none'.
        enable_surgery: Enable topological surgery (for ablation).
        enable_flow: Enable Ricci flow (for ablation).
        enable_topology: Enable PH diagnostics (for ablation).
        enable_persistent_metric: Enable persistent metric (for ablation).
        enable_portfolio: Enable per-basin strategy selection (for ablation).
        enable_saddle_injection: Enable saddle-directed exploration (for ablation).
    """

    # Strategy constants
    STRATEGY_LSHADE = 'lshade'
    STRATEGY_CMAES = 'cmaes'
    STRATEGY_NELDER_MEAD = 'nelder_mead'

    # Portfolio thresholds
    SMOOTH_BASIN_KAPPA_VAR = 0.05    # Below this -> CMA-ES (ratio-contrast units)
    TINY_BASIN_SIZE = 8              # Below this -> Nelder-Mead

    # Saddle injection parameters
    SADDLE_INJECT_RATIO = 2.5        # Minimum weight ratio to trigger injection
    SADDLE_INJECT_COOLDOWN = 10      # Gens between saddle injections
    SADDLE_N_SCOUTS = 3              # Number of scouts per saddle
    SADDLE_MUTATION_BIAS = 0.5       # Strength of cross-saddle mutation bias

    def __init__(self, problem, pop_size=None, dim=10, max_fe=200000,
                 archive_type='sheaf',
                 enable_surgery=True,
                 enable_flow=True,
                 enable_topology=True,
                 enable_persistent_metric=True,
                 enable_portfolio=True,
                 enable_saddle_injection=True):
        self.problem = problem
        self.dim = dim
        self.max_fe = max_fe
        self.archive_type = archive_type
        self.enable_portfolio = enable_portfolio
        self.enable_saddle_injection = enable_saddle_injection

        if pop_size is None:
            pop_size = min(18 * dim, 100)
        self.pop_size_init = pop_size

        domain_width = problem.bounds[1] - problem.bounds[0]

        # Initialize population
        self.pop = np.random.uniform(
            problem.bounds[0], problem.bounds[1], (pop_size, dim)
        )
        self.fitness = np.array([problem.evaluate(x) for x in self.pop])
        self.fe_count = pop_size
        
        # Best tracking
        best_idx = np.argmin(self.fitness)
        self.best_found = self.fitness[best_idx]
        self.best_solution = self.pop[best_idx].copy()

        # RSS Engine
        self.rss = RiemannianSwarm(
            self.pop, dim,
            archive_type=archive_type,
            domain_width=domain_width,
            enable_surgery=enable_surgery,
            enable_flow=enable_flow,
            enable_topology=enable_topology,
            enable_persistent_metric=enable_persistent_metric,
        )
        max_gens = max_fe // pop_size
        self.rss.max_generations = max_gens

        # Sub-populations: initially one global population
        self.sub_pops = [{
            'pop': self.pop,
            'fitness': self.fitness,
            'rss': self.rss,
            'lshade_state': self._init_lshade_state(pop_size),
            'stag_counter': 0,
            'stag_best': self.best_found,
            'strategy': self.STRATEGY_LSHADE,
            'cmaes_state': None,
            'forced_cmaes': False,
        }]

        self.generation = 0

        # Surgery event log for visualization
        self.surgery_events = []

        # Stagnation escape parameters
        self.stag_threshold = max(15, pop_size // 3)
        self.stag_reinit_fraction = 0.3  # Reinit 30% of stagnated pop

        # Saddle injection tracking
        self.last_saddle_inject_gen = -100

        # No geo_confidence mechanism needed: geometry is inherently safe
        # (neck-only coupling never modifies basin-interior agents).

    @staticmethod
    def _lhs_init(n, dim, bounds):
        """Latin Hypercube Sampling: guaranteed one sample per stratum."""
        lb, ub = bounds[0], bounds[1]
        result = np.empty((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                lo = lb + (ub - lb) * perm[i] / n
                hi = lb + (ub - lb) * (perm[i] + 1) / n
                result[i, d] = np.random.uniform(lo, hi)
        return result

    def _init_lshade_state(self, N, H=6):
        """Initialize L-SHADE adaptation state for a sub-population.

        archive_max = N matches pure L-SHADE (archive max = pop_size_init).
        Using 2.6*N (tried earlier) creates a larger r2 pool that reduces
        DE mutation correlation with the current population, hurting
        convergence on multimodal CEC functions.
        """
        return {
            'M_F': np.full(H, 0.5),
            'M_CR': np.full(H, 0.5),
            'k': 0,
            'H': H,
            'archive': [],
            'archive_max': N,  # was int(N * 2.6); match pure L-SHADE
        }

    def _init_cmaes_state(self, pop, fitness):
        """Initialize CMA-ES state for a sub-population."""
        N, D = pop.shape
        best_idx = np.argmin(fitness)
        mean = pop[best_idx].copy()
        # Divide by 2 so post-surgery CMA-ES starts with basin-local spread
        # rather than a domain-wide sigma that slows convergence.
        sigma = max(np.std(pop, axis=0).mean() / 2.0, 0.5)

        # CMA-ES parameters
        mu = max(2, N // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1,
                  2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs

        state = {
            'mean': mean,
            'sigma': sigma,
            'C': np.eye(D),
            'pc': np.zeros(D),
            'ps': np.zeros(D),
            'mu': mu,
            'weights': weights,
            'mueff': mueff,
            'cc': cc,
            'cs': cs,
            'c1': c1,
            'cmu': cmu,
            'damps': damps,
            'eigeneval': 0,
            'B': np.eye(D),
            'D_diag': np.ones(D),
            'invsqrtC': np.eye(D),
            'gen': 0,
        }
        return state

    # ================================================================== #
    #  EVOLUTION STRATEGIES                                                #
    # ================================================================== #

    def _evolve_sub_pop(self, sp):
        """
        Run one generation of L-SHADE with boost-only geometric enhancements.

        The key design principle: geometry can ONLY BOOST exploration, never
        reduce it. This makes coupling inherently safe -- basin-interior agents
        run at pure L-SHADE F values, neck agents get an upward boost.

        Specifically:
        - F boost ONLY on agents with weight_ratio > DEVELOPING_RATIO (necks)
        - No F reduction for basin-interior agents (curvature > 0 is ignored)
        - Escape direction steering on neck agents only
        - Saddle-directed mutation bias (agents near neck centroids)
        - Sheaf archive repulsion as mutation bias (post-surgery)
        - Convergence-driven exploitation (lower F / higher CR when uniformized)
        """
        pop = sp['pop']
        fitness = sp['fitness']
        rss = sp['rss']
        state = sp['lshade_state']
        N = len(pop)

        if N < 4:
            return

        # Generate F and CR from success history (pure L-SHADE values,
        # geometric boosts applied per-agent in the mutation loop)
        F_vals = np.empty(N)
        CR_vals = np.empty(N)
        for i in range(N):
            r = np.random.randint(0, state['H'])
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + state['M_F'][r]
                if Fi > 0:
                    break
            F_vals[i] = min(Fi, 1.0)
            CRi = np.random.normal(state['M_CR'][r], 0.1)
            CR_vals[i] = np.clip(CRi, 0.0, 1.0)

        # --- Geometric context ---
        geo_active = (rss.graph is not None and rss.enable_flow)

        # Get developing neck info for saddle-directed mutation bias
        developing_necks = []
        if (geo_active and self.enable_saddle_injection
                and rss.graph is not None):
            developing_necks = rss.get_developing_neck_info()

        # Convergence-driven exploitation: only for post-surgery sub-pops
        # that have confirmed their basin is smooth. NOT for the main population
        # where premature exploitation hurts L-SHADE's adaptive schedule.
        if geo_active and rss.is_uniformized and rss.total_surgeries > 0:
            CR_vals = np.clip(CR_vals * 1.3, 0.0, 1.0)
            F_vals = np.clip(F_vals * 0.65, 0.05, 1.0)

        # p for pbest: fixed at 0.2, matching standard L-SHADE.
        # Adaptive decay (tried earlier) hurts multimodal functions by
        # narrowing pbest too aggressively and causing premature convergence.
        p = max(2.0 / N, 0.2)
        p_count = max(2, int(round(p * N)))
        sorted_indices = np.argsort(fitness)

        # Combined pool for r2 selection
        if state['archive']:
            combined = np.vstack([pop, np.array(state['archive'])])
            else:
            combined = pop.copy()

        new_pop = np.empty_like(pop)
        new_fitness = np.empty(N)
        success_F, success_CR, success_delta = [], [], []

        # Pre-compute pop spread once (used by neck boost + saddle bias)
        pop_spread = np.std(pop, axis=0).mean() + 1e-10

        for i in range(N):
            if self.fe_count >= self.max_fe:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                continue

            Fi = F_vals[i]
            CRi = CR_vals[i]

            # pbest selection
            pbest_idx = sorted_indices[np.random.randint(0, p_count)]

            # r1 from population
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, N)

            # r2 from population + archive
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len(combined))

            # current-to-pbest/1 mutation (pure L-SHADE)
            mutant = (pop[i]
                      + Fi * (pop[pbest_idx] - pop[i])
                      + Fi * (pop[r1] - combined[r2]))

            # NOTE: Neck coupling (F boost + escape direction per agent) was
            # removed. It disrupted L-SHADE's adaptive parameter schedule
            # and caused systematic convergence loss. The surgery cascade
            # (population split at confirmed basin boundary → L-SHADE per
            # basin) is the correct geometric intervention.

            # SADDLE-DIRECTED mutation bias:
            # For agents near a developing neck centroid, bias across saddle.
            if developing_necks and i < len(rss.swarm):
                for neck in developing_necks:
                    dist_to_neck = np.linalg.norm(
                        pop[i] - neck['centroid']
                    )
                    if (dist_to_neck < 3.0 * pop_spread
                            and neck['direction'] is not None):
                        cross_saddle = neck['direction']
                        sign = 1.0 if np.random.rand() > 0.5 else -1.0
                        bias_strength = self.SADDLE_MUTATION_BIAS * Fi
                        mutant += (sign * bias_strength
                                   * cross_saddle * pop_spread)
                        break  # Apply to nearest neck only

            # Archive repulsion as mutation bias (post-surgery, always useful)
            if (geo_active
                    and rss.archive is not None
                    and rss.archive.num_ghosts() > 0):
                parent_grad = None
                if rss.cached_fitness is not None \
                        and i < len(rss.cached_fitness):
                    parent_grad = rss.estimate_gradient(
                        i, rss.cached_fitness
                    )
                rep = rss.archive.repulsion(pop[i], parent_grad)
                if rep > 5.0:
                    closest_ghost = self._find_closest_ghost(
                        pop[i], rss.archive
                    )
                    if closest_ghost is not None:
                        away = pop[i] - closest_ghost
                        norm = np.linalg.norm(away)
                        if norm > 1e-10:
                            away = away / norm
                            push_strength = min(rep / 50.0, 1.0)
                            mutant += push_strength * Fi * away * pop_spread

            # Boundary handling: bounce-back
            lb, ub = self.problem.bounds[0], self.problem.bounds[1]
            for d in range(self.dim):
                if mutant[d] < lb:
                    mutant[d] = (lb + pop[i][d]) / 2.0
                elif mutant[d] > ub:
                    mutant[d] = (ub + pop[i][d]) / 2.0

            # Binomial crossover
            j_rand = np.random.randint(0, self.dim)
            trial = pop[i].copy()
            for d in range(self.dim):
                if np.random.rand() < CRi or d == j_rand:
                    trial[d] = mutant[d]

            # Evaluate trial
            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1
            
            # CLEAN greedy selection -- NO repulsion in selection
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    state['archive'].append(pop[i].copy())
                    success_F.append(Fi)
                    success_CR.append(CRi)
                    success_delta.append(abs(fitness[i] - f_trial))

                new_pop[i] = trial
                new_fitness[i] = f_trial

                if f_trial < self.best_found:
                    self.best_found = f_trial
                    self.best_solution = trial.copy()
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]

        sp['pop'] = new_pop
        sp['fitness'] = new_fitness
        rss.swarm = new_pop
        rss.cached_fitness = new_fitness

        # Update success history
        if success_F:
            delta = np.array(success_delta)
            weights = delta / (np.sum(delta) + 1e-30)
            sf = np.array(success_F)
            num = np.sum(weights * sf ** 2)
            den = np.sum(weights * sf)
            state['M_F'][state['k']] = num / den if den > 1e-30 else 0.5
            scr = np.array(success_CR)
            state['M_CR'][state['k']] = np.sum(weights * scr)
            state['k'] = (state['k'] + 1) % state['H']

        # Trim external archive
        while len(state['archive']) > state['archive_max']:
            idx = np.random.randint(0, len(state['archive']))
            state['archive'].pop(idx)

    def _evolve_cmaes(self, sp):
        """
        Run one generation of CMA-ES for smooth basins.

        CMA-ES excels on smooth unimodal landscapes. It is selected
        by the portfolio when a basin's curvature variance is low
        (uniformized by the Ricci flow), indicating a smooth basin.
        """
        pop = sp['pop']
        fitness = sp['fitness']
        cma_st = sp['cmaes_state']
        N, D = pop.shape

        if cma_st is None:
            cma_st = self._init_cmaes_state(pop, fitness)
            sp['cmaes_state'] = cma_st

        lb, ub = self.problem.bounds[0], self.problem.bounds[1]

        # Recompute mu from current N in case LPSR or surgery changed pop size.
        mu = max(2, N // 2)
        if cma_st['mu'] != mu:
            cma_st['mu'] = mu
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            cma_st['weights'] = w
            cma_st['mueff'] = 1.0 / np.sum(w ** 2)

        # Generate offspring
        offspring = np.empty((N, D))
        offspring_fit = np.empty(N)

        # Eigendecomposition update (periodic)
        cma_st['eigeneval'] += 1
        if cma_st['eigeneval'] > N / (cma_st['c1'] + cma_st['cmu']) / D / 10:
            cma_st['eigeneval'] = 0
            C = cma_st['C']
            C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
            try:
                D_eig, B = np.linalg.eigh(C)
                D_eig = np.maximum(D_eig, 1e-20)
                cma_st['D_diag'] = np.sqrt(D_eig)
                cma_st['B'] = B
                cma_st['invsqrtC'] = B @ np.diag(
                    1.0 / cma_st['D_diag']
                ) @ B.T
            except np.linalg.LinAlgError:
                cma_st['C'] = np.eye(D)
                cma_st['D_diag'] = np.ones(D)
                cma_st['B'] = np.eye(D)
                cma_st['invsqrtC'] = np.eye(D)

        for k in range(N):
            if self.fe_count >= self.max_fe:
                offspring[k] = pop[k]
                offspring_fit[k] = fitness[k]
                 continue

            z = np.random.randn(D)
            x = cma_st['mean'] + cma_st['sigma'] * (
                cma_st['B'] @ (cma_st['D_diag'] * z)
            )
            # Boundary handling
            x = np.clip(x, lb, ub)

            f = self.problem.evaluate(x)
            self.fe_count += 1
            offspring[k] = x
            offspring_fit[k] = f

            if f < self.best_found:
                self.best_found = f
                self.best_solution = x.copy()

        # Sort by fitness and select mu best
        rank = np.argsort(offspring_fit)
        selected = rank[:mu]
        weights = cma_st['weights']

        # Update mean
        old_mean = cma_st['mean'].copy()
        cma_st['mean'] = np.sum(
            weights[:, None] * offspring[selected], axis=0
        )

        # Update evolution paths
        mean_diff = cma_st['mean'] - old_mean
        ps = cma_st['ps']
        pc = cma_st['pc']
        cs = cma_st['cs']
        cc = cma_st['cc']
        mueff = cma_st['mueff']
        sigma = cma_st['sigma']

        ps = (1 - cs) * ps + np.sqrt(
            cs * (2 - cs) * mueff
        ) * cma_st['invsqrtC'] @ (mean_diff / sigma)
        cma_st['ps'] = ps

        chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        hsig = (np.linalg.norm(ps)
                / np.sqrt(1 - (1 - cs) ** (2 * (cma_st['gen'] + 1)))
                < (1.4 + 2 / (D + 1)) * chiN)

        pc = (1 - cc) * pc + hsig * np.sqrt(
            cc * (2 - cc) * mueff
        ) * (mean_diff / sigma)
        cma_st['pc'] = pc

        # Update covariance matrix
        c1 = cma_st['c1']
        cmu = cma_st['cmu']
        artmp = (offspring[selected] - old_mean) / sigma
        C = ((1 - c1 - cmu) * cma_st['C']
             + c1 * (np.outer(pc, pc)
                     + (1 - hsig) * cc * (2 - cc) * cma_st['C'])
             + cmu * (artmp.T @ np.diag(weights) @ artmp))
        cma_st['C'] = C

        # Update sigma
        sigma *= np.exp(
            (cs / cma_st['damps']) * (np.linalg.norm(ps) / chiN - 1)
        )
        sigma = np.clip(sigma, 1e-20, 1e6)
        cma_st['sigma'] = sigma
        cma_st['gen'] += 1

        # Replace population with combined best
        all_pop = np.vstack([pop, offspring])
        all_fit = np.concatenate([fitness, offspring_fit])
        best_indices = np.argsort(all_fit)[:N]
        sp['pop'] = all_pop[best_indices].copy()
        sp['fitness'] = all_fit[best_indices].copy()
        sp['rss'].swarm = sp['pop']
        sp['rss'].cached_fitness = sp['fitness']

    def _evolve_nelder_mead(self, sp):
        """
        Run one iteration of Nelder-Mead simplex for tiny basins.

        For very small populations (< 10 agents), Nelder-Mead is
        efficient: zero overhead, gradient-free, good for local
        refinement. Selected by the portfolio for tiny basins.
        """
        pop = sp['pop']
        fitness = sp['fitness']
        N, D = pop.shape

        if N < 3:
            return

        lb, ub = self.problem.bounds[0], self.problem.bounds[1]

        # Sort by fitness
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        # Centroid of all but worst
        centroid = np.mean(pop[:-1], axis=0)
        worst = pop[-1]
        f_worst = fitness[-1]
        f_best = fitness[0]
        f_second_worst = fitness[-2]

        operations_done = 0
        max_ops = min(N, 5)  # Limit FEs per generation

        while operations_done < max_ops and self.fe_count < self.max_fe:
            # Reflection
            xr = centroid + 1.0 * (centroid - worst)
            xr = np.clip(xr, lb, ub)
            f_r = self.problem.evaluate(xr)
            self.fe_count += 1
            operations_done += 1

            if f_r < self.best_found:
                self.best_found = f_r
                self.best_solution = xr.copy()

            if f_best <= f_r < f_second_worst:
                pop[-1] = xr
                fitness[-1] = f_r
                break
            elif f_r < f_best:
                # Expansion
                if self.fe_count >= self.max_fe:
                    pop[-1] = xr
                    fitness[-1] = f_r
                    break
                xe = centroid + 2.0 * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                f_e = self.problem.evaluate(xe)
                self.fe_count += 1
                operations_done += 1

                if f_e < self.best_found:
                    self.best_found = f_e
                    self.best_solution = xe.copy()

                if f_e < f_r:
                    pop[-1] = xe
                    fitness[-1] = f_e
                             else:
                    pop[-1] = xr
                    fitness[-1] = f_r
                break
            else:
                # Contraction
                if self.fe_count >= self.max_fe:
                    break
                if f_r < f_worst:
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    xc = centroid + 0.5 * (worst - centroid)
                xc = np.clip(xc, lb, ub)
                f_c = self.problem.evaluate(xc)
                self.fe_count += 1
                operations_done += 1

                if f_c < self.best_found:
                    self.best_found = f_c
                    self.best_solution = xc.copy()

                if f_c < f_worst:
                    pop[-1] = xc
                    fitness[-1] = f_c
                    break
                else:
                    # Shrink
                    for i in range(1, N):
                        if self.fe_count >= self.max_fe:
                            break
                        pop[i] = pop[0] + 0.5 * (pop[i] - pop[0])
                        pop[i] = np.clip(pop[i], lb, ub)
                        fitness[i] = self.problem.evaluate(pop[i])
                        self.fe_count += 1
                        operations_done += 1

                        if fitness[i] < self.best_found:
                            self.best_found = fitness[i]
                            self.best_solution = pop[i].copy()
                    break

        sp['pop'] = pop
        sp['fitness'] = fitness
        sp['rss'].swarm = pop
        sp['rss'].cached_fitness = fitness

    # ================================================================== #
    #  PER-BASIN ALGORITHM PORTFOLIO (Addition 1)                          #
    # ================================================================== #

    def _select_basin_strategy(self, sp):
        """
        Select evolution strategy based on basin's curvature profile.

        Uses Ricci flow curvature statistics (computed FOR FREE as part
        of the flow) to characterize each basin and select the optimal
        optimizer:

        - Low curvature variance + uniformized (smooth basin) -> CMA-ES
        - High curvature variance (rugged basin) -> L-SHADE
        - Tiny population after surgery (< TINY_BASIN_SIZE) AND
          uniformized -> Nelder-Mead

        Strategy switches only happen when the flow has had time to
        characterize the basin (is_uniformized or strong curvature signal).
        This prevents premature switching on LPSR-reduced populations.

        Returns: strategy string ('lshade', 'cmaes', 'nelder_mead')
        """
        if not self.enable_portfolio:
            return self.STRATEGY_LSHADE

        # Respect externally forced CMA-ES (post-surgery cascade or
        # 70% budget fallback). Do not let the portfolio override these.
        if sp.get('forced_cmaes', False):
            return self.STRATEGY_CMAES

        N = len(sp['pop'])
        rss = sp['rss']

        # Only switch strategies if the flow has had time to characterize
        # the basin. This prevents switching on the main population that
        # is just being reduced by LPSR.
        if not rss.enable_flow or rss.graph is None:
            return self.STRATEGY_LSHADE

        # Portfolio only activates after surgery has split the population
        # OR after curvature has uniformized. The point is to use geometry
        # to SELECT the right optimizer for each basin.
        has_had_surgery = rss.total_surgeries > 0

        # Tiny basin after surgery -> Nelder-Mead
        # Only if this is a post-surgery sub-population, not LPSR reduction
        if (N < self.TINY_BASIN_SIZE and N >= 3
                and has_had_surgery and rss.is_uniformized):
            return self.STRATEGY_NELDER_MEAD

        # Get curvature variance from the Ricci flow
        kappa_var = rss.get_curvature_variance()

        # Smooth basin (uniformized by flow) -> CMA-ES, but ONLY for
        # post-surgery sub-pops (has_had_surgery). For the main population,
        # stay with L-SHADE -- it handles archive-based diversity better
        # than CMA-ES and a premature switch hurts convergence.
        if (has_had_surgery
                and rss.is_uniformized
                and kappa_var < self.SMOOTH_BASIN_KAPPA_VAR
                and N >= max(self.dim + 1, 8)):
            return self.STRATEGY_CMAES

        # Rugged basin or pre-surgery main pop -> L-SHADE (default)
        return self.STRATEGY_LSHADE

    # ================================================================== #
    #  SADDLE-DIRECTED EXPLORATION (Addition 2)                            #
    # ================================================================== #

    def _saddle_injection(self, sp):
        """
        Saddle-directed exploration: inject scouts at detected saddle points.

        When the Ricci flow detects developing necks (weight ratio above
        SADDLE_INJECT_RATIO but below surgery threshold), this method:

        1. Locates the saddle: centroid of developing neck midpoints
        2. Injects scouts AT the saddle with perturbation perpendicular
           to the neck direction
        3. Replaces worst agents to maintain population size

        This is the geometric alternative to random stagnation restart:
        the flow tells us WHERE to explore (at the saddle), rather than
        exploring randomly.
        """
        rss = sp['rss']
        if rss.graph is None or not rss.enable_flow:
            return

        if (self.generation - self.last_saddle_inject_gen
                < self.SADDLE_INJECT_COOLDOWN):
            return

        developing_necks = rss.get_developing_neck_info()
        if not developing_necks:
            return

        # Filter to necks above injection threshold
        necks = [n for n in developing_necks
                 if n['max_ratio'] >= self.SADDLE_INJECT_RATIO]
        if not necks:
            return

        lb, ub = self.problem.bounds[0], self.problem.bounds[1]
        pop = sp['pop']
        fitness = sp['fitness']
        N = len(pop)

        if N < 6:  # Need room for scouts
            return

        # Sort by fitness to identify worst agents for replacement
        sorted_idx = np.argsort(fitness)
        worst_cursor = N - 1

        n_injected = 0
        for neck in necks:
            if n_injected >= self.SADDLE_N_SCOUTS:
                                                 break
                                     
            saddle_point = neck['centroid']
            neck_dir = neck['direction']

            # Inject scouts at/near the saddle
            for s in range(self.SADDLE_N_SCOUTS):
                if (worst_cursor < 2
                        or self.fe_count >= self.max_fe
                        or n_injected >= self.SADDLE_N_SCOUTS * 2):
                                         break
                                 
                replace_idx = sorted_idx[worst_cursor]
                worst_cursor -= 1

                # Scout position: saddle + perturbation
                pop_spread = np.std(pop, axis=0).mean() + 1e-10
                perturbation = np.random.randn(self.dim) * pop_spread * 0.3

                if neck_dir is not None:
                    # Perturbation mostly PERPENDICULAR to neck direction
                    # (explore the other side of the saddle)
                    proj = np.dot(perturbation, neck_dir) * neck_dir
                    perp = perturbation - proj
                    # Also add component ALONG neck direction to cross it
                    cross_sign = 1.0 if np.random.rand() > 0.5 else -1.0
                    scout = (saddle_point
                             + perp * 0.5
                             + cross_sign * neck_dir * pop_spread * 0.8)
                else:
                    scout = saddle_point + perturbation

                scout = np.clip(scout, lb, ub)
                f_scout = self.problem.evaluate(scout)
                self.fe_count += 1

                pop[replace_idx] = scout
                fitness[replace_idx] = f_scout
                n_injected += 1

                if f_scout < self.best_found:
                    self.best_found = f_scout
                    self.best_solution = scout.copy()

        if n_injected > 0:
            sp['pop'] = pop
            sp['fitness'] = fitness
            rss.swarm = pop
            rss.cached_fitness = fitness
            self.last_saddle_inject_gen = self.generation

            logger.debug(
                f"Saddle injection at gen {self.generation}: "
                f"injected {n_injected} scouts at "
                f"{len(necks)} developing neck(s)"
            )

    # ================================================================== #
    #  STAGNATION ESCAPE (fallback)                                        #
    # ================================================================== #

    def _find_closest_ghost(self, pos, archive):
        """Find centroid of closest ghost basin/neck."""
        closest = None
        best_dist = float('inf')
        if hasattr(archive, 'basin_ghosts'):
            for g in archive.basin_ghosts:
                d = np.linalg.norm(pos - g['centroid'])
                if d < best_dist:
                    best_dist = d
                    closest = g['centroid']
        if hasattr(archive, 'neck_ghosts'):
            for g in archive.neck_ghosts:
                d = np.linalg.norm(pos - g['centroid'])
                if d < best_dist:
                    best_dist = d
                    closest = g['centroid']
        # Fallback for TabuArchive
        if closest is None and hasattr(archive, 'ghosts'):
            for g in archive.ghosts:
                centroid = g[0] if isinstance(g, list) else g
                d = np.linalg.norm(pos - centroid)
                if d < best_dist:
                    best_dist = d
                    closest = centroid
        return closest

    def _handle_stagnation(self, sp):
        """
        Hybrid stagnation escape: opposition-based + archive-free random.

        Diversifies reinit strategies to combine systematic exploration
        (opposition-based learning) with breakthrough potential (random):
        1. Opposition-based: reflect best through domain center (1 agent)
        2. Best-guided: large perturbation around best (1 agent)
        3. Archive-free random: bulk of reinit, preserves lucky finds

        With saddle-directed exploration (Addition 2), geometry handles
        most escapes. This stagnation restart is a safety net.
        """
        current_best = float(np.min(sp['fitness']))
        if current_best < sp['stag_best'] - 1e-12:
            sp['stag_best'] = current_best
            sp['stag_counter'] = 0
        else:
            sp['stag_counter'] += 1

        if sp['stag_counter'] < self.stag_threshold:
            return

        # Stagnation detected -- reinitialize worst agents
        N = len(sp['pop'])
        n_reinit = max(2, int(N * self.stag_reinit_fraction))
        n_reinit = min(n_reinit, N - 2)  # Keep best 2

        if n_reinit < 2:
            return

        # Find worst agents
        sorted_idx = np.argsort(sp['fitness'])
        worst_indices = sorted_idx[-n_reinit:]
        best_idx = sorted_idx[0]
        best_pos = sp['pop'][best_idx].copy()

        lb, ub = self.problem.bounds[0], self.problem.bounds[1]
        center = (lb + ub) / 2.0
        rss = sp['rss']

        for j, idx in enumerate(worst_indices):
            if self.fe_count >= self.max_fe:
                break

            if j == 0:
                # First agent: opposition-based learning
                candidate = 2.0 * center - best_pos
                candidate += np.random.randn(self.dim) * (ub - lb) * 0.05
                candidate = np.clip(candidate, lb, ub)
            elif j == 1:
                # Second agent: best-guided large perturbation
                scale = (ub - lb) * (0.15 + 0.25 * np.random.rand())
                candidate = best_pos + np.random.randn(self.dim) * scale
                candidate = np.clip(candidate, lb, ub)
            else:
                # Remaining agents: archive-free random
                best_candidate = None
                best_rep = float('inf')

                for attempt in range(5):
                    cand = np.random.uniform(lb, ub, self.dim)
                    if rss.archive is not None:
                        rep = rss.archive.repulsion(cand, None)
                    else:
                        rep = 0.0
                    if rep < best_rep:
                        best_rep = rep
                        best_candidate = cand

                candidate = best_candidate

            f_new = self.problem.evaluate(candidate)
            self.fe_count += 1

            sp['pop'][idx] = candidate
            sp['fitness'][idx] = f_new

            if f_new < self.best_found:
                self.best_found = f_new
                self.best_solution = candidate.copy()

        rss.swarm = sp['pop']
        rss.cached_fitness = sp['fitness']
        sp['stag_counter'] = 0
        sp['stag_best'] = float(np.min(sp['fitness']))

        logger.debug(
            f"Stagnation escape at gen {self.generation}: "
            f"reinitialized {n_reinit} agents"
        )

    # ================================================================== #
    #  MAIN STEP                                                           #
    # ================================================================== #

    def step(self):
        """
        One generation of RSS optimization with full GLD pipeline.

        Pipeline:
        1. Per-basin strategy selection (portfolio)
        2. Evolve with selected strategy (L-SHADE / CMA-ES / NM)
        3. Saddle-directed exploration (if developing necks detected)
        4. Stagnation escape (fallback)
        5. RSS geometry step (Ricci flow + dual-signal surgery)
        6. Handle graph splits (create new sub-populations)
        7. LPSR + elitism + merge
        """
        self.generation += 1

        next_gen_pops = []

        for sp in self.sub_pops:
            if len(sp['pop']) < 4:
                if sp['rss'].archive:
                    indices = list(range(len(sp['pop'])))
                    sp['rss'].prune_sub_swarm(
                        indices, sp['fitness'], None
                    )
                continue

            # 1. Select strategy based on basin curvature profile
            strategy = self._select_basin_strategy(sp)
            if strategy != sp.get('strategy', self.STRATEGY_LSHADE):
                old_strat = sp.get('strategy', self.STRATEGY_LSHADE)
                sp['strategy'] = strategy
                logger.debug(
                    f"Basin strategy switch: {old_strat} -> {strategy} "
                    f"(pop={len(sp['pop'])}, gen={self.generation})"
                )
                # Initialize CMA-ES state if switching to it
                if strategy == self.STRATEGY_CMAES \
                        and sp.get('cmaes_state') is None:
                    sp['cmaes_state'] = self._init_cmaes_state(
                        sp['pop'], sp['fitness']
                    )

            # 2. Evolve with selected strategy
            if strategy == self.STRATEGY_CMAES:
                self._evolve_cmaes(sp)
            elif strategy == self.STRATEGY_NELDER_MEAD:
                self._evolve_nelder_mead(sp)
            else:
                self._evolve_sub_pop(sp)

            # 3. Saddle-directed exploration (Addition 2)
            # Only inject after surgery has confirmed a basin boundary.
            # Pre-surgery, saddle injection steals FEs from L-SHADE's
            # productive mutations without geometric justification.
            if (self.enable_saddle_injection
                    and sp['rss'].total_surgeries > 0):
                self._saddle_injection(sp)

            # 4. RSS geometry step
            sub_graphs = None
            neck_info = None

            if sp['rss'].enable_flow:
                # Gate: suppress surgery before 25% of budget.
                # CRITICAL: must disable surgery in the RiemannianSwarm itself,
                # not just veto the sub_graphs afterward.  Vetoing afterward
                # still increments rss.total_surgeries, burning the surgery
                # budget (max=2) before any genuine basin structure has formed.
                rss = sp['rss']
                gate_active = (self.fe_count < 0.25 * self.max_fe
                               and rss.enable_surgery)
                if gate_active:
                    rss.enable_surgery = False
                sub_graphs, neck_info = rss.step(sp['fitness'])
                if gate_active:
                    rss.enable_surgery = True

            # 6. Handle graph splits from dual-signal surgery
            if sub_graphs is not None and neck_info is not None:
                self.surgery_events.append(
                    (self.fe_count, len(sub_graphs))
                )

                for sub_g in sub_graphs:
                    comp_idx = list(sub_g.nodes())

                    if any(idx >= len(sp['pop']) for idx in comp_idx):
                        next_gen_pops.append(sp)
                        break

                    if len(comp_idx) < RiemannianSwarm.MIN_AGENTS:
                        if self.rss.archive:
                            valid_idx = [i for i in comp_idx
                                         if i < len(sp['pop'])]
                            if valid_idx:
                                self.rss.archive.store(
                                    sp['pop'][valid_idx]
                                )
                        continue

                    comp_pop = sp['pop'][comp_idx].copy()
                    comp_fit = sp['fitness'][comp_idx].copy()

                    dw = self.problem.bounds[1] - self.problem.bounds[0]
                    new_rss = RiemannianSwarm(
                        comp_pop, self.dim,
                        archive_type=self.archive_type,
                        domain_width=dw,
                        enable_surgery=sp['rss'].enable_surgery,
                        enable_flow=sp['rss'].enable_flow,
                        enable_topology=sp['rss'].enable_topology,
                        enable_persistent_metric=sp['rss'].enable_persistent_metric,
                    )
                    new_rss.archive = self.rss.archive
                    new_rss.max_generations = sp['rss'].max_generations
                    new_rss.cached_fitness = comp_fit
                    new_rss.total_surgeries = sp['rss'].total_surgeries
                    new_rss.max_surgeries = sp['rss'].max_surgeries

                    # CASCADE: surgery identifies a basin -> keep L-SHADE
                    # initially to let each sub-pop's adaptive parameters
                    # stabilize. The portfolio will switch to CMA-ES once
                    # curvature uniformizes inside the basin.
                    new_sp = {
                        'pop': comp_pop,
                        'fitness': comp_fit,
                        'rss': new_rss,
                        'lshade_state': self._init_lshade_state(
                            len(comp_pop)
                        ),
                        'stag_counter': 0,
                        'stag_best': float(np.min(comp_fit)),
                        # CASCADE: MST surgery has confirmed a basin boundary.
                        # The basin is approximately unimodal under the evolved
                        # metric -- switch immediately to CMA-ES for quadratic
                        # convergence. This is the core Surgery-Cascade:
                        # L-SHADE explores, surgery detects, CMA-ES exploits.
                        'strategy': self.STRATEGY_CMAES,
                        'cmaes_state': self._init_cmaes_state(
                            comp_pop, comp_fit
                        ),
                        'forced_cmaes': True,
                    }
                    logger.info(
                        f"SURGERY CASCADE: sub-pop (n={len(comp_pop)}) "
                        f"switching to CMA-ES in isolated basin "
                        f"(best={np.min(comp_fit):.4f})"
                    )
                    next_gen_pops.append(new_sp)
            else:
                next_gen_pops.append(sp)

        self.sub_pops = next_gen_pops if next_gen_pops else self.sub_pops

        # Geometry-gated unimodal fallback: after 70% budget, consider
        # transitioning from L-SHADE to CMA-ES -- but ONLY if the Ricci flow
        # geometry confirms a unimodal landscape structure (low curvature
        # variance, indicating the flow has uniformized the metric).
        #
        # On multimodal functions, the curvature remains heterogeneous even
        # at 70%; CMA-ES on a multi-basin population would converge to a
        # random local optimum. L-SHADE's archive-based diversity is the right
        # choice there. The Ricci flow curvature variance is the discriminant.
        if (self.fe_count > 0.70 * self.max_fe
                and len(self.sub_pops) == 1
                and self.sub_pops[0].get('strategy',
                                         self.STRATEGY_LSHADE) == self.STRATEGY_LSHADE
                and not self.sub_pops[0].get('forced_cmaes', False)
                and not self.surgery_events):
            sp = self.sub_pops[0]
            rss = sp['rss']
            # Check curvature uniformization: if the flow has made the metric
            # approximately uniform (low kappa variance), the landscape is
            # effectively unimodal from the current population's perspective.
            kappa_var = rss.get_curvature_variance()
            # With ratio-contrast curvature, kappa_var is bounded and
            # meaningful. kappa_var < SMOOTH_BASIN_KAPPA_VAR = 0.05 means
            # all edges have nearly identical stretch relative to neighbours --
            # the flow has uniformized, indicating a single-basin landscape.
            # On complex hybrid functions, the contrast between edge ratios
            # stays high even at 70% budget; the test correctly vetoes CMA-ES.
            geo_says_unimodal = kappa_var < self.SMOOTH_BASIN_KAPPA_VAR
            if geo_says_unimodal:
                sp['strategy'] = self.STRATEGY_CMAES
                sp['cmaes_state'] = self._init_cmaes_state(
                    sp['pop'], sp['fitness']
                )
                sp['forced_cmaes'] = True
                logger.info(
                    f"GEO-GATED FALLBACK: L-SHADE→CMA-ES at {self.fe_count} FEs "
                    f"(70% budget, kappa_var={kappa_var:.4f}, "
                    f"is_uniformized={rss.is_uniformized})"
                )
            else:
                logger.debug(
                    f"70% budget reached but geometry says multimodal: "
                    f"kappa_var={kappa_var:.4f} > {self.SMOOTH_BASIN_KAPPA_VAR} "
                    f"-- keeping L-SHADE for diversity"
                )

        # Global LPSR
        self._apply_lpsr()

        # Elitism
        self._apply_elitism()

        # Merge sub-pops that are too small
        self._merge_tiny_subpops()

        # Update global best
        for sp in self.sub_pops:
            if len(sp['fitness']) > 0:
                bi = np.argmin(sp['fitness'])
                if sp['fitness'][bi] < self.best_found:
                    self.best_found = sp['fitness'][bi]
                    self.best_solution = sp['pop'][bi].copy()

        return self.best_found

    def _apply_lpsr(self):
        """
        Global LPSR: reduce total population linearly across all sub-pops.
        Minimum population matches pure L-SHADE (4), bounded by MIN_AGENTS
        when the flow is enabled (needs at least k=3 neighbours).
        """
            progress = self.fe_count / self.max_fe
        # Use min_pop=4 matching pure L-SHADE. When flow is active, k decays
        # to 3 (k_min), so a graph with N>=4 remains valid. Surgery's
        # MIN_AGENTS=6 check prevents splits on tiny populations naturally.
        min_pop = 4
        target = int(round(
            self.pop_size_init + (min_pop - self.pop_size_init) * progress
        ))
        target = max(target, min_pop)

        current_total = sum(len(sp['pop']) for sp in self.sub_pops)
        if current_total <= target:
            return

        to_remove = current_total - target

                all_agents = []
                for sp_idx, sp in enumerate(self.sub_pops):
            for li, fit in enumerate(sp['fitness']):
                all_agents.append((fit, sp_idx, li))
                
                protected = set()
                for sp_idx, sp in enumerate(self.sub_pops):
            n_protect = min(2, len(sp['pop']))
            if n_protect > 0:
                best_local = np.argsort(sp['fitness'])[:n_protect]
                for li in best_local:
                    protected.add((sp_idx, li))

        sp_sizes = {i: len(sp['pop']) for i, sp in enumerate(self.sub_pops)}

        all_agents.sort(key=lambda x: x[0], reverse=True)
        removal = {i: set() for i in range(len(self.sub_pops))}
        removed = 0
        for fit, sp_idx, li in all_agents:
            if removed >= to_remove:
                        break
            if (sp_idx, li) not in protected:
                remaining = sp_sizes[sp_idx] - len(removal[sp_idx])
                # Use min_pop (=4, matching pure L-SHADE) not MIN_AGENTS (=6,
                # which is the surgery split floor). LPSR must reach min_pop=4
                # for the final exploitation phase to match L-SHADE.
                if remaining > min_pop:
                    removal[sp_idx].add(li)
                    removed += 1

                for sp_idx, sp in enumerate(self.sub_pops):
            if not removal[sp_idx]:
                        continue
            keep = [i for i in range(len(sp['pop']))
                    if i not in removal[sp_idx]]
            if keep:
                sp['pop'] = sp['pop'][keep]
                sp['fitness'] = sp['fitness'][keep]
                sp['rss'].swarm = sp['pop']

    def _apply_elitism(self):
        """Inject best-ever solution into largest sub-swarm (post-surgery only).

        Elitism injects the global best into the worst slot every generation.
        For multimodal functions, this is HARMFUL on the main population:
        it biases all DE mutation directions toward the current best (which may
        be a local optimum) and reduces diversity, preventing basin escape.
        Elitism is only beneficial AFTER surgery has confirmed basin separation:
        it prevents the best solution from being lost when sub-populations are
        being pruned by LPSR.
        """
        # Only apply when multiple sub-populations exist (post-surgery).
        # Pre-surgery, the single main population must explore freely.
        if len(self.sub_pops) <= 1 or not self.sub_pops or self.best_solution is None:
            return

        largest = max(self.sub_pops, key=lambda x: len(x['pop']))
        if len(largest['pop']) > 1:
            worst_idx = np.argmax(largest['fitness'])
            if largest['fitness'][worst_idx] > self.best_found:
                largest['pop'][worst_idx] = self.best_solution.copy()
                largest['fitness'][worst_idx] = self.best_found

    def _merge_tiny_subpops(self):
        """Merge sub-populations that have fallen below MIN_AGENTS."""
        if len(self.sub_pops) <= 1:
            return

        viable = []
        tiny = []
        for sp in self.sub_pops:
            if len(sp['pop']) >= RiemannianSwarm.MIN_AGENTS:
                viable.append(sp)
                    else:
                tiny.append(sp)

        if not tiny or not viable:
            return

        for tsp in tiny:
            tsp_centroid = np.mean(tsp['pop'], axis=0)

            best_dist = float('inf')
            best_sp = viable[0]
            for vsp in viable:
                vsp_centroid = np.mean(vsp['pop'], axis=0)
                dist = np.linalg.norm(tsp_centroid - vsp_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_sp = vsp

            best_sp['pop'] = np.vstack([best_sp['pop'], tsp['pop']])
            best_sp['fitness'] = np.concatenate(
                [best_sp['fitness'], tsp['fitness']]
            )
            best_sp['rss'].swarm = best_sp['pop']

        self.sub_pops = viable

    def run(self):
        """Run optimization until budget exhausted. Returns convergence history."""
        history = []
        log_interval = max(1, self.max_fe // 20)

        while self.fe_count < self.max_fe:
            best = self.step()
            history.append(best)

            if self.fe_count % log_interval < self.pop_size_init:
                total_agents = sum(
                    len(sp['pop']) for sp in self.sub_pops
                )
                n_ghosts = (self.rss.archive.num_ghosts()
                            if self.rss.archive else 0)
                n_subs = len(self.sub_pops)
                strategies = [sp.get('strategy', '?')
                              for sp in self.sub_pops]
                logger.info(
                    f"FE: {self.fe_count:>7d} | "
                    f"Best: {best:.4e} | "
                    f"Pop: {total_agents} | "
                    f"Subs: {n_subs} | "
                    f"Ghosts: {n_ghosts} | "
                    f"Surgeries: {self.rss.total_surgeries} | "
                    f"Strategies: {strategies}"
                )

        return history

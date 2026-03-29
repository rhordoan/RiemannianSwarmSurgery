"""
Baseline Algorithm Implementations for CEC 2022 Comparison.

Provides standardized wrappers around:
- DE/rand/1/bin (via scipy)
- CMA-ES (via cma package)
- L-SHADE (our standalone implementation, no geometry)
- PSO (custom implementation)

All baselines use the same interface:
    run_baseline(problem, dim, max_fe, seed) -> (final_error, history)

Usage:
    python benchmarks/baselines.py                    # Run all baselines on F12
    python benchmarks/baselines.py --func 1 --dim 10  # Specific function
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import time
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from benchmarks.run_cec2022 import CEC2022Wrapper, get_budget

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('Baselines')


# ======================================================================
# Baseline 1: DE/rand/1/bin via scipy
# ======================================================================

def run_de_scipy(problem, dim, max_fe, seed):
    """Vanilla DE/rand/1/bin via scipy.optimize.differential_evolution."""
    from scipy.optimize import differential_evolution

    np.random.seed(seed)
    history = []
    fe_counter = [0]

    def callback(xk, convergence):
        # xk is the best solution
        if fe_counter[0] <= max_fe:
            history.append(problem.evaluate(xk))
        return fe_counter[0] >= max_fe

    bounds = [(problem.bounds[0], problem.bounds[1])] * dim

    result = differential_evolution(
        problem.evaluate,
        bounds,
        strategy='rand1bin',
        maxiter=max_fe // 15,
        popsize=15,
        tol=0,
        seed=seed,
        callback=callback,
        polish=False,
    )

    final_error = problem.evaluate(result.x)
    if not history:
        history = [final_error]
    return final_error, history


# ======================================================================
# Baseline 2: CMA-ES via cma package
# ======================================================================

def run_cmaes(problem, dim, max_fe, seed):
    """CMA-ES via the cma package."""
    try:
        import cma
    except ImportError:
        logger.error("cma package not installed. Skipping CMA-ES.")
        return float('inf'), []

    np.random.seed(seed)
    x0 = np.random.uniform(problem.bounds[0], problem.bounds[1], dim)
    sigma0 = (problem.bounds[1] - problem.bounds[0]) / 4.0

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {
            'maxfevals': max_fe,
            'bounds': [problem.bounds[0], problem.bounds[1]],
            'verbose': -9,
            'verb_log': 0,
            'verb_disp': 0,
            'seed': seed,
        }
    )

    history = []
    while not es.stop() and es.result.evaluations < max_fe:
        solutions = es.ask()
        fitnesses = [problem.evaluate(x) for x in solutions]
        es.tell(solutions, fitnesses)
        history.append(min(fitnesses))

    final_error = es.result.fbest
    return final_error, history


# ======================================================================
# Baseline 3: L-SHADE (our implementation, no geometry)
# ======================================================================

def run_lshade_standalone(problem, dim, max_fe, seed):
    """Plain L-SHADE without any geometric enhancements."""
    from src.lshade import LSHADE

    np.random.seed(seed)
    pop_size = min(18 * dim, 100)

    opt = LSHADE(
        problem, dim,
        pop_size=pop_size,
        max_fe=max_fe,
    )

    history = opt.run()
    final_error = opt.best_fitness
    return final_error, history


# ======================================================================
# Baseline 4: PSO (Particle Swarm Optimization)
# ======================================================================

def run_pso(problem, dim, max_fe, seed):
    """Standard PSO with inertia weight."""
    np.random.seed(seed)

    pop_size = min(40, max(20, 2 * dim))
    lb, ub = problem.bounds[0], problem.bounds[1]

    # Initialize
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(
        -(ub - lb) * 0.1, (ub - lb) * 0.1, (pop_size, dim)
    )
    fitness = np.array([problem.evaluate(x) for x in positions])
    fe_count = pop_size

    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    history = [gbest_fit]

    # PSO parameters
    w_max, w_min = 0.9, 0.4
    c1, c2 = 2.0, 2.0

    while fe_count < max_fe:
        progress = fe_count / max_fe
        w = w_max - (w_max - w_min) * progress

        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)

        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))

        # Velocity clamping
        v_max = (ub - lb) * 0.2
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lb, ub)

        for i in range(pop_size):
            if fe_count >= max_fe:
                break
            fitness[i] = problem.evaluate(positions[i])
            fe_count += 1

            if fitness[i] < pbest_fit[i]:
                pbest_fit[i] = fitness[i]
                pbest_pos[i] = positions[i].copy()

                if fitness[i] < gbest_fit:
                    gbest_fit = fitness[i]
                    gbest_pos = positions[i].copy()

        history.append(gbest_fit)

    return gbest_fit, history


# ======================================================================
# Baseline 5: Random Search (sanity check baseline)
# ======================================================================

def run_random_search(problem, dim, max_fe, seed):
    """Pure random search. Should be beaten by everything."""
    np.random.seed(seed)
    lb, ub = problem.bounds[0], problem.bounds[1]

    best = float('inf')
    history = []

    for i in range(max_fe):
        x = np.random.uniform(lb, ub, dim)
        f = problem.evaluate(x)
        if f < best:
            best = f
        if i % 1000 == 0:
            history.append(best)

    return best, history


# ======================================================================
# Baseline 6: NL-SHADE-RSP-like (L-SHADE + restart + rank-based p)
# ======================================================================

def run_nlshade_rsp(problem, dim, max_fe, seed):
    """
    NL-SHADE-RSP-like: L-SHADE with restart mechanism and
    rank-based p_best adaptation. Simplified version of the
    CEC competition winner.
    """
    np.random.seed(seed)
    lb, ub = problem.bounds[0], problem.bounds[1]

    pop_size_init = min(25 * dim, 300)
    pop_size_min = 4
    H = 10

    best_global = float('inf')
    best_x = None
    history = []
    fe_count = 0

    n_restarts = 0
    max_restarts = 5

    while fe_count < max_fe and n_restarts <= max_restarts:
        pop_size = pop_size_init
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([problem.evaluate(x) for x in pop])
        fe_count += pop_size

        bi = np.argmin(fitness)
        if fitness[bi] < best_global:
            best_global = fitness[bi]
            best_x = pop[bi].copy()

        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = int(pop_size * 2.6)

        fe_at_start = fe_count
        budget_per_restart = (max_fe - fe_count) // max(
            1, max_restarts - n_restarts + 1
        )
        stagnation_counter = 0
        stagnation_best = best_global

        while fe_count < fe_at_start + budget_per_restart \
                and fe_count < max_fe:
            N = len(pop)
            if N < pop_size_min:
                break

            # Rank-based p (NL-SHADE-RSP innovation)
            progress = (fe_count - fe_at_start) / max(
                budget_per_restart, 1
            )
            p = max(2.0 / N, 0.05 + 0.20 * (1.0 - progress))
            p_count = max(2, int(round(p * N)))

            F_vals = np.empty(N)
            CR_vals = np.empty(N)
            for i in range(N):
                r = np.random.randint(0, H)
                while True:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[r]
                    if Fi > 0:
                        break
                F_vals[i] = min(Fi, 1.0)
                CRi = np.random.normal(M_CR[r], 0.1)
                CR_vals[i] = np.clip(CRi, 0.0, 1.0)

            sorted_idx = np.argsort(fitness)
            if archive:
                combined = np.vstack([pop, np.array(archive)])
            else:
                combined = pop.copy()

            new_pop = np.empty_like(pop)
            new_fitness = np.empty(N)
            sF, sCR, sDelta = [], [], []

            for i in range(N):
                if fe_count >= max_fe:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    continue

                pbest = sorted_idx[np.random.randint(0, p_count)]
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(0, N)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, len(combined))

                mutant = (pop[i]
                          + F_vals[i] * (pop[pbest] - pop[i])
                          + F_vals[i] * (pop[r1] - combined[r2]))

                for d in range(dim):
                    if mutant[d] < lb:
                        mutant[d] = (lb + pop[i][d]) / 2.0
                    elif mutant[d] > ub:
                        mutant[d] = (ub + pop[i][d]) / 2.0

                j_rand = np.random.randint(0, dim)
                trial = pop[i].copy()
                for d in range(dim):
                    if np.random.rand() < CR_vals[i] or d == j_rand:
                        trial[d] = mutant[d]

                f_trial = problem.evaluate(trial)
                fe_count += 1

                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        sF.append(F_vals[i])
                        sCR.append(CR_vals[i])
                        sDelta.append(abs(fitness[i] - f_trial))
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_global:
                        best_global = f_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            pop = new_pop
            fitness = new_fitness

            if sF:
                delta = np.array(sDelta)
                weights = delta / (np.sum(delta) + 1e-30)
                sf = np.array(sF)
                M_F[k] = np.sum(weights * sf ** 2) / (
                    np.sum(weights * sf) + 1e-30
                )
                M_CR[k] = np.sum(weights * np.array(sCR))
                k = (k + 1) % H

            while len(archive) > archive_max:
                archive.pop(np.random.randint(0, len(archive)))

            # LPSR
            progress = (fe_count - fe_at_start) / max(
                budget_per_restart, 1
            )
            new_pop_size = int(round(
                pop_size_init + (pop_size_min - pop_size_init) * progress
            ))
            new_pop_size = max(new_pop_size, pop_size_min)

            if len(pop) > new_pop_size:
                keep = np.argsort(fitness)[:new_pop_size]
                pop = pop[keep]
                fitness = fitness[keep]

            history.append(best_global)

            # Check stagnation for restart
            if best_global < stagnation_best - 1e-12:
                stagnation_best = best_global
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > 50:
                break

        n_restarts += 1

    if not history:
        history = [best_global]
    return best_global, history


# ======================================================================
# Baseline 7: jDE (Self-Adaptive DE)
# ======================================================================

def run_jde(problem, dim, max_fe, seed):
    """
    jDE: Self-adaptive Differential Evolution.

    F and CR are self-adapted per individual with probability tau.
    Reference: Brest et al. (2006).
    """
    np.random.seed(seed)
    lb, ub = problem.bounds[0], problem.bounds[1]

    pop_size = min(10 * dim, 100)
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([problem.evaluate(x) for x in pop])
    fe_count = pop_size

    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_x = pop[best_idx].copy()
    history = [best_fit]

    # Per-individual F and CR
    F_vals = np.full(pop_size, 0.5)
    CR_vals = np.full(pop_size, 0.9)
    tau1, tau2 = 0.1, 0.1

    while fe_count < max_fe:
        new_pop = np.empty_like(pop)
        new_fitness = np.empty(pop_size)
        new_F = F_vals.copy()
        new_CR = CR_vals.copy()

        for i in range(pop_size):
            if fe_count >= max_fe:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                continue

            # Self-adaptation
            if np.random.rand() < tau1:
                new_F[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau2:
                new_CR[i] = np.random.rand()

            Fi = new_F[i]
            CRi = new_CR[i]

            # DE/rand/1/bin
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(
                candidates, 3, replace=False
            )

            mutant = pop[r1] + Fi * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)

            j_rand = np.random.randint(0, dim)
            trial = pop[i].copy()
            for d in range(dim):
                if np.random.rand() < CRi or d == j_rand:
                    trial[d] = mutant[d]

            f_trial = problem.evaluate(trial)
            fe_count += 1

            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_x = trial.copy()
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                new_F[i] = F_vals[i]
                new_CR[i] = CR_vals[i]

        pop = new_pop
        fitness = new_fitness
        F_vals = new_F
        CR_vals = new_CR
        history.append(best_fit)

    return best_fit, history


# ======================================================================
# Baseline 8: BIPOP-CMA-ES (restart CMA-ES)
# ======================================================================

def run_bipop_cmaes(problem, dim, max_fe, seed):
    """
    BIPOP-CMA-ES: CMA-ES with interleaved restart strategy.

    Alternates between large (2x) and small random population restarts.
    Reference: Hansen (2009).
    """
    try:
        import cma
    except ImportError:
        logger.error("cma package not installed. Skipping BIPOP-CMA-ES.")
        return float('inf'), []

    np.random.seed(seed)
    lb, ub = problem.bounds[0], problem.bounds[1]
    sigma0 = (ub - lb) / 4.0

    best_global = float('inf')
    best_x = None
    history = []
    fe_count = 0

    default_popsize = int(4 + 3 * np.log(dim))
    large_popsize = default_popsize
    small_budget_used = 0
    large_budget_used = 0
    restart_count = 0

    while fe_count < max_fe:
        # Alternate large and small restarts
        if restart_count % 2 == 0 or restart_count == 0:
            popsize = large_popsize
            large_popsize = int(large_popsize * 2)
        else:
            # Small restart: random small population
            popsize = max(2, int(default_popsize * (
                0.5 * np.random.rand() ** 2
            )))

        popsize = min(popsize, max_fe - fe_count)
        if popsize < 2:
            break

        x0 = np.random.uniform(lb, ub, dim)
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {
                'maxfevals': max_fe - fe_count,
                'bounds': [lb, ub],
                'verbose': -9,
                'verb_log': 0,
                'verb_disp': 0,
                'seed': seed + restart_count,
                'popsize': popsize,
            }
        )

        while not es.stop() and fe_count < max_fe:
            solutions = es.ask()
            fitnesses = [problem.evaluate(x) for x in solutions]
            fe_count += len(solutions)
            es.tell(solutions, fitnesses)

            if min(fitnesses) < best_global:
                best_global = min(fitnesses)
                best_x = solutions[np.argmin(fitnesses)].copy()
            history.append(best_global)

        restart_count += 1

    if not history:
        history = [best_global]
    return best_global, history


# ======================================================================
# Registry
# ======================================================================

BASELINES = {
    'DE': run_de_scipy,
    'CMA-ES': run_cmaes,
    'L-SHADE': run_lshade_standalone,
    'PSO': run_pso,
    'Random': run_random_search,
    'NL-SHADE-RSP': run_nlshade_rsp,
    'jDE': run_jde,
    'BIPOP-CMA-ES': run_bipop_cmaes,
}


def run_all_baselines(func_nums=None, dims=None, n_runs=25,
                      output_dir='results'):
    """Run all baselines on specified functions and dimensions."""
    if func_nums is None:
        func_nums = list(range(1, 13))
    if dims is None:
        dims = [10]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for dim in dims:
        max_fe = get_budget(dim)
        for func_num in func_nums:
            for name, runner in BASELINES.items():
                logger.info(f"\n--- {name} on F{func_num} (D={dim}) ---")
                errors = []

                for run_id in range(n_runs):
                    seed = run_id + 2022
                    problem = CEC2022Wrapper(func_num, dim)

                    t0 = time.time()
                    error, history = runner(problem, dim, max_fe, seed)
                    elapsed = time.time() - t0

                    errors.append(error)
                    logger.info(
                        f"  Run {run_id:>2d}: Error={error:.4e}, "
                        f"Time={elapsed:.1f}s"
                    )

                    # Save convergence
                    hist_path = os.path.join(
                        output_dir,
                        f'{name}_F{func_num}_D{dim}_run{run_id}.csv'
                    )
                    with open(hist_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['generation', 'best_error'])
                        for gen, val in enumerate(history):
                            writer.writerow([gen, val])

                errors = np.array(errors)
                result = {
                    'algorithm': name,
                    'function': func_num,
                    'dimension': dim,
                    'mean': np.mean(errors),
                    'median': np.median(errors),
                    'std': np.std(errors),
                    'best': np.min(errors),
                    'worst': np.max(errors),
                }
                all_results.append(result)
                logger.info(
                    f"  Summary: Mean={result['mean']:.4e}, "
                    f"Std={result['std']:.4e}"
                )

    # Save summary
    summary_path = os.path.join(output_dir, 'baselines_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'function', 'dimension',
            'mean', 'median', 'std', 'best', 'worst'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info(f"\nBaseline summary saved to {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run baseline algorithms')
    parser.add_argument('--func', type=int, nargs='+', default=[12])
    parser.add_argument('--dim', type=int, nargs='+', default=[10])
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    run_all_baselines(
        func_nums=args.func,
        dims=args.dim,
        n_runs=args.runs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

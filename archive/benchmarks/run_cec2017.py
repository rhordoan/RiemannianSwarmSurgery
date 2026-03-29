"""
CEC 2017 Benchmark Runner for CARS.

Runs CARS and ablation baselines (NL-SHADE, NL-SHADE-R, jSO)
on CEC 2017 F1-F30 with parallel execution and statistical testing.

Usage:
    python benchmarks/run_cec2017.py                          # Quick (3 runs, D=30, composition only)
    python benchmarks/run_cec2017.py --full                   # Full benchmark
    python benchmarks/run_cec2017.py --func 21 --dim 30       # Single function
    python benchmarks/run_cec2017.py --workers 190            # Parallel execution
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import time
import logging
import numpy as np
import opfunu
import warnings
warnings.filterwarnings("ignore")

from src.cars import CARS, NLSHADEStagnationRestart, NLSHADEPeriodicRestart
from benchmarks.nlshade import NLSHADE
from benchmarks.jso import JSO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('CEC2017')


# -----------------------------------------------------------------------
# Problem wrapper
# -----------------------------------------------------------------------

class CEC2017Wrapper:
    """Wraps opfunu CEC 2017 functions. Returns error f(x) - f*."""

    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2017")
        self.problem = problem_class(ndim=dim)
        self.f_bias = self.problem.f_bias
        self.bounds = [-100.0, 100.0]

    def evaluate(self, x):
        return max(0.0, self.problem.evaluate(x) - self.f_bias)


def get_budget(dim):
    """CEC 2017 standard budget: 10000 * D."""
    return 10000 * dim


# -----------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------

def run_cars(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = CARS(problem, dim, max_fe=max_fe)
    best_sol, best_fit, log = algo.run()
    return best_fit, log, len(algo.restart_log)


def run_nlshade(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = NLSHADE(problem, dim, max_fe=max_fe)
    history = algo.run()
    log = [(algo.fe_count, algo.best_fitness)]
    return algo.best_fitness, log, 0


def run_nlshade_stag_restart(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = NLSHADEStagnationRestart(problem, dim, max_fe=max_fe)
    best_sol, best_fit, log = algo.run()
    return best_fit, log, len(algo.tabu_zones)


def run_nlshade_periodic_restart(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = NLSHADEPeriodicRestart(problem, dim, max_fe=max_fe)
    best_sol, best_fit, log = algo.run()
    return best_fit, log, len(algo.tabu_zones)


def run_jso(problem, dim, max_fe, seed):
    np.random.seed(seed)
    algo = JSO(problem, dim, max_fe=max_fe)
    history = algo.run()
    log = [(algo.fe_count, algo.best_fitness)]
    return algo.best_fitness, log, 0


ALGORITHMS = {
    "CARS":       run_cars,
    "jSO":        run_jso,
    "NL-SHADE":   run_nlshade,
    "NL-SHADE-R": run_nlshade_stag_restart,
}


# -----------------------------------------------------------------------
# Single-trial worker (for parallel execution)
# -----------------------------------------------------------------------

def _run_single_trial(task):
    """Execute one (algorithm, function, dimension, seed) trial."""
    algo_name = task["algo_name"]
    func_num = task["func_num"]
    dim = task["dim"]
    max_fe = task["max_fe"]
    seed = task["seed"]
    run_id = task["run_id"]
    output_dir = task["output_dir"]

    algo_fn = ALGORITHMS[algo_name]
    problem = CEC2017Wrapper(func_num, dim)

    t0 = time.time()
    best_fit, log, n_restarts = algo_fn(problem, dim, max_fe, seed)
    elapsed = time.time() - t0

    csv_path = os.path.join(
        output_dir,
        f"{algo_name}_F{func_num}_D{dim}_run{run_id}.csv"
    )
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["fe_count", "best_error"])
        for fe, err in log:
            w.writerow([fe, f"{err:.10e}"])

    return {
        "algo_name": algo_name,
        "func_num": func_num,
        "dim": dim,
        "run_id": run_id,
        "best_fit": best_fit,
        "n_restarts": n_restarts,
        "elapsed": elapsed,
    }


# -----------------------------------------------------------------------
# Statistical testing
# -----------------------------------------------------------------------

def compute_statistics(all_errors, algos, func_nums, dims):
    """Compute Wilcoxon signed-rank tests and Friedman test."""
    from scipy.stats import wilcoxon, friedmanchisquare

    stats_lines = []
    ref_algo = "CARS"

    for dim in dims:
        stats_lines.append(f"\n{'='*70}")
        stats_lines.append(f"Statistical Tests -- D={dim}")
        stats_lines.append(f"{'='*70}")

        header = f"{'F':>3s} | {'Comparison':>20s} | {'p-value':>10s} | {'Result':>8s}"
        stats_lines.append(header)
        stats_lines.append("-" * len(header))

        wins = {a: 0 for a in algos if a != ref_algo}
        losses = {a: 0 for a in algos if a != ref_algo}
        ties = {a: 0 for a in algos if a != ref_algo}

        all_ranks = []

        for func_num in func_nums:
            key_ref = (ref_algo, func_num, dim)
            if key_ref not in all_errors or len(all_errors[key_ref]) < 5:
                continue

            ref_errs = np.array(all_errors[key_ref])
            func_ranks = []

            for algo_name in algos:
                key = (algo_name, func_num, dim)
                if key not in all_errors:
                    continue
                errs = np.array(all_errors[key])
                func_ranks.append((algo_name, np.mean(errs)))

                if algo_name == ref_algo:
                    continue

                diff = ref_errs - errs
                if np.all(np.abs(diff) < 1e-15):
                    p_val = 1.0
                    result = "TIE"
                    ties[algo_name] += 1
                else:
                    try:
                        _, p_val = wilcoxon(ref_errs, errs, alternative='two-sided')
                    except Exception:
                        p_val = 1.0

                    if p_val < 0.05:
                        if np.mean(ref_errs) < np.mean(errs):
                            result = "WIN"
                            wins[algo_name] += 1
                        else:
                            result = "LOSS"
                            losses[algo_name] += 1
                    else:
                        result = "TIE"
                        ties[algo_name] += 1

                stats_lines.append(
                    f"F{func_num:>2d} | {ref_algo+' vs '+algo_name:>20s} | "
                    f"{p_val:10.4f} | {result:>8s}"
                )

            if len(func_ranks) == len(algos):
                sorted_ranks = sorted(func_ranks, key=lambda x: x[1])
                rank_map = {}
                for rank_pos, (a, _) in enumerate(sorted_ranks, 1):
                    rank_map[a] = rank_pos
                all_ranks.append(rank_map)

        stats_lines.append("")
        stats_lines.append("Summary (CARS vs each baseline):")
        for a in algos:
            if a == ref_algo:
                continue
            stats_lines.append(
                f"  vs {a:>12s}: {wins[a]:2d}W / {losses[a]:2d}L / {ties[a]:2d}T"
            )

        if len(all_ranks) >= 3:
            rank_matrix = {a: [] for a in algos}
            for rm in all_ranks:
                for a in algos:
                    if a in rm:
                        rank_matrix[a].append(rm[a])
            avg_ranks = {a: np.mean(rank_matrix[a]) for a in algos if rank_matrix[a]}
            stats_lines.append("")
            stats_lines.append("Average Friedman ranks:")
            for a in sorted(avg_ranks, key=avg_ranks.get):
                stats_lines.append(f"  {a:>12s}: {avg_ranks[a]:.2f}")

            if len(algos) >= 3 and all(len(rank_matrix[a]) == len(all_ranks) for a in algos):
                try:
                    groups = [rank_matrix[a] for a in algos]
                    stat, p_friedman = friedmanchisquare(*groups)
                    stats_lines.append(f"  Friedman chi2={stat:.2f}  p={p_friedman:.4f}")
                except Exception:
                    pass

    return "\n".join(stats_lines)


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def run_experiment(func_nums, dims, n_runs, output_dir="results/cec2017",
                   algos=None, n_workers=1):
    """Run full experiment grid with optional parallelism."""
    os.makedirs(output_dir, exist_ok=True)
    algos = algos or list(ALGORITHMS.keys())

    tasks = []
    for dim in dims:
        max_fe = get_budget(dim)
        for func_num in func_nums:
            for algo_name in algos:
                if algo_name not in ALGORITHMS:
                    logger.warning(f"Unknown algorithm: {algo_name}, skipping")
                    continue
                for run_id in range(n_runs):
                    seed = 1000 * func_num + 100 * dim + run_id
                    tasks.append({
                        "algo_name": algo_name,
                        "func_num": func_num,
                        "dim": dim,
                        "max_fe": max_fe,
                        "seed": seed,
                        "run_id": run_id,
                        "output_dir": output_dir,
                    })

    logger.info(f"Total tasks: {len(tasks)} ({len(algos)} algos x "
                f"{len(func_nums)} funcs x {len(dims)} dims x {n_runs} runs)")
    logger.info(f"Using {n_workers} worker(s)")

    results = []
    t_start = time.time()

    if n_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_trial, t): t for t in tasks}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    res = future.result()
                    results.append(res)
                    if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                        elapsed = time.time() - t_start
                        logger.info(
                            f"  [{i+1}/{len(tasks)}] completed "
                            f"({elapsed:.0f}s elapsed)"
                        )
                except Exception as e:
                    task = futures[future]
                    logger.error(
                        f"  FAILED: {task['algo_name']} F{task['func_num']} "
                        f"D{task['dim']} run{task['run_id']}: {e}"
                    )
    else:
        for i, task in enumerate(tasks):
            res = _run_single_trial(task)
            results.append(res)
            logger.info(
                f"  [{i+1}/{len(tasks)}] {res['algo_name']} F{res['func_num']} "
                f"D{res['dim']} run{res['run_id']}: "
                f"error={res['best_fit']:.6e}  "
                f"restarts={res['n_restarts']}  time={res['elapsed']:.1f}s"
            )

    total_time = time.time() - t_start
    logger.info(f"All {len(results)} tasks completed in {total_time:.0f}s")

    from collections import defaultdict
    grouped = defaultdict(list)
    all_errors = defaultdict(list)

    for res in results:
        key = (res["algo_name"], res["func_num"], res["dim"])
        grouped[key].append(res)
        all_errors[key].append(res["best_fit"])

    summary_rows = []
    for (algo_name, func_num, dim), res_list in sorted(grouped.items()):
        errors = [r["best_fit"] for r in res_list]
        restarts = [r["n_restarts"] for r in res_list]
        times = [r["elapsed"] for r in res_list]
        errs = np.array(errors)
        row = {
            "algorithm": algo_name,
            "function": func_num,
            "dimension": dim,
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "std": float(np.std(errs)),
            "best": float(np.min(errs)),
            "worst": float(np.max(errs)),
            "mean_restarts": float(np.mean(restarts)),
            "mean_time": float(np.mean(times)),
        }
        summary_rows.append(row)

    summary_path = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        logger.info(f"Summary saved to {summary_path}")

    stats_text = compute_statistics(all_errors, algos, func_nums, dims)
    stats_path = os.path.join(output_dir, "statistics.txt")
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    logger.info(f"Statistics saved to {stats_path}")
    print(stats_text)

    return summary_rows


def print_comparison_table(summary_rows):
    """Print a comparison table to stdout."""
    from collections import defaultdict
    by_func = defaultdict(dict)
    for row in summary_rows:
        key = (row["function"], row["dimension"])
        by_func[key][row["algorithm"]] = row

    algos = list(dict.fromkeys(r["algorithm"] for r in summary_rows))
    header = f"{'F':>3s} {'D':>3s} | " + " | ".join(f"{a:>14s}" for a in algos)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for (func, dim), algo_dict in sorted(by_func.items()):
        vals = []
        means = {a: algo_dict[a]["mean"] for a in algos if a in algo_dict}
        best_mean = min(means.values()) if means else float("inf")
        for a in algos:
            if a in algo_dict:
                m = algo_dict[a]["mean"]
                marker = " *" if m <= best_mean * 1.001 else "  "
                vals.append(f"{m:12.4e}{marker}")
            else:
                vals.append(f"{'N/A':>14s}")
        print(f"F{func:>2d} D{dim:>2d} | " + " | ".join(vals))

    print("=" * len(header) + "\n")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="CEC 2017 CARS benchmark")
    parser.add_argument("--full", action="store_true",
                        help="Full benchmark: F1-30, D=10/30, 25 runs")
    parser.add_argument("--func", type=int, nargs="+", default=None,
                        help="Function number(s)")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="Dimensions")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of independent runs")
    parser.add_argument("--algos", type=str, nargs="+", default=None,
                        help="Algorithm names to run")
    parser.add_argument("--output", type=str, default="results/cec2017",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default 1)")
    args = parser.parse_args()

    if args.full:
        func_nums = [i for i in range(1, 31) if i != 2]
        dims = [10, 30]
        n_runs = 25
    elif args.func:
        func_nums = args.func
        dims = args.dims or [30]
        n_runs = args.runs or 3
    else:
        func_nums = list(range(21, 31))
        dims = args.dims or [30]
        n_runs = args.runs or 3

    summary = run_experiment(
        func_nums, dims, n_runs,
        output_dir=args.output,
        algos=args.algos,
        n_workers=args.workers,
    )
    print_comparison_table(summary)

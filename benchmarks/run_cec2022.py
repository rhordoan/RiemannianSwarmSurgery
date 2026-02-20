"""
Full CEC 2022 Benchmark Runner.

Runs RSS (and optionally baselines/ablations) on all 12 CEC 2022
functions at D=10 and D=20 with 25 independent runs each.

Results are saved to CSV for statistical analysis.

Usage:
    python benchmarks/run_cec2022.py                    # Full benchmark
    python benchmarks/run_cec2022.py --func 12 --dim 10  # Single function
    python benchmarks/run_cec2022.py --quick              # Quick test (3 runs)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import time
import logging
import numpy as np
import opfunu
import warnings
warnings.filterwarnings("ignore")

from benchmarks.rss_optimizer import RSSOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('CEC2022')


class CEC2022Wrapper:
    """Wrapper for opfunu CEC 2022 functions. Returns error (f(x) - f*)."""

    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2022")
        self.problem = problem_class(ndim=dim)
        self.bounds = [-100, 100]
        self.f_bias = self.problem.f_bias

    def evaluate(self, x):
        return max(0.0, self.problem.evaluate(x) - self.f_bias)


def get_budget(dim):
    """Standard CEC 2022 evaluation budget."""
    if dim <= 10:
        return 200000
    elif dim <= 20:
        return 1000000
    else:
        return 1000000


def run_single(func_num, dim, seed, max_fe, archive_type='sheaf',
               enable_surgery=True, enable_flow=True, enable_topology=True,
               enable_persistent_metric=True, enable_portfolio=True,
               enable_saddle_injection=True, fitness_alpha=None):
    """Run a single trial. Returns (final_error, convergence_history, elapsed_time)."""
    np.random.seed(seed)
    problem = CEC2022Wrapper(func_num, dim)

    pop_size = min(18 * dim, 100)

    opt = RSSOptimizer(
        problem,
        pop_size=pop_size,
        dim=dim,
        max_fe=max_fe,
        archive_type=archive_type,
        enable_surgery=enable_surgery,
        enable_flow=enable_flow,
        enable_topology=enable_topology,
        enable_persistent_metric=enable_persistent_metric,
        enable_portfolio=enable_portfolio,
        enable_saddle_injection=enable_saddle_injection,
    )

    # Override fitness_alpha if specified (for Spatial-Only ablation)
    if fitness_alpha is not None:
        opt.rss.fitness_alpha = fitness_alpha
        for sp in opt.sub_pops:
            sp['rss'].fitness_alpha = fitness_alpha

    t0 = time.time()
    history = opt.run()
    elapsed = time.time() - t0

    final_error = history[-1] if history else float('inf')
    return final_error, history, elapsed


def run_experiment(func_nums=None, dims=None, n_runs=25,
                   archive_type='sheaf', label='RSS-Sheaf',
                   enable_surgery=True, enable_flow=True,
                   enable_topology=True, output_dir='results'):
    """
    Run full benchmark experiment.

    Args:
        func_nums: List of function numbers (default: 1-12).
        dims: List of dimensions (default: [10, 20]).
        n_runs: Number of independent runs per config.
        archive_type: Archive type for RSS.
        label: Algorithm label for output files.
        enable_surgery/flow/topology: Ablation switches.
        output_dir: Directory for output CSV files.
    """
    if func_nums is None:
        func_nums = list(range(1, 13))
    if dims is None:
        dims = [10]

    os.makedirs(output_dir, exist_ok=True)

    # Results storage
    all_results = []

    for dim in dims:
        max_fe = get_budget(dim)
        logger.info(f"\n{'='*60}")
        logger.info(f"Dimension: {dim}, Budget: {max_fe} FEs")
        logger.info(f"{'='*60}")

        for func_num in func_nums:
            logger.info(f"\n--- F{func_num} (D={dim}) ---")
            errors = []
            times = []

            for run_id in range(n_runs):
                seed = run_id + 2022

                error, history, elapsed = run_single(
                    func_num, dim, seed, max_fe,
                    archive_type=archive_type,
                    enable_surgery=enable_surgery,
                    enable_flow=enable_flow,
                    enable_topology=enable_topology,
                )

                errors.append(error)
                times.append(elapsed)

                # Save convergence history
                hist_path = os.path.join(
                    output_dir,
                    f'{label}_F{func_num}_D{dim}_run{run_id}.csv'
                )
                with open(hist_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['generation', 'best_error'])
                    for gen, val in enumerate(history):
                        writer.writerow([gen, val])

                logger.info(
                    f"  Run {run_id:>2d}: Error={error:.4e}, "
                    f"Time={elapsed:.1f}s"
                )

            errors = np.array(errors)
            result = {
                'algorithm': label,
                'function': func_num,
                'dimension': dim,
                'mean': np.mean(errors),
                'median': np.median(errors),
                'std': np.std(errors),
                'best': np.min(errors),
                'worst': np.max(errors),
                'mean_time': np.mean(times),
            }
            all_results.append(result)

            logger.info(
                f"  Summary: Mean={result['mean']:.4e}, "
                f"Median={result['median']:.4e}, "
                f"Std={result['std']:.4e}, "
                f"Best={result['best']:.4e}"
            )

    # Save summary CSV
    summary_path = os.path.join(output_dir, f'{label}_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'function', 'dimension',
            'mean', 'median', 'std', 'best', 'worst', 'mean_time'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info(f"\nSummary saved to {summary_path}")
    return all_results


def run_full_benchmark(func_nums=None, dims=None, n_runs=25,
                       output_dir='results'):
    """
    Run the complete A* benchmark protocol:
    1. Full GLD (all components)
    2. All baselines (DE, CMA-ES, L-SHADE, PSO, NL-SHADE-RSP, jDE, BIPOP)
    3. Ablation study (9 configurations)
    4. Statistical tests (Wilcoxon, Friedman)
    5. Publication figures

    Args:
        func_nums: Function numbers (default: 1-12).
        dims: Dimensions (default: [10, 20]).
        n_runs: Independent runs per config (default: 25).
        output_dir: Output directory.
    """
    if func_nums is None:
        func_nums = list(range(1, 13))
    if dims is None:
        dims = [10, 20]

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GLD Full Benchmark Protocol")
    logger.info(f"Functions: F{func_nums[0]}-F{func_nums[-1]}")
    logger.info(f"Dimensions: {dims}")
    logger.info(f"Runs per config: {n_runs}")
    logger.info("=" * 60)

    # 1. Run GLD (Full)
    logger.info("\n--- Phase 1: Full GLD ---")
    run_experiment(
        func_nums=func_nums,
        dims=dims,
        n_runs=n_runs,
        label='GLD',
        output_dir=output_dir,
    )

    # 2. Run baselines
    logger.info("\n--- Phase 2: Baselines ---")
    try:
        from benchmarks.baselines import run_all_baselines
        run_all_baselines(
            func_nums=func_nums,
            dims=dims,
            n_runs=n_runs,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error(f"Baseline run failed: {e}")

    # 3. Ablation study
    logger.info("\n--- Phase 3: Ablation ---")
    try:
        from benchmarks.ablation import run_ablation
        run_ablation(
            func_nums=func_nums,
            dims=dims,
            n_runs=n_runs,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error(f"Ablation run failed: {e}")

    # 4. Statistical tests
    logger.info("\n--- Phase 4: Statistical Tests ---")
    try:
        from benchmarks.statistics import (
            load_results, wilcoxon_test, friedman_test,
            generate_latex_table,
        )
        data = load_results(output_dir)
        if data:
            wilcoxon_results = wilcoxon_test(data, reference='GLD')
            stat, p_value, avg_ranks = friedman_test(data)
            generate_latex_table(
                data, wilcoxon_results,
                os.path.join(output_dir, 'results_table.tex')
            )
            if avg_ranks:
                logger.info("Algorithm rankings:")
                for algo, rank in sorted(
                    avg_ranks.items(), key=lambda x: x[1]
                ):
                    logger.info(f"  {algo}: {rank:.2f}")
    except Exception as e:
        logger.error(f"Statistical tests failed: {e}")

    # 5. Figures
    logger.info("\n--- Phase 5: Publication Figures ---")
    try:
        from benchmarks.visualize import (
            load_convergence_histories, plot_convergence,
            plot_ablation_bar, plot_critical_difference,
        )
        conv_data = load_convergence_histories(output_dir)
        if conv_data:
            for func in func_nums:
                for dim in dims:
                    plot_convergence(
                        conv_data, func, dim,
                        os.path.join(
                            output_dir,
                            f'convergence_F{func}_D{dim}.png'
                        )
                    )

        abl_path = os.path.join(output_dir, 'ablation_summary.csv')
        if os.path.exists(abl_path):
            plot_ablation_bar(
                abl_path,
                os.path.join(output_dir, 'ablation_chart.png')
            )

        if avg_ranks:
            from benchmarks.visualize import plot_critical_difference
            plot_critical_difference(
                avg_ranks,
                os.path.join(output_dir, 'cd_diagram.png')
            )
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark protocol complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='CEC 2022 Benchmark Runner')
    parser.add_argument('--func', type=int, nargs='+', default=None,
                        help='Function numbers (default: 1-12)')
    parser.add_argument('--dim', type=int, nargs='+', default=[10],
                        help='Dimensions (default: 10)')
    parser.add_argument('--runs', type=int, default=25,
                        help='Number of runs (default: 25)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 3 runs, F12 only')
    parser.add_argument('--full', action='store_true',
                        help='Full A* benchmark: all functions, baselines, '
                             'ablation, stats, figures')
    parser.add_argument('--archive', type=str, default='sheaf',
                        choices=['sheaf', 'tabu', 'none'],
                        help='Archive type')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    if args.full:
        run_full_benchmark(
            func_nums=args.func,
            dims=args.dim,
            n_runs=args.runs,
            output_dir=args.output,
        )
    elif args.quick:
        func_nums = [12]
        n_runs = 3
        run_experiment(
            func_nums=func_nums,
            dims=args.dim,
            n_runs=n_runs,
            archive_type=args.archive,
            output_dir=args.output,
        )
    else:
        func_nums = args.func
        n_runs = args.runs
        run_experiment(
            func_nums=func_nums,
            dims=args.dim,
            n_runs=n_runs,
            archive_type=args.archive,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()

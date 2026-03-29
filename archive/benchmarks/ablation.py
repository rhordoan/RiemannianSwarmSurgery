"""
Ablation Study Framework for Geometric Landscape Decomposition (GLD).

Tests 9 configurations to isolate the contribution of each
geometric mechanism in the Perelman surgery pipeline:

1. Full-GLD:          All components (flow + dual-signal surgery + portfolio + saddle)
2. Spatial-Only:      Same flow/surgery but Euclidean-only weights (proves fitness weighting)
3. No-Persistence:    Graph rebuilt fresh each gen (proves persistent metric)
4. No-Surgery:        Flow runs but never cuts (proves surgery matters)
5. No-Archive:        Surgery without ghost memory (proves archive matters)
6. No-Flow:           No Ricci flow at all (just L-SHADE + archive)
7. No-Portfolio:      No per-basin algorithm selection (proves portfolio matters)
8. No-Saddle:         No saddle-directed exploration (proves saddle injection matters)
9. Pure-L-SHADE:      Base optimizer only (zero geometric overhead)

Usage:
    python benchmarks/ablation.py                     # Full ablation
    python benchmarks/ablation.py --func 12 --dim 10  # Single function
    python benchmarks/ablation.py --quick              # Quick test
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from benchmarks.run_cec2022 import CEC2022Wrapper, get_budget, run_single
from benchmarks.baselines import run_lshade_standalone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('Ablation')


# Ablation configurations for the GLD architecture
CONFIGS = {
    'Full-GLD': {
        'archive_type': 'sheaf',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': True,
        'enable_saddle_injection': True,
    },
    'Spatial-Only-Metric': {
        'archive_type': 'sheaf',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': True,
        'enable_saddle_injection': True,
        'fitness_alpha': 0.0,  # Euclidean-only initial weights
    },
    'No-Persistence': {
        'archive_type': 'sheaf',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': False,  # Metric amnesia
        'enable_portfolio': True,
        'enable_saddle_injection': True,
    },
    'No-Surgery': {
        'archive_type': 'sheaf',
        'enable_surgery': False,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': True,
        'enable_saddle_injection': True,
    },
    'No-Archive': {
        'archive_type': 'none',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': True,
        'enable_saddle_injection': True,
    },
    'No-Flow': {
        'archive_type': 'sheaf',
        'enable_surgery': False,
        'enable_flow': False,
        'enable_topology': False,
        'enable_persistent_metric': True,
        'enable_portfolio': False,  # No flow => no curvature => no portfolio
        'enable_saddle_injection': False,  # No flow => no developing necks
    },
    'No-Portfolio': {
        'archive_type': 'sheaf',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': False,  # Always L-SHADE
        'enable_saddle_injection': True,
    },
    'No-Saddle-Injection': {
        'archive_type': 'sheaf',
        'enable_surgery': True,
        'enable_flow': True,
        'enable_topology': True,
        'enable_persistent_metric': True,
        'enable_portfolio': True,
        'enable_saddle_injection': False,  # No saddle scouts
    },
    # Pure L-SHADE handled separately
}


def run_ablation(func_nums=None, dims=None, n_runs=25,
                 output_dir='results'):
    """
    Run complete ablation study.

    Args:
        func_nums: List of function numbers.
        dims: List of dimensions.
        n_runs: Number of independent runs.
        output_dir: Output directory.
    """
    if func_nums is None:
        func_nums = [11, 12]
    if dims is None:
        dims = [10]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for dim in dims:
        max_fe = get_budget(dim)
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation Study: D={dim}, Budget={max_fe}")
        logger.info(f"{'='*60}")

        for func_num in func_nums:
            logger.info(f"\n--- F{func_num} (D={dim}) ---")

            for config_name, config in CONFIGS.items():
                logger.info(f"\n  Config: {config_name}")
                errors = []

                for run_id in range(n_runs):
                    seed = run_id + 2022
                    error, history, elapsed = run_single(
                        func_num, dim, seed, max_fe,
                        **config,
                    )
                    errors.append(error)

                    hist_path = os.path.join(
                        output_dir,
                        f'ablation_{config_name}_F{func_num}_D{dim}_run{run_id}.csv'
                    )
                    with open(hist_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['generation', 'best_error'])
                        for gen, val in enumerate(history):
                            writer.writerow([gen, val])

                errors = np.array(errors)
                result = {
                    'config': config_name,
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
                    f"    Mean={result['mean']:.4e}, "
                    f"Std={result['std']:.4e}, "
                    f"Best={result['best']:.4e}"
                )

            # Pure L-SHADE
            logger.info(f"\n  Config: Pure-LSHADE")
            errors = []
            for run_id in range(n_runs):
                seed = run_id + 2022
                problem = CEC2022Wrapper(func_num, dim)
                error, history = run_lshade_standalone(
                    problem, dim, max_fe, seed
                )
                errors.append(error)

            errors = np.array(errors)
            result = {
                'config': 'Pure-LSHADE',
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
                f"    Mean={result['mean']:.4e}, "
                f"Std={result['std']:.4e}, "
                f"Best={result['best']:.4e}"
            )

    # Save summary
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'config', 'function', 'dimension',
            'mean', 'median', 'std', 'best', 'worst'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info(f"\nAblation summary saved to {summary_path}")
    _print_comparison_table(all_results)
    return all_results


def _print_comparison_table(results):
    """Print formatted comparison table."""
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY RESULTS")
    logger.info(f"{'='*80}")

    funcs = sorted(set(r['function'] for r in results))
    configs = sorted(set(r['config'] for r in results))

    header = f"{'Config':<24}"
    for f in funcs:
        header += f"{'F' + str(f) + ' Mean':<18}{'F' + str(f) + ' Std':<15}"
    logger.info(header)
    logger.info("-" * len(header))

    for config in configs:
        line = f"{config:<24}"
        for f in funcs:
            matching = [r for r in results
                        if r['config'] == config and r['function'] == f]
            if matching:
                r = matching[0]
                line += f"{r['mean']:<18.4e}{r['std']:<15.4e}"
            else:
                line += f"{'N/A':<18}{'N/A':<15}"
        logger.info(line)


def main():
    parser = argparse.ArgumentParser(description='RSS Ablation Study')
    parser.add_argument('--func', type=int, nargs='+', default=[11, 12])
    parser.add_argument('--dim', type=int, nargs='+', default=[10])
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    if args.quick:
        run_ablation(
            func_nums=[12],
            dims=[10],
            n_runs=3,
            output_dir=args.output,
        )
    else:
        run_ablation(
            func_nums=args.func,
            dims=args.dim,
            n_runs=args.runs,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()

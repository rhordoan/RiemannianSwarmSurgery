"""
TMI CEC 2022 Benchmark: Four-Variant Comparison.

Variants:
    A  – L-SHADE           (linear PSR baseline)
    B  – L-SHADE  + TMI    (our L-SHADE contribution)
    C  – NL-SHADE          (SOTA baseline, Stanovov et al. CEC 2022)
    D  – NL-SHADE + TMI    (our proposed full system)

Protocol:
    - CEC 2022 benchmark suite, functions F1–F12
    - D = 10, max_FE = 200,000 (competition standard)
    - 30 independent runs per (variant, function) combination
    - Wilcoxon rank-sum test: D vs C, B vs A  (two-sided, alpha = 0.05)

Usage:
    python benchmarks/run_tmi_benchmark.py            # Full benchmark (30 seeds)
    python benchmarks/run_tmi_benchmark.py --seeds 5  # Quick 5-seed test
    python benchmarks/run_tmi_benchmark.py --funcs 7 8 12   # Specific functions
    python benchmarks/run_tmi_benchmark.py --dim 20          # Higher dimension
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import wilcoxon
import opfunu
warnings.filterwarnings('ignore')

from src.lshade import LSHADE
from benchmarks.nlshade import NLSHADE
from benchmarks.tmi_optimizer import TMIOptimizer


# ---------------------------------------------------------------------------
# Problem wrapper
# ---------------------------------------------------------------------------

class CEC2022Problem:
    """
    Thin wrapper around opfunu CEC 2022 functions.
    Returns error: max(0, f(x) - f*).
    """

    def __init__(self, func_num: int, dim: int):
        self.func_num = func_num
        self.dim = dim
        cls = getattr(opfunu.cec_based, f'F{func_num}2022')
        self._prob = cls(ndim=dim)
        self.bounds = [-100.0, 100.0]
        self.f_bias = self._prob.f_bias

    def evaluate(self, x: np.ndarray) -> float:
        return max(0.0, float(self._prob.evaluate(x)) - self.f_bias)

    def __repr__(self):
        return f'CEC2022-F{self.func_num} (D={self.dim})'


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Top-level function for multiprocessing.Pool.

    args = (variant, func_num, dim, seed, max_fe)

    Returns dict with run results.
    """
    variant, func_num, dim, seed, max_fe = args
    np.random.seed(seed)

    problem = CEC2022Problem(func_num, dim)
    pop_size = min(18 * dim, 100)

    t0 = time.perf_counter()

    if variant == 'A':
        opt = LSHADE(problem, dim, pop_size, max_fe)
        opt.run()
        final_error = opt.best_fitness
        inj_count, inj_fes, saddles = 0, 0, 0

    elif variant == 'B':
        opt = TMIOptimizer(problem, base='lshade', dim=dim,
                           pop_size=pop_size, max_fe=max_fe)
        opt.run()
        final_error = opt.best_fitness
        stats = opt.get_run_stats()
        inj_count = stats['injection_count']
        inj_fes = stats['total_injection_fes']
        saddles = stats['saddles_archived']

    elif variant == 'C':
        opt = NLSHADE(problem, dim, pop_size, max_fe)
        opt.run()
        final_error = opt.best_fitness
        inj_count, inj_fes, saddles = 0, 0, 0

    elif variant == 'D':
        opt = TMIOptimizer(problem, base='nlshade', dim=dim,
                           pop_size=pop_size, max_fe=max_fe)
        opt.run()
        final_error = opt.best_fitness
        stats = opt.get_run_stats()
        inj_count = stats['injection_count']
        inj_fes = stats['total_injection_fes']
        saddles = stats['saddles_archived']

    else:
        raise ValueError(f'Unknown variant: {variant}')

    elapsed = time.perf_counter() - t0

    return {
        'variant': variant,
        'func_num': func_num,
        'dim': dim,
        'seed': seed,
        'final_error': final_error,
        'elapsed_s': elapsed,
        'injection_count': inj_count,
        'injection_fes': inj_fes,
        'saddles_archived': saddles,
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _wilcoxon_p(errors_new: np.ndarray, errors_base: np.ndarray) -> tuple:
    """
    Two-sided Wilcoxon signed-rank test.

    Returns (statistic, p_value). If all differences are zero, returns (0, 1.0).
    """
    diff = errors_new - errors_base
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(errors_new, errors_base, alternative='two-sided')
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0


def _significance_marker(p: float) -> str:
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _improvement_marker(mean_new: float, mean_base: float, p: float) -> str:
    """Show whether TMI significantly improves (+), worsens (-), or is neutral (~)."""
    if p >= 0.05:
        return '~'
    return '+' if mean_new < mean_base else '-'


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

FUNC_LABELS = {
    1: 'Zakharov (unimodal)',
    2: 'Rosenbrock (unimodal)',
    3: 'Expanded Schaffer F6 (unimodal)',
    4: 'Non-continuous Rastrigin (multimodal)',
    5: 'Levy (multimodal)',
    6: 'Modified Schwefel (multimodal)',
    7: 'Bent Cigar (multimodal)',
    8: 'HGBat (multimodal)',
    9: 'Katsuura (multimodal)',
    10: 'Ackley (multimodal)',
    11: 'Weierstrass (multimodal)',
    12: 'Expanded Griewank+Rosenbrock (multimodal)',
}


def _print_summary_table(results_by_func: dict, func_nums: list, n_seeds: int):
    """Print a formatted table matching CEC paper conventions."""

    header_fmt = '{:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}  {:>6}  {:>4}  {:>4}'
    row_fmt    = '{:>4}  {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>6.3f}  {:>6.3f}  {:>4}  {:>4}'

    print()
    print('=' * 85)
    print(f'TMI CEC 2022 Benchmark  |  D=10  |  {n_seeds} seeds  |  max_FE=200,000')
    print('=' * 85)
    print('Variants: A=L-SHADE  B=L-SHADE+TMI  C=NL-SHADE  D=NL-SHADE+TMI')
    print('Wilcoxon: * p<0.05  ** p<0.01  ns not significant')
    print('TMI effect: + better  - worse  ~ no sig. difference')
    print('-' * 85)
    print(header_fmt.format('F#', 'A(mean)', 'B(mean)', 'C(mean)', 'D(mean)',
                             'p(B/A)', 'p(D/C)', 'B/A', 'D/C'))
    print('-' * 85)

    total_wins_B = 0
    total_wins_D = 0

    for fn in func_nums:
        if fn not in results_by_func:
            continue
        data = results_by_func[fn]

        err_A = np.array([r['final_error'] for r in data if r['variant'] == 'A'])
        err_B = np.array([r['final_error'] for r in data if r['variant'] == 'B'])
        err_C = np.array([r['final_error'] for r in data if r['variant'] == 'C'])
        err_D = np.array([r['final_error'] for r in data if r['variant'] == 'D'])

        mean_A = err_A.mean() if len(err_A) else float('nan')
        mean_B = err_B.mean() if len(err_B) else float('nan')
        mean_C = err_C.mean() if len(err_C) else float('nan')
        mean_D = err_D.mean() if len(err_D) else float('nan')

        _, p_BA = _wilcoxon_p(err_B, err_A) if (len(err_A) and len(err_B)) else (0, 1.0)
        _, p_DC = _wilcoxon_p(err_D, err_C) if (len(err_C) and len(err_D)) else (0, 1.0)

        sig_BA = _significance_marker(p_BA)
        sig_DC = _significance_marker(p_DC)
        eff_BA = _improvement_marker(mean_B, mean_A, p_BA)
        eff_DC = _improvement_marker(mean_D, mean_C, p_DC)

        if eff_BA == '+':
            total_wins_B += 1
        if eff_DC == '+':
            total_wins_D += 1

        tag = f'{sig_BA}{eff_BA}'
        tag2 = f'{sig_DC}{eff_DC}'
        print(row_fmt.format(fn, mean_A, mean_B, mean_C, mean_D,
                             p_BA, p_DC, tag, tag2))

    print('-' * 85)
    print(f'TMI wins (B>A): {total_wins_B}/{len(func_nums)}    '
          f'TMI wins (D>C): {total_wins_D}/{len(func_nums)}')
    print('=' * 85)


def _print_injection_stats(results_all: list, func_nums: list):
    """Print average injection statistics for B and D variants."""
    print()
    print('Injection Statistics')
    print('-' * 55)
    print(f'{"F#":>4}  {"B:injections":>12}  {"B:inj_FEs":>10}  '
          f'{"D:injections":>12}  {"D:inj_FEs":>10}')
    print('-' * 55)
    by_func = {}
    for r in results_all:
        by_func.setdefault(r['func_num'], []).append(r)
    for fn in func_nums:
        if fn not in by_func:
            continue
        rows = by_func[fn]
        b_inj = [r['injection_count'] for r in rows if r['variant'] == 'B']
        b_fes = [r['injection_fes'] for r in rows if r['variant'] == 'B']
        d_inj = [r['injection_count'] for r in rows if r['variant'] == 'D']
        d_fes = [r['injection_fes'] for r in rows if r['variant'] == 'D']
        print(f'{fn:>4}  {np.mean(b_inj) if b_inj else 0:>12.1f}  '
              f'{np.mean(b_fes) if b_fes else 0:>10.1f}  '
              f'{np.mean(d_inj) if d_inj else 0:>12.1f}  '
              f'{np.mean(d_fes) if d_fes else 0:>10.1f}')
    print('-' * 55)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_csv(results_all: list, out_path: Path):
    fieldnames = ['variant', 'func_num', 'dim', 'seed', 'final_error',
                  'elapsed_s', 'injection_count', 'injection_fes', 'saddles_archived']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_all)
    print(f'\nRaw results saved to {out_path}')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TMI CEC 2022 Benchmark')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of independent runs per variant×function (default: 30)')
    parser.add_argument('--funcs', type=int, nargs='+', default=list(range(1, 13)),
                        help='Function numbers to benchmark (default: 1-12)')
    parser.add_argument('--dim', type=int, default=10,
                        help='Problem dimensionality (default: 10)')
    parser.add_argument('--max-fe', type=int, default=None,
                        help='Max function evaluations (default: 200000 for D=10)')
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['A', 'B', 'C', 'D'],
                        help='Variants to run: A B C D (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: cpu_count - 1)')
    parser.add_argument('--out', type=str, default='results/tmi_cec2022.csv',
                        help='Output CSV path')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable multiprocessing (useful for debugging)')
    args = parser.parse_args()

    dim = args.dim
    max_fe = args.max_fe if args.max_fe else (200_000 if dim <= 10 else 1_000_000)
    func_nums = sorted(set(args.funcs))
    variants = [v.upper() for v in args.variants]
    n_seeds = args.seeds
    n_workers = args.workers if args.workers else max(1, cpu_count() - 1)

    print('=' * 60)
    print('TMI - Topological Manifold Injection Benchmark')
    print('=' * 60)
    print(f'Functions : F{func_nums[0]}–F{func_nums[-1]}  (D={dim})')
    print(f'Budget    : {max_fe:,} FEs per run')
    print(f'Runs      : {n_seeds} seeds × {len(variants)} variants × {len(func_nums)} functions')
    total_runs = n_seeds * len(variants) * len(func_nums)
    print(f'Total runs: {total_runs}')
    print(f'Workers   : {n_workers if not args.no_parallel else 1}')
    print('=' * 60)

    # Build task list
    tasks = [
        (variant, fn, dim, seed, max_fe)
        for variant in variants
        for fn in func_nums
        for seed in range(n_seeds)
    ]

    # Execute
    t_start = time.perf_counter()
    if args.no_parallel or n_workers == 1:
        results_all = []
        for i, task in enumerate(tasks):
            r = _run_one(task)
            results_all.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                pct = 100 * (i + 1) / len(tasks)
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (len(tasks) - i - 1)
                print(f'  [{pct:5.1f}%] {i+1}/{len(tasks)} done  '
                      f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s  '
                      f'F{r["func_num"]}/{r["variant"]} err={r["final_error"]:.3e}')
    else:
        with Pool(processes=n_workers) as pool:
            results_all = []
            for i, r in enumerate(pool.imap_unordered(_run_one, tasks)):
                results_all.append(r)
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    pct = 100 * (i + 1) / len(tasks)
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / (i + 1) * (len(tasks) - i - 1)
                    print(f'  [{pct:5.1f}%] {i+1}/{len(tasks)} done  '
                          f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

    total_elapsed = time.perf_counter() - t_start
    print(f'\nAll runs complete in {total_elapsed:.1f}s')

    # Organise by function
    results_by_func: dict = {}
    for r in results_all:
        results_by_func.setdefault(r['func_num'], []).append(r)

    # Print summary table
    _print_summary_table(results_by_func, func_nums, n_seeds)

    # Print injection statistics
    if any(v in variants for v in ['B', 'D']):
        _print_injection_stats(results_all, func_nums)

    # Save CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(results_all, out_path)


if __name__ == '__main__':
    main()

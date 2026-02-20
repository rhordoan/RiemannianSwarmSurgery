"""
ORC-SHADE vs NL-SHADE Benchmark on CEC 2022.

Compares:
  A  - NL-SHADE         (SOTA baseline, Stanovov et al. CEC 2022)
  B  - ORC-SHADE        (curvature-modulated, this work)

Protocol
--------
  - CEC 2022 benchmark suite, F1-F12
  - D = 10, max_FE = 200,000 (competition standard)
  - 30 independent runs per (variant, function)
  - Wilcoxon signed-rank test: B vs A (two-sided, alpha = 0.05)

Usage
-----
  python benchmarks/run_orc_shade_benchmark.py              # full 30-seed run
  python benchmarks/run_orc_shade_benchmark.py --seeds 5    # quick test
  python benchmarks/run_orc_shade_benchmark.py --funcs 4 8  # subset of funcs
  python benchmarks/run_orc_shade_benchmark.py --dim 20     # higher dimension
  python benchmarks/run_orc_shade_benchmark.py --no_parallel  # single process
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import wilcoxon
import opfunu
warnings.filterwarnings('ignore')

from src.orc_shade import ORCSHADE
from benchmarks.nlshade import NLSHADE


# ---------------------------------------------------------------------------
# Problem wrapper
# ---------------------------------------------------------------------------

class CEC2022Problem:
    """Thin wrapper: returns error = max(0, f(x) - f_bias)."""

    def __init__(self, func_num, dim):
        cls = getattr(opfunu.cec_based, 'F{}2022'.format(func_num))
        self._prob = cls(ndim=dim)
        self.bounds = [-100.0, 100.0]
        self.f_bias = self._prob.f_bias

    def evaluate(self, x):
        return max(0.0, float(self._prob.evaluate(x)) - self.f_bias)

    def __repr__(self):
        return 'CEC2022-F{}'.format(self.func_num)


# ---------------------------------------------------------------------------
# Single-run worker (top-level for multiprocessing)
# ---------------------------------------------------------------------------

def _run_one(args):
    variant, func_num, dim, seed, max_fe = args
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings('ignore')
    from src.orc_shade import ORCSHADE
    from benchmarks.nlshade import NLSHADE
    import opfunu

    class _Prob:
        def __init__(self, fn, d):
            cls = getattr(opfunu.cec_based, 'F{}2022'.format(fn))
            self._prob = cls(ndim=d)
            self.bounds = [-100.0, 100.0]
            self.f_bias = self._prob.f_bias
        def evaluate(self, x):
            return max(0.0, float(self._prob.evaluate(x)) - self.f_bias)

    t0 = time.perf_counter()
    problem = _Prob(func_num, dim)

    stats = {}
    if variant == 'NL-SHADE':
        opt = NLSHADE(problem, dim, max_fe=max_fe)
        opt.run()
        best_fit = opt.best_fitness
    else:
        opt = ORCSHADE(problem, dim, max_fe=max_fe, pop_schedule='nonlinear')
        opt.run()
        best_fit = opt.best_fitness
        stats = opt.get_run_stats()

    elapsed = time.perf_counter() - t0
    return {
        'variant': variant,
        'func_num': func_num,
        'seed': seed,
        'best_fit': float(best_fit),
        'elapsed': elapsed,
        **stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--funcs', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--max_fe', type=int, default=200_000)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    parser.add_argument('--no_parallel', action='store_true')
    args = parser.parse_args()

    workers = 1 if args.no_parallel else args.workers
    print('ORC-SHADE vs NL-SHADE  |  D={}  max_FE={:,}  seeds={}  funcs={}'.format(
        args.dim, args.max_fe, args.seeds, args.funcs))
    print('Workers: {}'.format(workers), flush=True)
    print()

    VARIANTS = ['NL-SHADE', 'ORC-SHADE']
    tasks = [
        (v, f, args.dim, seed, args.max_fe)
        for v in VARIANTS
        for f in args.funcs
        for seed in range(args.seeds)
    ]

    results = {v: {f: [] for f in args.funcs} for v in VARIANTS}
    orc_stats = {f: {'explore_pct': [], 'mean_kappa': []} for f in args.funcs}

    total = len(tasks)
    report_every = max(1, total // 20)

    if workers > 1:
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                results[r['variant']][r['func_num']].append(r['best_fit'])
                if r['variant'] == 'ORC-SHADE' and 'explore_pct' in r:
                    orc_stats[r['func_num']]['explore_pct'].append(r['explore_pct'])
                    orc_stats[r['func_num']]['mean_kappa'].append(r['mean_kappa'])
                if i % report_every == 0 or i == total:
                    print('  [{}/{}] done'.format(i, total), flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            r = _run_one(task)
            results[r['variant']][r['func_num']].append(r['best_fit'])
            if r['variant'] == 'ORC-SHADE' and 'explore_pct' in r:
                orc_stats[r['func_num']]['explore_pct'].append(r['explore_pct'])
                orc_stats[r['func_num']]['mean_kappa'].append(r['mean_kappa'])
            if i % report_every == 0 or i == total:
                print('  [{}/{}] done'.format(i, total), flush=True)

    print()
    print('='*75)
    header = '{:<5} | {:>17} | {:>17} | {:>6} | {:>8}'.format(
        'Func', 'NL-SHADE (mean+/-std)', 'ORC-SHADE (mean+/-std)', 'Win?', 'p-value')
    print(header)
    print('-'*75)

    orc_wins = nl_wins = ties = 0
    for f in args.funcs:
        nl_arr = np.array(results['NL-SHADE'][f])
        orc_arr = np.array(results['ORC-SHADE'][f])

        try:
            if np.allclose(nl_arr, orc_arr):
                p = 1.0
            else:
                _, p = wilcoxon(nl_arr, orc_arr, alternative='two-sided')
        except Exception:
            p = float('nan')

        if np.isfinite(p) and p < 0.05:
            if orc_arr.mean() < nl_arr.mean():
                win = 'ORC+'
                orc_wins += 1
            else:
                win = 'NL+'
                nl_wins += 1
        else:
            win = 'TIE'
            ties += 1

        row = 'F{:<4} | {:>8.2e}+/-{:.1e} | {:>8.2e}+/-{:.1e} | {:>6} | {:>8.4f}'.format(
            f,
            nl_arr.mean(), nl_arr.std(),
            orc_arr.mean(), orc_arr.std(),
            win, p if np.isfinite(p) else float('nan'))
        print(row)

    print('='*75)
    print('ORC-SHADE wins: {}  NL-SHADE wins: {}  Ties: {}'.format(orc_wins, nl_wins, ties))

    print()
    print('ORC-SHADE exploration diagnostics:')
    print('{:<5} | {:>10} | {:>11}'.format('Func', 'explore %', 'mean kappa'))
    print('-'*35)
    for f in args.funcs:
        ep = orc_stats[f]['explore_pct']
        mk = orc_stats[f]['mean_kappa']
        if ep:
            print('F{:<4} | {:>10.1f} | {:>11.3f}'.format(f, np.mean(ep), np.mean(mk)))


if __name__ == '__main__':
    main()

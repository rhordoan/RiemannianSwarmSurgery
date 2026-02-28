#!/usr/bin/env python3
"""
ORC-BO Expensive Optimization Benchmark
========================================
Compares ORC-BO (topology-aware BO) against EGO, CMA-ES, and NL-SHADE
on CEC 2022 F1-F12 under expensive evaluation budgets.

Expensive budgets (much smaller than competition spec):
  D=10 -> 1,000 FEs (100*D)   |   D=20 -> 2,000 FEs (100*D)

Usage
-----
  # Quick smoke test (3 seeds, D=10, 2 functions)
  python3 benchmarks/run_orc_bo_benchmark.py --seeds 3 --dims 10 --funcs 1 4

  # Full benchmark (30 seeds, D=10 + D=20, all functions, 160 cores)
  python3 benchmarks/run_orc_bo_benchmark.py --seeds 30 --workers 160

  # Resume after interruption
  python3 benchmarks/run_orc_bo_benchmark.py --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Problem wrapper (same as run_overnight.py)
# ---------------------------------------------------------------------------

def _make_problem(func_num: int, dim: int):
    """Build a CEC 2022 problem object (error = f(x) - f*)."""
    import opfunu
    warnings.filterwarnings("ignore")
    cls = getattr(opfunu.cec_based, f"F{func_num}2022")
    inner = cls(ndim=dim)
    f_bias = inner.f_bias

    class _Prob:
        bounds = [-100.0, 100.0]
        def evaluate(self, x):
            return max(0.0, float(inner.evaluate(x)) - f_bias)

    return _Prob()


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS: dict = {
    "ORC-BO": {
        "builder": "_build_orc_bo",
    },
    "EGO": {
        "builder": "_build_ego",
    },
    "CMA-ES": {
        "builder": "_build_cmaes",
    },
    "NL-SHADE": {
        "builder": "_build_nlshade",
    },
    "Random": {
        "builder": "_build_random",
    },
}

MAIN_VARIANTS = ["ORC-BO", "EGO", "CMA-ES", "NL-SHADE"]


def _build_orc_bo(problem, dim, budget, seed):
    from src.orc_bo import ORCBO
    return ORCBO(problem, dim, budget, seed=seed)


def _build_ego(problem, dim, budget, seed):
    from benchmarks.baselines_bo import EGO
    return EGO(problem, dim, budget, seed=seed)


def _build_cmaes(problem, dim, budget, seed):
    """CMA-ES with limited budget. Returns a wrapper with .run() interface."""
    return _CMAESWrapper(problem, dim, budget, seed)


def _build_nlshade(problem, dim, budget, seed):
    """NL-SHADE with the same limited budget (for reference)."""
    from benchmarks.nlshade import NLSHADE
    return NLSHADE(problem, dim, max_fe=budget)


def _build_random(problem, dim, budget, seed):
    from benchmarks.baselines_bo import RandomSearch
    return RandomSearch(problem, dim, budget, seed=seed)


class _CMAESWrapper:
    """Thin wrapper around cma.fmin to match the .run() / .convergence_log interface."""

    def __init__(self, problem, dim, budget, seed):
        self.problem = problem
        self.dim = dim
        self.budget = budget
        self.seed = seed
        self.lb = float(problem.bounds[0])
        self.ub = float(problem.bounds[1])
        self.best_fitness = np.inf
        self.best_solution = None
        self.convergence_log = []
        self.fe_count = 0

    def run(self):
        import cma
        np.random.seed(self.seed)
        x0 = np.random.uniform(self.lb, self.ub, self.dim)
        sigma0 = (self.ub - self.lb) / 4.0
        opts = {
            'maxfevals': self.budget,
            'bounds': [[self.lb] * self.dim, [self.ub] * self.dim],
            'seed': self.seed,
            'verbose': -9,
            'tolfun': 0,
            'tolx': 0,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and self.fe_count < self.budget:
            solutions = es.ask()
            fitnesses = []
            for x in solutions:
                if self.fe_count >= self.budget:
                    fitnesses.append(1e30)
                    continue
                y = float(self.problem.evaluate(x))
                self.fe_count += 1
                fitnesses.append(y)
                if y < self.best_fitness:
                    self.best_fitness = y
                    self.best_solution = np.array(x).copy()
            es.tell(solutions, fitnesses)
            self.convergence_log.append((self.fe_count, self.best_fitness))

        if self.best_solution is None:
            self.best_solution = np.array(es.result.xbest)
            self.best_fitness = float(es.result.fbest)

        return self.best_solution, self.best_fitness

    def get_run_stats(self):
        return {'best_fitness': self.best_fitness, 'fe_count': self.fe_count}


# ---------------------------------------------------------------------------
# Convergence milestones
# ---------------------------------------------------------------------------

def _sample_convergence(opt, max_fe, milestones=(0.1, 0.25, 0.5, 0.75, 1.0)):
    log = getattr(opt, "convergence_log", None)
    if not log:
        return {}
    results = {}
    for m in milestones:
        target_fe = int(m * max_fe)
        best = None
        for fe, bf in log:
            if fe <= target_fe:
                best = bf
        if best is not None:
            results[f"err_at_{int(m*100)}pct"] = best
    return results


# ---------------------------------------------------------------------------
# Single-run worker
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> dict:
    variant, func_num, dim, seed, budget = args

    import sys, os, warnings
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    warnings.filterwarnings("ignore")

    # Prevent BLAS thread contention across parallel workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import numpy as np
    np.random.seed(seed)

    t0 = __import__("time").perf_counter()
    problem = _make_problem(func_num, dim)

    builder_name = VARIANTS[variant]["builder"]
    builder = globals()[builder_name]
    opt = builder(problem, dim, budget, seed)

    # For NL-SHADE, wrap step to add convergence logging
    if not hasattr(opt, "convergence_log") or opt.convergence_log is None:
        opt.convergence_log = []
        if hasattr(opt, 'step'):
            _orig_step = opt.step
            def _logged_step():
                ret = _orig_step()
                opt.convergence_log.append((opt.fe_count, opt.best_fitness))
                return ret
            opt.step = _logged_step

    opt.run()
    elapsed = __import__("time").perf_counter() - t0

    row = {
        "variant": variant,
        "dim": dim,
        "func": func_num,
        "seed": seed,
        "budget": budget,
        "best_fit": float(opt.best_fitness),
        "elapsed_s": round(elapsed, 2),
    }
    row.update(_sample_convergence(opt, budget))

    if hasattr(opt, "get_run_stats"):
        st = opt.get_run_stats()
        row["n_basins"] = st.get("n_basins_final", "")
        row["n_saddles"] = st.get("n_saddles_final", "")
        row["avg_basins"] = round(st.get("avg_basins", 0), 1) if st.get("avg_basins") else ""

    return row


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

FIXED_FIELDS = [
    "variant", "dim", "func", "seed", "budget",
    "best_fit", "elapsed_s",
    "n_basins", "n_saddles", "avg_basins",
    "err_at_10pct", "err_at_25pct", "err_at_50pct", "err_at_75pct", "err_at_100pct",
]


def _load_done(csv_path: Path) -> set:
    done: set = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["variant"], int(row["dim"]),
                       int(row["func"]), int(row["seed"])))
    return done


def _append_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    for k in FIXED_FIELDS:
        row.setdefault(k, "")
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIXED_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Live summary
# ---------------------------------------------------------------------------

def _print_summary(csv_path: Path, dims: list, funcs: list):
    from scipy.stats import wilcoxon
    from collections import defaultdict

    data: dict = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            k = (row["variant"], int(row["dim"]), int(row["func"]))
            data[k].append(float(row["best_fit"]))

    variants_found = sorted(set(v for v, _, _ in data.keys()))
    ref = "ORC-BO"

    for dim in dims:
        has = any(k[1] == dim for k in data)
        if not has:
            continue
        print(f"\n{'='*90}\nD = {dim}  (budget = {100*dim} FEs)\n{'='*90}")
        header_parts = [f"{'F':<5}"]
        for v in variants_found:
            header_parts.append(f"{v:>19}")
        hdr = " | ".join(header_parts)
        print(hdr)
        print("-" * len(hdr))

        wins = {v: 0 for v in variants_found}
        losses = {v: 0 for v in variants_found}

        for f in funcs:
            parts = [f"F{f:<4}"]
            ref_vals = np.array(data.get((ref, dim, f), []))
            for v in variants_found:
                vals = np.array(data.get((v, dim, f), []))
                if len(vals) == 0:
                    parts.append(f"{'---':>19}")
                    continue
                parts.append(f"{vals.mean():>9.2e}+/-{vals.std():.1e}")

                if v != ref and len(ref_vals) and len(vals):
                    n = min(len(ref_vals), len(vals))
                    a, b = ref_vals[:n], vals[:n]
                    try:
                        p = wilcoxon(a, b, alternative="two-sided").pvalue if not np.allclose(a, b) else 1.0
                    except Exception:
                        p = 1.0
                    if p < 0.05 and a.mean() < b.mean():
                        wins[ref] += 1
                        losses[v] += 1
                    elif p < 0.05 and b.mean() < a.mean():
                        wins[v] += 1
                        losses[ref] += 1

            print(" | ".join(parts))

        print("-" * len(hdr))
        print("ORC-BO W/L vs others: ", end="")
        for v in variants_found:
            if v == ref:
                continue
            print(f"{v}: {wins.get(ref, 0)}W/{losses.get(ref, 0)}L  ", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ORC-BO Expensive Optimization Benchmark")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--funcs", type=int, nargs="+",
                        default=list(range(1, 13)))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budget_mult", type=int, default=100,
                        help="Budget = budget_mult * D (default 100)")
    parser.add_argument("--workers", type=int,
                        default=max(1, cpu_count() - 2))
    parser.add_argument("--out", default="results/orc_bo_cec2022.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to run")
    parser.add_argument("--no_parallel", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_variants = args.variants if args.variants else MAIN_VARIANTS
    done = _load_done(out_path) if args.resume else set()

    tasks = []
    for dim in args.dims:
        budget = args.budget_mult * dim
        for v in run_variants:
            if v not in VARIANTS:
                print(f"Warning: unknown variant '{v}', skipping.")
                continue
            for f in args.funcs:
                for seed in range(args.seeds):
                    if (v, dim, f, seed) not in done:
                        tasks.append((v, f, dim, seed, budget))

    total = len(tasks)
    if not total:
        print("All runs complete.")
        _print_summary(out_path, args.dims, args.funcs)
        return

    workers = 1 if args.no_parallel else args.workers
    print(
        f"\nORC-BO Expensive Optimization Benchmark\n"
        f"  Variants : {run_variants}\n"
        f"  Dims     : {args.dims}  "
        f"(budget = {args.budget_mult}*D)\n"
        f"  Seeds    : {args.seeds}  |  Funcs: {args.funcs}\n"
        f"  Workers  : {workers}  |  Tasks: {total}\n"
        f"  Output   : {out_path}\n",
        flush=True,
    )

    completed = 0
    t_start = time.perf_counter()

    def _handle(row: dict):
        nonlocal completed
        _append_row(out_path, row)
        completed += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / completed * (total - completed) if completed else 0
        pct = 100.0 * completed / total
        basins_str = (f"  basins={row.get('n_basins', '')}"
                      if row.get('n_basins', '') != '' else '')
        print(
            f"  [{completed:>{len(str(total))}}/{total}] "
            f"D={row['dim']} F{row['func']} {row['variant']:<12} "
            f"err={row['best_fit']:.3e}{basins_str}  "
            f"{pct:.1f}%  ETA {eta/60:.1f}m  "
            f"({row['elapsed_s']:.1f}s)",
            flush=True,
        )

    if workers == 1:
        for task in tasks:
            _handle(_run_one(task))
    else:
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_run_one, tasks):
                _handle(row)

    print("\n\nAll runs complete.\n")
    _print_summary(out_path, args.dims, args.funcs)

    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        _print_summary(out_path, args.dims, args.funcs)
    summary_path = out_path.with_suffix(".summary.txt")
    summary_path.write_text(buf.getvalue())
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    freeze_support()
    main()

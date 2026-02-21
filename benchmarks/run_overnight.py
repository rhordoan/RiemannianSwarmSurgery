#!/usr/bin/env python3
"""
ORC-SHADE Overnight Benchmark  --  CEC 2022
============================================
Compares NL-SHADE vs ORC-SHADE (and optional ablations) on F1-F12.
Results are written incrementally to CSV; interrupted runs resume cleanly.

Standard FE budgets (CEC 2022 competition spec):
  D=10 -> 200,000 FEs   |   D=20 -> 1,000,000 FEs

Usage
-----
  # Full overnight: D=10 + D=20, 30 seeds, ablations included
  python benchmarks/run_overnight.py --ablation

  # Quick test (3 seeds, D=10 only, 30k FE)
  python benchmarks/run_overnight.py --seeds 3 --dims 10 --budget_d10 30000

  # Resume after interruption
  python benchmarks/run_overnight.py --resume

  # Specific worker count for M2 Ultra (default = cpu_count - 2)
  python benchmarks/run_overnight.py --workers 20
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

# Make the project root importable in both main and worker processes
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Problem wrapper
# ---------------------------------------------------------------------------

def _make_problem(func_num: int, dim: int):
    """Build a CEC 2022 problem object (error = f(x) - f*)."""
    import opfunu, warnings
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
    "NL-SHADE": {
        "cls": "NLSHADE",
        "kwargs": {},
    },
    # ORC-SHADE v2: full algorithm (kappa_min=0.15, p_elite=0.2 are defaults)
    "ORC-SHADE": {
        "cls": "ORCSHADE",
        "kwargs": {},
    },
    # Ablation A: no kappa threshold (fires on all negative curvature incl. noise)
    "ORC-SHADE[km=0]": {
        "cls": "ORCSHADE",
        "kwargs": {"kappa_min": 0.0},
    },
    # Ablation B: no elite protection (all agents can explore)
    "ORC-SHADE[pe=0]": {
        "cls": "ORCSHADE",
        "kwargs": {"p_elite": 0.0},
    },
    # Ablation C: ORC disabled (kappa_scale huge => alpha=0 always => NL-SHADE clone)
    "ORC-SHADE[no-orc]": {
        "cls": "ORCSHADE",
        "kwargs": {"kappa_scale": 999.0},
    },
}

MAIN_VARIANTS = ["NL-SHADE", "ORC-SHADE"]


def _build_opt(name: str, problem, dim: int, max_fe: int):
    cfg = VARIANTS[name]
    if cfg["cls"] == "NLSHADE":
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from benchmarks.nlshade import NLSHADE
        return NLSHADE(problem, dim, max_fe=max_fe)
    else:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from src.orc_shade import ORCSHADE
        return ORCSHADE(problem, dim, max_fe=max_fe, **cfg["kwargs"])


# ---------------------------------------------------------------------------
# Convergence milestone helper
# ---------------------------------------------------------------------------

def _sample_convergence(opt, max_fe: int, milestones=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """
    Sample best fitness at key FE checkpoints.
    Works for both NLSHADE (no convergence_log) and ORCSHADE.
    """
    log = getattr(opt, "convergence_log", None)
    if not log:
        return {}
    # log is [(fe_count, best_fit), ...]
    results = {}
    for m in milestones:
        target_fe = int(m * max_fe)
        # Find last entry with fe_count <= target_fe
        best = None
        for fe, bf in log:
            if fe <= target_fe:
                best = bf
        if best is not None:
            results[f"err_at_{int(m*100)}pct"] = best
    return results


# ---------------------------------------------------------------------------
# Single-run worker (top-level, safe for multiprocessing spawn on macOS)
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> dict:
    variant, func_num, dim, seed, max_fe = args

    # Ensure project root is importable inside spawned worker (macOS / spawn)
    import sys, os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(seed)

    t0 = __import__("time").perf_counter()
    problem = _make_problem(func_num, dim)
    opt = _build_opt(variant, problem, dim, max_fe)

    # Run with convergence logging for NLSHADE (wrap step to track)
    if not hasattr(opt, "convergence_log"):
        opt.convergence_log = [(opt.fe_count, opt.best_fitness)]
        _orig_step = opt.step
        def _logged_step():
            ret = _orig_step()
            opt.convergence_log.append((opt.fe_count, opt.best_fitness))
            return ret
        opt.step = _logged_step

    opt.run()
    elapsed = __import__("time").perf_counter() - t0

    row: dict = {
        "variant": variant,
        "dim": dim,
        "func": func_num,
        "seed": seed,
        "best_fit": float(opt.best_fitness),
        "elapsed_s": round(elapsed, 2),
        "explore_pct": "",
        "mean_kappa": "",
        "effective_threshold": "",
        "n_explore_agents": ""}
    # Convergence milestones
    row.update(_sample_convergence(opt, max_fe))

    if hasattr(opt, "get_run_stats"):
        st = opt.get_run_stats()
        row["explore_pct"] = round(st.get("explore_pct", 0.0), 2)
        row["mean_kappa"] = round(st.get("mean_kappa", 0.0), 4)
        row["effective_threshold"] = round(st.get("kappa_min", 0.0), 4)
        row["n_explore_agents"] = st.get("n_explore_agents", 0)
        row["M_CR_exploit"] = round(st.get("M_CR_exploit", 0.5), 3)
        row["M_CR_explore"] = round(st.get("M_CR_explore", 0.8), 3)

    return row


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

FIXED_FIELDS = [
    "variant", "dim", "func", "seed",
    "best_fit", "elapsed_s",
    "explore_pct", "mean_kappa", "effective_threshold", "n_explore_agents",
    "err_at_10pct", "err_at_25pct", "err_at_50pct", "err_at_75pct", "err_at_100pct",
]


def _load_done(csv_path: Path) -> set:
    done: set = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["variant"], int(row["dim"]), int(row["func"]), int(row["seed"])))
    return done


def _append_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    # Ensure all fixed fields exist
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

def _print_summary(csv_path: Path, dims: list, funcs: list,
                   var_a: str = "NL-SHADE", var_b: str = "ORC-SHADE"):
    from scipy.stats import wilcoxon, friedmanchisquare
    from collections import defaultdict
    data: dict = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            k = (row["variant"], int(row["dim"]), int(row["func"]))
            data[k].append(float(row["best_fit"]))

    for dim in dims:
        has = any(k[1] == dim for k in data)
        if not has:
            continue
        print(f"\n{'='*72}\nD = {dim}\n{'='*72}")
        hdr = f"{'F':<5} | {'NL-SHADE':>19} | {'ORC-SHADE':>19} | {'Win':>5} | {'p':>8}"
        print(hdr)
        print("-" * len(hdr))
        orc_w = nl_w = ties = 0
        for f in funcs:
            a = np.array(data.get((var_a, dim, f), []))
            b = np.array(data.get((var_b, dim, f), []))
            if not len(a) or not len(b):
                continue
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            try:
                p = wilcoxon(a, b, alternative="two-sided").pvalue if not np.allclose(a, b) else 1.0
            except Exception:
                p = float("nan")
            sig = np.isfinite(p) and p < 0.05
            if sig and b.mean() < a.mean():
                win, orc_w = "ORC+", orc_w + 1
            elif sig and a.mean() < b.mean():
                win, nl_w = "NL+ ", nl_w + 1
            else:
                win, ties = "TIE ", ties + 1
            print(f"F{f:<4} | {a.mean():>9.2e} +/-{a.std():.1e} | {b.mean():>9.2e} +/-{b.std():.1e} | {win:>5} | {p:>8.4f}")
        print("-" * len(hdr))
        print(f"ORC wins: {orc_w}  NL wins: {nl_w}  Ties: {ties}")

        # Friedman test across all 12 functions
        try:
            groups = [data.get((v, dim, f), []) for v in [var_a, var_b] for f in funcs if data.get((v, dim, f))]
            if len(groups) >= 3:
                stat, pf = friedmanchisquare(*groups)
                print(f"Friedman chi2={stat:.2f}  p={pf:.4f}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--funcs", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budget_d10", type=int, default=200_000,
                        help="FE budget for D=10 (default: 200000)")
    parser.add_argument("--budget_d20", type=int, default=1_000_000,
                        help="FE budget for D=20 (default: 1000000)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--out", default="results/orc_shade_cec2022.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Include 5 ablation variants (adds ~4x runtime)")
    parser.add_argument("--no_parallel", action="store_true")
    args = parser.parse_args()

    budgets = {10: args.budget_d10, 20: args.budget_d20}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_variants = list(VARIANTS.keys()) if args.ablation else MAIN_VARIANTS
    done = _load_done(out_path) if args.resume else set()

    tasks: list[tuple] = []
    for dim in args.dims:
        max_fe = budgets.get(dim, 200_000)
        for v in run_variants:
            for f in args.funcs:
                for seed in range(args.seeds):
                    if (v, dim, f, seed) not in done:
                        tasks.append((v, f, dim, seed, max_fe))

    total = len(tasks)
    if not total:
        print("All runs complete. Run --resume with additional seeds/dims to extend.")
        _print_summary(out_path, args.dims, args.funcs)
        return

    workers = 1 if args.no_parallel else args.workers
    print(
        f"\nORC-SHADE Overnight Benchmark\n"
        f"  Variants : {run_variants}\n"
        f"  Dims     : {args.dims}  (budgets: D10={args.budget_d10:}  D20={args.budget_d20:})\n"
        f"  Seeds    : {args.seeds}  |  Funcs: F1-F12\n"
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
        ep = row.get("explore_pct", "")
        ep_str = f"  expl={ep:.0f}%" if ep != "" else ""
        print(
            f"  [{completed:>{len(str(total))}}/{total}] "
            f"D={row['dim']} F{row['func']} {row['variant']:<28} "
            f"err={row['best_fit']:.3e}{ep_str}  "
            f"{pct:.1f}%  ETA {eta/3600:.2f}h",
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

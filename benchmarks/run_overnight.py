#!/usr/bin/env python3
"""
ORC-SHADE Overnight Benchmark
CEC 2022 | D=10 and D=20 | 30 seeds | NL-SHADE vs ORC-SHADE (+ ablations)

Designed for M2 Ultra / multi-core machines. Results are saved incrementally
to CSV so a crashed run can be resumed without losing completed work.

Usage
-----
# Full overnight run (30 seeds, D=10+D=20, ablations included)
python benchmarks/run_overnight.py

# Quick sanity check (3 seeds, D=10 only)
python benchmarks/run_overnight.py --seeds 3 --dims 10 --budget 30000

# Resume after interruption (skips already-saved rows)
python benchmarks/run_overnight.py --resume

# Specify number of parallel workers (default: all cores - 2)
python benchmarks/run_overnight.py --workers 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Problem wrapper
# ---------------------------------------------------------------------------

def _make_problem(func_num: int, dim: int):
    import opfunu
    warnings.filterwarnings("ignore")
    cls = getattr(opfunu.cec_based, f"F{func_num}2022")
    prob = cls(ndim=dim)

    class _Prob:
        bounds = [-100.0, 100.0]
        f_bias = prob.f_bias
        _inner = prob

        def evaluate(self, x: np.ndarray) -> float:
            return max(0.0, float(self._inner.evaluate(x)) - self.f_bias)

    return _Prob()


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------

VARIANTS: dict = {
    "NL-SHADE": {
        "cls": "NLSHADE",
        "kwargs": {},
    },
    "ORC-SHADE": {
        "cls": "ORCSHADE",
        "kwargs": {
            "orc_threshold": -0.30,
            "max_explore_frac": 0.25,
            "orc_lambda": 0.5,
        },
    },
    # Ablation: more aggressive (tighter threshold = explores more often)
    "ORC-SHADE[tau=-0.10]": {
        "cls": "ORCSHADE",
        "kwargs": {
            "orc_threshold": -0.10,
            "max_explore_frac": 0.25,
            "orc_lambda": 0.5,
        },
    },
    # Ablation: more conservative (threshold = -0.50)
    "ORC-SHADE[tau=-0.50]": {
        "cls": "ORCSHADE",
        "kwargs": {
            "orc_threshold": -0.50,
            "max_explore_frac": 0.25,
            "orc_lambda": 0.5,
        },
    },
    # Ablation: no explore cap (frac=1.0, essentially explore all saddles)
    "ORC-SHADE[frac=0.40]": {
        "cls": "ORCSHADE",
        "kwargs": {
            "orc_threshold": -0.30,
            "max_explore_frac": 0.40,
            "orc_lambda": 0.5,
        },
    },
    # Ablation: no ORC (pure NL-SHADE reimplemented via ORCSHADE with tau=0)
    "ORC-SHADE[no-orc]": {
        "cls": "ORCSHADE",
        "kwargs": {
            "orc_threshold": 0.0,         # Never triggers (all kappa <= 0)
            "max_explore_frac": 0.0,      # Explore fraction = 0
            "orc_lambda": 0.5,
        },
    },
}


def _build_variant(name: str, problem, dim: int, max_fe: int):
    cfg = VARIANTS[name]
    if cfg["cls"] == "NLSHADE":
        from benchmarks.nlshade import NLSHADE
        return NLSHADE(problem, dim, max_fe=max_fe)
    else:
        from src.orc_shade import ORCSHADE
        return ORCSHADE(problem, dim, max_fe=max_fe, **cfg["kwargs"])


# ---------------------------------------------------------------------------
# Single-run worker (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> dict:
    variant, func_num, dim, seed, max_fe = args
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings("ignore")

    t0 = time.perf_counter()
    problem = _make_problem(func_num, dim)
    opt = _build_variant(variant, problem, dim, max_fe)
    opt.run()
    elapsed = time.perf_counter() - t0

    row = {
        "variant": variant,
        "dim": dim,
        "func": func_num,
        "seed": seed,
        "best_fit": float(opt.best_fitness),
        "elapsed_s": round(elapsed, 2),
        "explore_pct": "",
        "mean_kappa": "",
        "effective_threshold": "",
        "n_explore_agents": "",
    }

    if hasattr(opt, "get_run_stats"):
        st = opt.get_run_stats()
        row["explore_pct"] = round(st.get("explore_pct", 0.0), 2)
        row["mean_kappa"] = round(st.get("mean_kappa", 0.0), 4)
        row["effective_threshold"] = round(st.get("effective_threshold", 0.0), 4)
        row["n_explore_agents"] = st.get("n_explore_agents", 0)

    return row


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "variant", "dim", "func", "seed",
    "best_fit", "elapsed_s",
    "explore_pct", "mean_kappa", "effective_threshold", "n_explore_agents",
]


def _load_existing(csv_path: Path) -> set:
    """Return set of (variant, dim, func, seed) already completed."""
    done: set = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["variant"], int(row["dim"]), int(row["func"]), int(row["seed"])))
    return done


def _append_row(csv_path: Path, row: dict):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Summary / statistics
# ---------------------------------------------------------------------------

def _print_summary(csv_path: Path, variant_a: str = "NL-SHADE", variant_b: str = "ORC-SHADE"):
    """Print a Wilcoxon-tested summary table from the CSV."""
    import csv as _csv
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))

    if not rows:
        print("No data yet.")
        return

    dims = sorted({int(r["dim"]) for r in rows})
    funcs = sorted({int(r["func"]) for r in rows})

    for dim in dims:
        print()
        print(f"=== D = {dim} ===")
        header = (
            f"{'F':<5} | {'NL-SHADE (mean)':>17} | {'ORC-SHADE (mean)':>17} "
            f"| {'Win':>6} | {'p':>8}"
        )
        print(header)
        print("-" * len(header))

        orc_w = nl_w = ties = 0
        for f in funcs:
            a_vals = [float(r["best_fit"]) for r in rows
                      if r["variant"] == variant_a and int(r["dim"]) == dim and int(r["func"]) == f]
            b_vals = [float(r["best_fit"]) for r in rows
                      if r["variant"] == variant_b and int(r["dim"]) == dim and int(r["func"]) == f]

            if not a_vals or not b_vals:
                continue

            a, b = np.array(a_vals), np.array(b_vals)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]

            try:
                if np.allclose(a, b):
                    p = 1.0
                else:
                    _, p = wilcoxon(a, b, alternative="two-sided")
            except Exception:
                p = float("nan")

            sig = p < 0.05 and np.isfinite(p)
            if sig and b.mean() < a.mean():
                win, orc_w = "ORC+", orc_w + 1
            elif sig and a.mean() < b.mean():
                win, nl_w = "NL+ ", nl_w + 1
            else:
                win, ties = "TIE ", ties + 1

            print(
                f"F{f:<4} | {a.mean():>8.2e}+/-{a.std():.1e} "
                f"| {b.mean():>8.2e}+/-{b.std():.1e} "
                f"| {win:>6} | {p:>8.4f}"
            )

        print("-" * len(header))
        print(f"ORC wins: {orc_w}  NL wins: {nl_w}  Ties: {ties}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ORC-SHADE overnight benchmark")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20],
                        help="Dimensions to benchmark (default: 10 20)")
    parser.add_argument("--funcs", type=int, nargs="+", default=list(range(1, 13)),
                        help="CEC 2022 function IDs (default: 1-12)")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of independent seeds (default: 30)")
    parser.add_argument("--budget", type=int, default=0,
                        help="Override FE budget for all dims (0 = use CEC 2022 standard)")
    parser.add_argument("--workers", type=int,
                        default=max(1, cpu_count() - 2),
                        help="Parallel workers (default: cpu_count - 2)")
    parser.add_argument("--out", type=str, default="results/orc_shade_cec2022.csv",
                        help="Output CSV path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (variant,dim,func,seed) combos")
    parser.add_argument("--ablation", action="store_true",
                        help="Include ablation variants (slower ??? 6 variants total)")
    parser.add_argument("--no_parallel", action="store_true",
                        help="Disable multiprocessing (for debugging)")
    args = parser.parse_args()

    # CEC 2022 standard FE budgets
    std_budget = {10: 200_000, 20: 1_000_000}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide which variants to run
    if args.ablation:
        run_variants = list(VARIANTS.keys())
    else:
        run_variants = ["NL-SHADE", "ORC-SHADE"]

    done = _load_existing(out_path) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} runs already completed, skipping them.")

    tasks: list[tuple] = []
    for dim in args.dims:
        max_fe = args.budget if args.budget > 0 else std_budget.get(dim, 200_000)
        for v in run_variants:
            for f in args.funcs:
                for seed in range(args.seeds):
                    key = (v, dim, f, seed)
                    if key not in done:
                        tasks.append((v, f, dim, seed, max_fe))

    total = len(tasks)
    if total == 0:
        print("All runs already complete. Run with --resume to add more seeds/dims.")
        _print_summary(out_path)
        return

    print(
        f"\nORC-SHADE Overnight Benchmark\n"
        f"  Variants : {run_variants}\n"
        f"  Dims     : {args.dims}\n"
        f"  Funcs    : F1-F12\n"
        f"  Seeds    : {args.seeds}\n"
        f"  Workers  : {1 if args.no_parallel else args.workers}\n"
        f"  Tasks    : {total}\n"
        f"  Output   : {out_path}\n"
    )

    completed = 0
    t_start = time.perf_counter()

    def _on_result(row: dict):
        nonlocal completed
        _append_row(out_path, row)
        completed += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / completed * (total - completed) if completed else 0
        pct = 100.0 * completed / total
        print(
            f"  [{completed:>{len(str(total))}}/{total}] "
            f"D={row['dim']} F{row['func']} {row['variant']:<24} "
            f"err={row['best_fit']:.3e}  "
            f"{pct:.1f}%  ETA {eta/3600:.1f}h",
            flush=True,
        )

    if args.no_parallel or args.workers == 1:
        for task in tasks:
            _on_result(_run_one(task))
    else:
        with Pool(args.workers) as pool:
            for row in pool.imap_unordered(_run_one, tasks):
                _on_result(row)

    print("\n\nAll runs complete. Summary:\n")
    _print_summary(out_path)

    # Also save a plain-text summary next to the CSV
    summary_path = out_path.with_suffix(".summary.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_summary(out_path)
    summary_path.write_text(buf.getvalue())
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    freeze_support()
    main()

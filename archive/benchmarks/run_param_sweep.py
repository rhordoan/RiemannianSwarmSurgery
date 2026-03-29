#!/usr/bin/env python3
"""
Quick parameter sweep for ORC-SHADE tuning.
Tests multiple configs on D=10, 5 seeds, CEC 2022 F1-F12.
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

warnings.filterwarnings("ignore")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


SWEEP_CONFIGS = {
    "NL-SHADE": {"cls": "NLSHADE", "kwargs": {}},
    "A:km=0,pe=.05,k=10": {"cls": "ORCSHADE", "kwargs": {"kappa_min": 0.0, "p_elite": 0.05, "k_max": 10}},
    "B:km=.05,pe=.05,k=10": {"cls": "ORCSHADE", "kwargs": {"kappa_min": 0.05, "p_elite": 0.05, "k_max": 10}},
    "C:km=0,pe=0,k=10": {"cls": "ORCSHADE", "kwargs": {"kappa_min": 0.0, "p_elite": 0.0, "k_max": 10}},
    "D:km=0,pe=.05,k=7": {"cls": "ORCSHADE", "kwargs": {"kappa_min": 0.0, "p_elite": 0.05, "k_max": 7}},
    "E:km=0,pe=.05,k=15": {"cls": "ORCSHADE", "kwargs": {"kappa_min": 0.0, "p_elite": 0.05, "k_max": 15}},
}


def _run_one(args):
    variant, func_num, dim, seed, max_fe = args
    import sys, os, warnings
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(seed)
    import opfunu

    cls_obj = getattr(opfunu.cec_based, f"F{func_num}2022")
    inner = cls_obj(ndim=dim)
    f_bias = inner.f_bias

    class _Prob:
        bounds = [-100.0, 100.0]
        def evaluate(self, x):
            return max(0.0, float(inner.evaluate(x)) - f_bias)

    problem = _Prob()
    cfg = SWEEP_CONFIGS[variant]

    t0 = time.perf_counter()
    if cfg["cls"] == "NLSHADE":
        from benchmarks.nlshade import NLSHADE
        opt = NLSHADE(problem, dim, max_fe=max_fe)
    else:
        from src.orc_shade import ORCSHADE
        opt = ORCSHADE(problem, dim, max_fe=max_fe, **cfg["kwargs"])
    opt.run()
    elapsed = time.perf_counter() - t0

    row = {
        "variant": variant,
        "func": func_num,
        "seed": seed,
        "best_fit": float(opt.best_fitness),
        "elapsed_s": round(elapsed, 2),
    }
    if hasattr(opt, "get_run_stats"):
        st = opt.get_run_stats()
        row["explore_pct"] = round(st.get("explore_pct", 0.0), 2)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--max_fe", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--out", default="results/param_sweep.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    funcs = list(range(1, 13))

    tasks = [
        (v, f, args.dim, seed, args.max_fe)
        for v in SWEEP_CONFIGS
        for f in funcs
        for seed in range(args.seeds)
    ]

    total = len(tasks)
    print(f"Parameter Sweep: {len(SWEEP_CONFIGS)} configs x {len(funcs)} funcs x {args.seeds} seeds = {total} tasks")
    print(f"Workers: {args.workers}", flush=True)

    fields = ["variant", "func", "seed", "best_fit", "elapsed_s", "explore_pct"]
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writeheader()

    completed = 0
    t_start = time.perf_counter()

    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_run_one, tasks):
            completed += 1
            with open(out_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writerow(row)
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                print(f"  [{completed}/{total}] {100*completed/total:.0f}%  ETA {eta:.0f}s", flush=True)

    print("\nDone. Analyzing...\n")

    # Analysis
    from collections import defaultdict
    from scipy.stats import wilcoxon

    data = defaultdict(list)
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            data[(row["variant"], int(row["func"]))].append(float(row["best_fit"]))

    orc_variants = [v for v in SWEEP_CONFIGS if v != "NL-SHADE"]

    for v in orc_variants:
        wins = losses = ties = 0
        print(f"--- {v} vs NL-SHADE ---")
        for func in funcs:
            a = np.array(data.get(("NL-SHADE", func), []))
            b = np.array(data.get((v, func), []))
            if not len(a) or not len(b):
                continue
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            try:
                p = wilcoxon(a, b, alternative="two-sided").pvalue if not np.allclose(a, b) else 1.0
            except:
                p = 1.0
            if p < 0.05 and b.mean() < a.mean():
                mark = "WIN"
                wins += 1
            elif p < 0.05 and a.mean() < b.mean():
                mark = "LOSS"
                losses += 1
            else:
                mark = "tie"
                ties += 1
            print(f"  F{func:<3} NL={a.mean():.3e} ORC={b.mean():.3e}  p={p:.4f} {mark}")
        print(f"  TOTAL: wins={wins} losses={losses} ties={ties}\n")


if __name__ == "__main__":
    freeze_support()
    main()

#!/usr/bin/env python3
"""
analyze_overnight.py  --  Post-process overnight benchmark CSV into paper-ready output.

Usage
-----
  python benchmarks/analyze_overnight.py results/orc_shade_cec2022.csv

Produces:
  - LaTeX table fragment (copy-paste into paper)
  - Wilcoxon test summary
  - Exploration diagnostic table
  - Individual convergence curves (one per function, saved to results/figs/)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(csv_path: Path) -> dict:
    """Returns {(variant, dim, func): [best_fit, ...]}"""
    data: dict = defaultdict(list)
    diag: dict = defaultdict(list)  # (variant, dim, func): [{stats}]
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["variant"], int(row["dim"]), int(row["func"]))
            data[key].append(float(row["best_fit"]))
            if row.get("explore_pct"):
                diag[key].append({
                    "explore_pct": float(row["explore_pct"]),
                    "mean_kappa": float(row["mean_kappa"]),
                })
    return data, diag


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

LATEX_TEMPLATE = r"""\begin{{table*}}[t]
\centering
\caption{{CEC 2022 results (D={dim}, {n} independent runs, {budget} FEs). Error = \(f(\mathbf{{x}}) - f^*\). Mean $\pm$ Std. Wilcoxon signed-rank test ($\alpha=0.05$): \(\dagger\) ORC-SHADE wins, \(\ddagger\) NL-SHADE wins.}}
\label{{tab:cec2022_d{dim}}}
\begin{{tabular}}{{l rr rr c}}
\toprule
Func & \multicolumn{{2}}{{c}}{{NL-SHADE}} & \multicolumn{{2}}{{c}}{{ORC-SHADE}} & Wilcoxon \\
     & Mean & Std & Mean & Std & \\
\midrule
{rows}
\midrule
Wins & \multicolumn{{4}}{{c}}{{ORC: {orc_w} \quad NL: {nl_w} \quad Tie: {ties}}} & \\
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def _sci(x: float) -> str:
    if x == 0.0:
        return r"\mathbf{0}"
    m = f"{x:.2e}"
    base, exp = m.split("e")
    exp = int(exp)
    return rf"{base}\times10^{{{exp}}}"


def make_latex(data: dict, dim: int, funcs: list[int],
               var_a: str = "NL-SHADE", var_b: str = "ORC-SHADE") -> str:
    rows = []
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
            marker, orc_w = r"\dagger", orc_w + 1
        elif sig and a.mean() < b.mean():
            marker, nl_w = r"\ddagger", nl_w + 1
        else:
            marker, ties = "", ties + 1

        # Bold the better mean
        a_str = _sci(a.mean())
        b_str = _sci(b.mean())
        if b.mean() < a.mean() and sig:
            b_str = rf"\mathbf{{{b_str}}}"
        elif a.mean() < b.mean() and sig:
            a_str = rf"\mathbf{{{a_str}}}"

        rows.append(
            rf"F{f} & ${a_str}$ & ${_sci(a.std())}$ "
            rf"& ${b_str}$ & ${_sci(b.std())}$ & {marker} \\"
        )

    n_seeds = min(len(data.get((var_a, dim, funcs[0]), [])),
                  len(data.get((var_b, dim, funcs[0]), [])))
    budget = {10: "200{,}000", 20: "1{,}000{,}000"}.get(dim, "200{,}000")

    return LATEX_TEMPLATE.format(
        dim=dim, n=n_seeds, budget=budget,
        rows="\n".join(rows),
        orc_w=orc_w, nl_w=nl_w, ties=ties,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(data: dict, diag: dict, dim: int, funcs: list[int],
                  var_a: str = "NL-SHADE", var_b: str = "ORC-SHADE"):
    header = (
        f"{'F':<5} | {'NL-SHADE':>18} | {'ORC-SHADE':>18} "
        f"| {'Win':>6} | {'p':>8} | {'expl%':>6} | {'kappa':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

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

        d_stats = diag.get((var_b, dim, f), [])
        ep = np.mean([d["explore_pct"] for d in d_stats]) if d_stats else float("nan")
        mk = np.mean([d["mean_kappa"] for d in d_stats]) if d_stats else float("nan")

        print(
            f"F{f:<4} | {a.mean():>8.2e}+/-{a.std():.1e} "
            f"| {b.mean():>8.2e}+/-{b.std():.1e} "
            f"| {win:>6} | {p:>8.4f} | {ep:>6.1f} | {mk:>7.3f}"
        )

    print(sep)
    print(f"ORC wins: {orc_w}  NL wins: {nl_w}  Ties: {ties}")


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def print_ablation(data: dict, dim: int, funcs: list[int]):
    variants = [v for v in {k[0] for k in data if k[1] == dim} if "ORC" in v]
    if not variants:
        return
    variants = sorted(variants)

    print(f"\nAblation study (D={dim})  -- mean error over all F1-F12")
    print(f"{'Variant':<30} | {'Mean':>10} | {'Best fn':>8} | {'Worst fn':>9}")
    print("-" * 65)
    for v in variants:
        means = []
        for f in funcs:
            arr = np.array(data.get((v, dim, f), []))
            if len(arr):
                means.append((f, arr.mean()))
        if not means:
            continue
        vals = [m for _, m in means]
        best_f = means[np.argmin(vals)][0]
        worst_f = means[np.argmax(vals)][0]
        print(f"{v:<30} | {np.mean(vals):>10.3e} | {'F'+str(best_f):>8} | {'F'+str(worst_f):>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/orc_shade_cec2022.csv")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--funcs", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    data, diag = load(csv_path)

    for dim in args.dims:
        has = any(k[1] == dim for k in data)
        if not has:
            continue
        print(f"\n{'='*70}\nD = {dim}\n{'='*70}")
        print_summary(data, diag, dim, args.funcs)

        if args.ablation:
            print_ablation(data, dim, args.funcs)

        if args.latex:
            print(f"\n--- LaTeX table (D={dim}) ---")
            print(make_latex(data, dim, args.funcs))


if __name__ == "__main__":
    main()

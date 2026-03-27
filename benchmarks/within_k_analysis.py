#!/usr/bin/env python3
"""
Within-K Difficulty Prediction Analysis.

Computes Spearman rank correlations between landscape metrics and
algorithm performance WITHIN each fixed K (or nu) level, not pooled
across ruggedness levels.

This answers the skeptical reviewer question: "Is ORC just measuring
ruggedness with extra steps, or does it capture instance-level structure
within a fixed ruggedness class?"

Usage:
    python3 benchmarks/within_k_analysis.py
    python3 benchmarks/within_k_analysis.py --data results/landscape_discrete_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def compute_within_group_correlations(results: list) -> dict:
    """Compute Spearman correlations within each config group."""
    groups = defaultdict(list)
    for r in results:
        if r['type'] == 'NK':
            key = f"NK N={r['config']['N']}, K={r['config']['K']}"
        elif r['type'] == 'WModel':
            key = f"W nu={r['config']['nu']}"
        else:
            continue

        if r['n_local_optima'] <= 1:
            continue

        groups[key].append(r)

    metrics = [
        ('Mean ORC', 'mean_orc'),
        ('FDC', 'fdc'),
        ('Autocorr. len', 'autocorrelation_length'),
        ('Info content H', 'information_content_H'),
        ('Basin entropy', 'basin_entropy'),
    ]

    targets = [
        ('HC success', 'algo_HC_success_rate'),
        ('EA success', 'algo_EA_success_rate'),
        ('RS mean fit', 'algo_RS_mean'),
    ]

    all_correlations = {}

    for group_key in sorted(groups.keys()):
        rows = groups[group_key]
        if len(rows) < 5:
            continue

        group_corrs = {}
        for metric_name, metric_key in metrics:
            for target_name, target_key in targets:
                x = np.array([r[metric_key] for r in rows])
                y = np.array([r[target_key] for r in rows])

                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    rho, pval = 0.0, 1.0
                else:
                    rho, pval = spearmanr(x, y)

                group_corrs[(metric_name, target_name)] = {
                    'rho': float(rho),
                    'pval': float(pval),
                    'n': len(rows),
                }

        all_correlations[group_key] = group_corrs

    return all_correlations


def print_table(correlations: dict):
    metrics = ['Mean ORC', 'FDC', 'Autocorr. len', 'Info content H', 'Basin entropy']
    target = 'HC success'

    print(f"\n{'='*100}")
    print(f"Within-Group Spearman Correlations with {target}")
    print(f"{'='*100}")

    header = f"{'Group':<25}"
    for m in metrics:
        header += f" {m:>15}"
    print(header)
    print("-" * 100)

    for group in sorted(correlations.keys()):
        corrs = correlations[group]
        row = f"{group:<25}"
        for m in metrics:
            key = (m, target)
            if key in corrs:
                rho = corrs[key]['rho']
                pval = corrs[key]['pval']
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                row += f" {rho:>11.3f}{sig:<4}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    # Also print for EA and RS targets
    for target in ['EA success', 'RS mean fit']:
        print(f"\n{'='*100}")
        print(f"Within-Group Spearman Correlations with {target}")
        print(f"{'='*100}")

        header = f"{'Group':<25}"
        for m in metrics:
            header += f" {m:>15}"
        print(header)
        print("-" * 100)

        for group in sorted(correlations.keys()):
            corrs = correlations[group]
            row = f"{group:<25}"
            for m in metrics:
                key = (m, target)
                if key in corrs:
                    rho = corrs[key]['rho']
                    pval = corrs[key]['pval']
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    row += f" {rho:>11.3f}{sig:<4}"
                else:
                    row += f" {'N/A':>15}"
            print(row)


def print_pooled_vs_within(results: list, correlations: dict):
    """Compare pooled correlations with within-group correlations."""
    nontrivial = [r for r in results if r['n_local_optima'] > 1]

    metrics = [
        ('Mean ORC', 'mean_orc'),
        ('FDC', 'fdc'),
        ('Autocorr. len', 'autocorrelation_length'),
        ('Basin entropy', 'basin_entropy'),
    ]

    target_key = 'algo_HC_success_rate'

    print(f"\n{'='*100}")
    print("POOLED vs WITHIN-GROUP: HC Success Rate")
    print(f"{'='*100}")

    for metric_name, metric_key in metrics:
        x_all = np.array([r[metric_key] for r in nontrivial])
        y_all = np.array([r[target_key] for r in nontrivial])
        rho_pooled, p_pooled = spearmanr(x_all, y_all)

        within_rhos = []
        for group in sorted(correlations.keys()):
            corrs = correlations[group]
            key = (metric_name, 'HC success')
            if key in corrs and corrs[key]['n'] >= 10:
                within_rhos.append(corrs[key]['rho'])

        mean_within = np.mean(within_rhos) if within_rhos else float('nan')

        print(f"  {metric_name:<20}  pooled rho={rho_pooled:+.3f}  "
              f"mean within-group rho={mean_within:+.3f}  "
              f"(from {len(within_rhos)} groups with n>=10)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='results/landscape_discrete_v2.json')
    parser.add_argument('--out', default='results/within_k_correlations.json')
    args = parser.parse_args()

    results = load_results(args.data)
    print(f"Loaded {len(results)} instances from {args.data}")

    correlations = compute_within_group_correlations(results)

    print_table(correlations)
    print_pooled_vs_within(results, correlations)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for group, corrs in correlations.items():
        serializable[group] = {
            f"{m}|{t}": v for (m, t), v in corrs.items()
        }
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved correlations to {out_path}")


if __name__ == '__main__':
    main()

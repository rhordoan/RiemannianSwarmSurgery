#!/usr/bin/env python3
"""
Generate publication-quality figures for the PPSN 2026 paper.

Reads results/landscape_discrete_v2.json and produces PDF figures
in paper/figures/.

Figures:
  1. ORC advantage bar chart (hero figure)
  2. Mean ORC vs HC success rate scatter
  3. Correlation heatmap
  4. ORC vs K/nu trend lines
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / 'results' / 'landscape_discrete_v2.json'
RESULTS_N20_FILE = ROOT / 'results' / 'landscape_discrete_n20.json'
FIG_DIR = ROOT / 'paper' / 'figures'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_results():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return data


def load_n20_results():
    if RESULTS_N20_FILE.exists():
        with open(RESULTS_N20_FILE) as f:
            return json.load(f)
    return []


def group_by_label(data):
    groups = defaultdict(list)
    for r in data:
        groups[r['label']].append(r)
    return groups


def fig1_orc_advantage(data, out_path):
    """Bar chart: %Better for ORC vs Random vs Worst-ORC directions."""
    groups = group_by_label(data)

    configs = [
        'NK(N=16,K=2,model=adjacent)',
        'NK(N=16,K=4,model=adjacent)',
        'NK(N=16,K=6,model=adjacent)',
        'NK(N=16,K=8,model=adjacent)',
        'NK(N=16,K=12,model=adjacent)',
        'NK(N=16,K=15,model=adjacent)',
        'W(n=16,nu=3)',
        'W(n=16,nu=4)',
        'W(n=16,nu=6)',
        'W(n=16,nu=8)',
        'W(n=16,nu=16)',
    ]
    short_labels = [
        'NK\nK=2', 'NK\nK=4', 'NK\nK=6', 'NK\nK=8', 'NK\nK=12', 'NK\nK=15',
        'W\nν=3', 'W\nν=4', 'W\nν=6', 'W\nν=8', 'W\nν=16',
    ]

    orc_vals, rand_vals, worst_vals = [], [], []
    orc_errs, rand_errs, worst_errs = [], [], []

    for cfg in configs:
        rows = groups[cfg]
        orc = [r['frac_leads_to_better'] for r in rows]
        rand = [r['frac_random_leads_to_better'] for r in rows]
        worst = [r['frac_worst_orc_leads_to_better'] for r in rows]
        orc_vals.append(np.mean(orc) * 100)
        rand_vals.append(np.mean(rand) * 100)
        worst_vals.append(np.mean(worst) * 100)
        orc_errs.append(stats.sem(orc) * 100)
        rand_errs.append(stats.sem(rand) * 100)
        worst_errs.append(stats.sem(worst) * 100)

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - width, orc_vals, width, yerr=orc_errs, label='Min-ORC direction',
           color='#2166ac', capsize=2, linewidth=0.5, edgecolor='white')
    ax.bar(x, rand_vals, width, yerr=rand_errs, label='Random direction',
           color='#999999', capsize=2, linewidth=0.5, edgecolor='white')
    ax.bar(x + width, worst_vals, width, yerr=worst_errs, label='Max-ORC direction',
           color='#b2182b', capsize=2, linewidth=0.5, edgecolor='white')

    ax.set_ylabel('Reaches better optimum (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def fig2_scatter(data, out_path):
    """Scatter: mean ORC at optima vs HC success rate."""
    nk_orc, nk_hc = [], []
    wm_orc, wm_hc = [], []

    for r in data:
        if r['n_local_optima'] <= 1:
            continue
        orc = r['mean_orc']
        hc = r['algo_HC_success_rate'] * 100

        if r['type'] == 'NK':
            nk_orc.append(orc)
            nk_hc.append(hc)
        else:
            wm_orc.append(orc)
            wm_hc.append(hc)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(nk_orc, nk_hc, s=12, alpha=0.5, label='NK landscapes',
               color='#2166ac', edgecolors='none')
    ax.scatter(wm_orc, wm_hc, s=12, alpha=0.5, label='W-model',
               color='#d6604d', marker='s', edgecolors='none')

    all_orc = nk_orc + wm_orc
    all_hc = nk_hc + wm_hc
    rho, pval = stats.spearmanr(all_orc, all_hc)
    ax.set_xlabel('Mean ORC at local optima')
    ax.set_ylabel('HC success rate (%)')
    ax.set_title(f'Spearman ρ = {rho:.3f} (p < 0.001)')
    ax.legend(frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def fig3_heatmap(data, out_path):
    """Correlation heatmap: metrics vs algorithm performance."""
    filtered = [r for r in data if r['n_local_optima'] > 1]

    feature_keys = [
        ('mean_orc', 'Mean ORC'),
        ('frac_leads_to_better', '% ORC→better'),
        ('frac_random_leads_to_better', '% Random→better'),
        ('fdc', 'FDC'),
        ('autocorrelation_length', 'Autocorr. length'),
        ('information_content_H', 'Info. content H'),
        ('basin_entropy', 'Basin entropy'),
    ]
    target_keys = [
        ('algo_HC_success_rate', 'HC success'),
        ('algo_EA_success_rate', 'EA success'),
        ('algo_RS_mean', 'RS mean fit.'),
    ]

    n_feat = len(feature_keys)
    n_targ = len(target_keys)
    corr_matrix = np.zeros((n_feat, n_targ))

    for i, (fk, _) in enumerate(feature_keys):
        feat_vals = [r[fk] for r in filtered]
        for j, (tk, _) in enumerate(target_keys):
            targ_vals = [r[tk] for r in filtered]
            rho, _ = stats.spearmanr(feat_vals, targ_vals)
            corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_targ))
    ax.set_xticklabels([t[1] for t in target_keys], rotation=30, ha='right')
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f[1] for f in feature_keys])

    for i in range(n_feat):
        for j in range(n_targ):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ')
    ax.set_title('Feature–Performance Correlations')

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def fig4_trends(data, n20_data, out_path):
    """Trend lines: ORC advantage and mean ORC vs K (NK) and nu (W-model)."""
    groups = group_by_label(data)
    n20_groups = group_by_label(n20_data) if n20_data else {}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: NK landscapes - ORC advantage vs K
    nk_Ks = [2, 4, 6, 8, 12, 15]
    orc_adv, rand_adv = [], []
    for K in nk_Ks:
        rows = groups[f'NK(N=16,K={K},model=adjacent)']
        orc_adv.append(np.mean([r['frac_leads_to_better'] for r in rows]) * 100)
        rand_adv.append(np.mean([r['frac_random_leads_to_better'] for r in rows]) * 100)

    ax = axes[0]
    ax.plot(nk_Ks, orc_adv, 'o-', color='#2166ac', label='Min-ORC dir. (N=16)', markersize=5)
    ax.plot(nk_Ks, rand_adv, 's--', color='#999999', label='Random dir. (N=16)', markersize=4)

    if n20_groups:
        n20_Ks = [4, 8, 15, 19]
        orc_n20, rand_n20 = [], []
        for K in n20_Ks:
            key = f'NK(N=20,K={K},model=adjacent)'
            if key in n20_groups:
                rows = n20_groups[key]
                orc_n20.append(np.mean([r['frac_leads_to_better'] for r in rows]) * 100)
                rand_n20.append(np.mean([r['frac_random_leads_to_better'] for r in rows]) * 100)
            else:
                orc_n20.append(np.nan)
                rand_n20.append(np.nan)
        ax.plot(n20_Ks, orc_n20, marker='^', color='#2166ac', label='Min-ORC dir. (N=20)',
                markersize=5, alpha=0.6, linestyle=':')
        ax.plot(n20_Ks, rand_n20, marker='v', color='#999999', label='Random dir. (N=20)',
                markersize=4, alpha=0.6, linestyle=':')

    ax.set_xlabel('Epistasis K')
    ax.set_ylabel('Reaches better optimum (%)')
    ax.set_title('(a) NK landscapes')
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: W-model - ORC advantage vs nu
    wm_nus = [3, 4, 6, 8, 16]
    orc_w, rand_w = [], []
    for nu in wm_nus:
        rows = groups[f'W(n=16,nu={nu})']
        orc_w.append(np.mean([r['frac_leads_to_better'] for r in rows]) * 100)
        rand_w.append(np.mean([r['frac_random_leads_to_better'] for r in rows]) * 100)

    ax = axes[1]
    ax.plot(wm_nus, orc_w, 'o-', color='#d6604d', label='Min-ORC direction', markersize=5)
    ax.plot(wm_nus, rand_w, 's--', color='#999999', label='Random direction', markersize=4)
    ax.set_xlabel('Epistasis ν')
    ax.set_ylabel('Reaches better optimum (%)')
    ax.set_title('(b) W-model')
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading results...')
    data = load_results()
    n20_data = load_n20_results()
    print(f'  Loaded {len(data)} instances (N=16)')
    if n20_data:
        n20_only = [r for r in n20_data if r.get('config', {}).get('N') == 20]
        print(f'  Loaded {len(n20_only)} instances (N=20)')
    else:
        n20_only = []

    print('Generating figures...')
    fig1_orc_advantage(data, FIG_DIR / 'fig_orc_advantage.pdf')
    fig2_scatter(data, FIG_DIR / 'fig_scatter_orc_hc.pdf')
    fig3_heatmap(data, FIG_DIR / 'fig_correlation_heatmap.pdf')
    fig4_trends(data, n20_only, FIG_DIR / 'fig_trends.pdf')
    print('Done.')


if __name__ == '__main__':
    main()

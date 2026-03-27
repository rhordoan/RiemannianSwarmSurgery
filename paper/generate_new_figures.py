#!/usr/bin/env python3
"""Generate new publication-quality PDF figures for the revised paper."""

from __future__ import annotations
import json, sys, os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def fig_phase_transition():
    """Phase transition plot: OTG properties vs K and nu."""
    with open(ROOT / 'results' / 'otg_funnel_analysis.json') as f:
        data = json.load(f)

    nk_data = defaultdict(list)
    wm_data = defaultdict(list)
    for r in data:
        if r['type'] == 'NK':
            nk_data[r['config']['K']].append(r)
        else:
            wm_data[r['config']['nu']].append(r)

    metrics = [
        ('n_otg_funnels', 'n_local_optima', 'Compression\n(funnels / optima)', True),
        ('dag_depth', None, 'DAG depth', False),
        ('frac_in_multi_cycles', None, 'Fraction in\nmulti-node cycles', False),
        ('otg_top5', None, 'Fraction reaching\ntop-5% attractor', False),
    ]

    fig, axes = plt.subplots(len(metrics), 2, figsize=(7, 8), sharex='col')

    for row, (metric, denom, ylabel, is_ratio) in enumerate(metrics):
        for col, (dd, xlabel, title) in enumerate([
            (nk_data, '$K$', 'NK Landscapes'),
            (wm_data, r'$\nu$', 'W-Model'),
        ]):
            ax = axes[row, col]
            params = sorted(dd.keys())
            means, stds = [], []
            for p in params:
                rows = dd[p]
                if is_ratio and denom:
                    vals = [r[metric] / max(r[denom], 1) for r in rows]
                else:
                    vals = [r[metric] for r in rows]
                means.append(np.mean(vals))
                stds.append(np.std(vals) / np.sqrt(len(vals)))

            ax.errorbar(params, means, yerr=stds, marker='s', capsize=3,
                        linewidth=1.8, markersize=5, color='#2c7bb6',
                        markeredgecolor='white', markeredgewidth=0.5)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title, fontweight='bold')
            if row == len(metrics) - 1:
                ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.tight_layout(h_pad=1.2)
    fig.savefig(FIG_DIR / 'fig_phase_transition.pdf')
    plt.close(fig)
    print('  Saved fig_phase_transition.pdf')


def fig_otg_visualization():
    """OTG visualization for W(n=16,nu=3) seed=0."""
    import networkx as nx

    with open(ROOT / 'results' / 'otg_funnel_analysis.json') as f:
        all_data = json.load(f)

    # We need the full data with OTG edges. Re-run for one instance.
    sys.path.insert(0, str(ROOT))
    from benchmarks.otg_funnel_analysis import _analyze_funnels_instance

    result = _analyze_funnels_instance({
        'type': 'WModel', 'config': {'n': 16, 'nu': 3},
        'seed': 0, 'gamma': 1.0,
    })

    vis = result['_vis_data']
    otg_edges = vis['otg_edges']
    basins = vis['basins']
    opt_fit = vis['opt_fit']
    funnel_labels = vis['otg_funnel']
    global_opt = vis['global_opt']
    cycle_nodes = set(vis['cycle_nodes'])
    multi_cycle_nodes = set(vis['multi_cycle_nodes'])

    G = nx.DiGraph()
    for src, dst in otg_edges.items():
        if src != dst:
            G.add_edge(int(src), int(dst))
        else:
            G.add_node(int(src))

    nodes = list(G.nodes())
    n = len(nodes)

    funnel_sizes = Counter(funnel_labels.values())
    sorted_funnels = sorted(funnel_sizes.keys(), key=lambda f: -funnel_sizes[f])

    n_colored = min(10, len(sorted_funnels))
    cmap_colors = plt.colormaps['tab10']
    funnel_color_map = {}
    for i, f in enumerate(sorted_funnels[:n_colored]):
        funnel_color_map[f] = cmap_colors(i)
    gray = (0.82, 0.82, 0.82, 0.5)
    for f in sorted_funnels[n_colored:]:
        funnel_color_map[f] = gray

    global_funnel = funnel_labels.get(str(global_opt), funnel_labels.get(global_opt))
    if global_funnel is not None:
        funnel_color_map[global_funnel] = (1.0, 0.78, 0.0, 1.0)

    node_colors = []
    for nd in G.nodes():
        fl = funnel_labels.get(str(nd), funnel_labels.get(nd, -1))
        node_colors.append(funnel_color_map.get(fl, gray))

    bsizes = np.array([basins.get(str(nd), basins.get(nd, 1)) for nd in G.nodes()])
    node_sizes = 25 + 600 * (bsizes / max(bsizes.max(), 1))

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u in multi_cycle_nodes and v in multi_cycle_nodes:
            edge_colors.append('firebrick')
            edge_widths.append(2.0)
        else:
            edge_colors.append((0.35, 0.35, 0.35, 0.25))
            edge_widths.append(0.4)

    elw = [1.5 if nd in cycle_nodes else 0.3 for nd in G.nodes()]
    eec = ['black' if nd in cycle_nodes else (0.4, 0.4, 0.4) for nd in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(n + 1), iterations=120)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=6, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.06', alpha=0.65,
        min_source_margin=3, min_target_margin=3,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        edgecolors=eec, linewidths=elw,
    )

    if global_opt in G.nodes():
        gidx = list(G.nodes()).index(global_opt)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=[global_opt],
            node_size=[node_sizes[gidx] * 2.5],
            node_color=['gold'], edgecolors='black', linewidths=2.5,
            node_shape='*',
        )

    legend_els = []
    for i, f in enumerate(sorted_funnels[:n_colored]):
        sz = funnel_sizes[f]
        lbl = f"Funnel ({sz} optima)"
        if f == global_funnel:
            lbl += " ★"
        legend_els.append(mpatches.Patch(
            facecolor=funnel_color_map[f], edgecolor='gray', linewidth=0.5,
            label=lbl))
    n_gray_f = len(sorted_funnels) - n_colored
    if n_gray_f > 0:
        legend_els.append(mpatches.Patch(
            facecolor=gray, edgecolor='gray', linewidth=0.5,
            label=f"{n_gray_f} small funnels"))

    ax.legend(handles=legend_els, loc='upper left', fontsize=6.5,
              framealpha=0.85, ncol=1, handlelength=1.2, handletextpad=0.4)

    ax.set_title(
        f"ORC Transition Graph — W-Model ($n$=16, $\\nu$=3)\n"
        f"{n} optima, {result['n_otg_funnels']} funnels, "
        f"DAG depth={result['dag_depth']}, "
        f"top-5% = {result['otg_top5']:.0%}",
        fontsize=10, fontweight='bold',
    )
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_otg_visualization.pdf')
    plt.close(fig)
    print('  Saved fig_otg_visualization.pdf')


if __name__ == '__main__':
    print('Generating figures...')
    fig_phase_transition()
    fig_otg_visualization()
    print('Done.')

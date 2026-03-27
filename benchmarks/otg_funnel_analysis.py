#!/usr/bin/env python3
"""
OTG Funnel Analysis:
  1. SCC decomposition of OTG, comparison with LON-based funnels (ARI/NMI)
  2. Publication-quality OTG visualization with funnel highlighting
  3. Phase transition analysis of OTG properties vs ruggedness

Usage:
    python3 benchmarks/otg_funnel_analysis.py --workers 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Clustering comparison metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def _adjusted_rand_index(labels1, labels2):
    n = len(labels1)
    if n <= 1:
        return 1.0
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    cross = Counter(zip(labels1, labels2))

    def comb2(k):
        return k * (k - 1) / 2

    sum_nij = sum(comb2(v) for v in cross.values())
    sum_ai = sum(comb2(v) for v in c1.values())
    sum_bj = sum(comb2(v) for v in c2.values())
    comb_n = comb2(n)

    if comb_n == 0:
        return 1.0
    expected = sum_ai * sum_bj / comb_n
    max_idx = (sum_ai + sum_bj) / 2

    if abs(max_idx - expected) < 1e-12:
        return 1.0
    return (sum_nij - expected) / (max_idx - expected)


def _nmi(labels1, labels2):
    n = len(labels1)
    if n <= 1:
        return 1.0
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    cross = Counter(zip(labels1, labels2))

    def entropy(counts):
        return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)

    h1 = entropy(c1)
    h2 = entropy(c2)

    mi = 0.0
    for (l1, l2), nij in cross.items():
        if nij > 0:
            mi += (nij / n) * math.log(nij * n / (c1[l1] * c2[l2]))

    denom = h1 + h2
    if denom < 1e-12:
        return 1.0
    return 2 * mi / denom


# ---------------------------------------------------------------------------
# Funnel computation on functional graphs
# ---------------------------------------------------------------------------

def _follow_to_terminal(edges, start):
    """Follow edges until cycle detected. Return cycle representative (min node in cycle)."""
    visited = set()
    path = []
    current = start
    while current not in visited:
        visited.add(current)
        path.append(current)
        nxt = edges.get(current, current)
        if nxt == current:
            return current
        current = nxt
    # current is in a cycle — find cycle and return min node as representative
    cycle_start = path.index(current)
    cycle = path[cycle_start:]
    return min(cycle)


def _compute_funnels(edges, all_nodes):
    """Each node -> terminal attractor representative."""
    return {node: _follow_to_terminal(edges, node) for node in all_nodes}


def _find_cycles(edges, nodes):
    """Find all cycles in a functional graph. Return list of cycles (each a list of nodes)."""
    visited_global = set()
    cycles = []

    for start in nodes:
        if start in visited_global:
            continue
        path = []
        visited = set()
        current = start
        while current not in visited and current not in visited_global:
            visited.add(current)
            path.append(current)
            nxt = edges.get(current, current)
            if nxt == current:
                break
            current = nxt

        if current in visited and current not in visited_global:
            idx = path.index(current)
            cycle = path[idx:]
            cycles.append(cycle)

        visited_global.update(visited)

    return cycles


# ---------------------------------------------------------------------------
# Per-instance analysis
# ---------------------------------------------------------------------------

def _analyze_funnels_instance(args):
    """Full funnel analysis for one landscape instance."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.orc_discrete import full_landscape_analysis, hill_climb

    benchmark_type = args['type']
    config = args['config']
    seed = args['seed']
    gamma = args.get('gamma', 1.0)

    if benchmark_type == 'NK':
        from src.nk_landscape import NKLandscape
        landscape = NKLandscape(
            N=config['N'], K=config['K'],
            model=config.get('model', 'adjacent'), seed=seed,
        )
        n_bits = config['N']
        label = f"NK(N={config['N']},K={config['K']})"
    elif benchmark_type == 'WModel':
        from src.wmodel import WModel
        landscape = WModel(
            n=config['n'], nu=config['nu'],
            gamma=config.get('gamma', 0),
            mu=config.get('mu', 1), seed=seed,
        )
        n_bits = config['n']
        label = f"W(n={config['n']},nu={config['nu']})"
    else:
        raise ValueError(f"Unknown: {benchmark_type}")

    fitness = landscape.fitness
    neighbor_fn = landscape.neighbor_fn
    space_size = landscape.space_size
    global_opt = int(np.argmin(fitness))

    # ---- 1. Build OTG via full_landscape_analysis ----
    orc_result = full_landscape_analysis(
        space_size, fitness, neighbor_fn, gamma,
        n_random_trials=30, seed=seed,
    )

    local_optima = orc_result['local_optima']
    analyses = orc_result['orc_analyses']
    basins = orc_result['basin_sizes']
    n_opt = len(local_optima)

    if n_opt < 5:
        return None

    otg_edges = {}
    for a in analyses:
        src = a['opt_idx']
        dst = a['dest_opt']
        otg_edges[src] = dst if dst is not None else src

    # ---- 2. Build LON d=1 (exhaustive: all N neighbors → HC → mode dest) ----
    lon_d1_edges = {}
    for opt in local_optima:
        nbrs = neighbor_fn(opt)
        dest_counts = Counter()
        for nbr in nbrs:
            dest = hill_climb(nbr, fitness, neighbor_fn)
            dest_counts[dest] += 1
        lon_d1_edges[opt] = dest_counts.most_common(1)[0][0]

    # ---- 3. Build LON d=3 (stochastic: 100 trials of flip-3 → HC → mode dest) ----
    rng = np.random.RandomState(seed * 1000 + 777)
    lon_d3_edges = {}
    for opt in local_optima:
        dest_counts = Counter()
        for _ in range(100):
            perturbed = opt
            bits = rng.choice(n_bits, size=min(3, n_bits), replace=False)
            for b in bits:
                perturbed ^= (1 << b)
            dest = hill_climb(perturbed, fitness, neighbor_fn)
            dest_counts[dest] += 1
        lon_d3_edges[opt] = dest_counts.most_common(1)[0][0]

    # ---- 4. Compute funnels for each graph ----
    otg_funnels = _compute_funnels(otg_edges, local_optima)
    lon_d1_funnels = _compute_funnels(lon_d1_edges, local_optima)
    lon_d3_funnels = _compute_funnels(lon_d3_edges, local_optima)

    otg_labels = [otg_funnels[o] for o in local_optima]
    lon_d1_labels = [lon_d1_funnels[o] for o in local_optima]
    lon_d3_labels = [lon_d3_funnels[o] for o in local_optima]

    n_otg_funnels = len(set(otg_labels))
    n_lon_d1_funnels = len(set(lon_d1_labels))
    n_lon_d3_funnels = len(set(lon_d3_labels))

    # ---- 5. Comparison metrics ----
    ari_otg_d1 = _adjusted_rand_index(otg_labels, lon_d1_labels)
    ari_otg_d3 = _adjusted_rand_index(otg_labels, lon_d3_labels)
    ari_d1_d3 = _adjusted_rand_index(lon_d1_labels, lon_d3_labels)
    nmi_otg_d1 = _nmi(otg_labels, lon_d1_labels)
    nmi_otg_d3 = _nmi(otg_labels, lon_d3_labels)
    nmi_d1_d3 = _nmi(lon_d1_labels, lon_d3_labels)

    # ---- 6. SCC / cycle statistics for OTG ----
    otg_cycles = _find_cycles(otg_edges, local_optima)
    n_self_loops = sum(1 for c in otg_cycles if len(c) == 1)
    multi_cycles = [c for c in otg_cycles if len(c) > 1]
    n_multi_cycles = len(multi_cycles)
    n_in_multi_cycles = sum(len(c) for c in multi_cycles)
    frac_in_multi_cycles = n_in_multi_cycles / n_opt if n_opt > 0 else 0.0

    # Number of SCCs = number of cycles (terminal) + number of tail singletons
    # In a functional graph, #SCCs = #nodes - #tail_nodes_in_cycles + #cycles
    # But for our purposes, the number of distinct attractors (funnels) is what matters
    n_attractors = len(otg_cycles)  # self-loops count as attractors

    # ---- 7. Funnel quality: mean terminal rank ----
    all_fit = np.array([fitness[o] for o in local_optima])
    sorted_fit = np.sort(all_fit)

    def _terminal_ranks(funnels_dict):
        ranks = []
        for opt in local_optima:
            term = funnels_dict[opt]
            rank = float(np.searchsorted(sorted_fit, fitness[term]) / max(n_opt - 1, 1))
            ranks.append(rank)
        return ranks

    otg_ranks = _terminal_ranks(otg_funnels)
    d1_ranks = _terminal_ranks(lon_d1_funnels)
    d3_ranks = _terminal_ranks(lon_d3_funnels)

    # ---- 8. SCC condensation DAG properties ----
    # Build DAG: funnel_rep -> set of downstream funnel_reps
    # (In OTG, each funnel rep is a cycle; tails flow toward cycles)
    import networkx as nx
    G = nx.DiGraph()
    for src, dst in otg_edges.items():
        G.add_edge(src, dst)
    sccs = list(nx.strongly_connected_components(G))
    condensation = nx.condensation(G)
    n_scc = len(sccs)
    dag_depth = nx.dag_longest_path_length(condensation) if len(condensation) > 0 else 0
    # Number of leaves in DAG (= terminal attractors = sinks)
    dag_leaves = sum(1 for n in condensation.nodes() if condensation.out_degree(n) == 0)
    # Number of roots in DAG (= optima with no incoming OTG edge)
    dag_roots = sum(1 for n in condensation.nodes() if condensation.in_degree(n) == 0)

    return {
        'label': label,
        'type': benchmark_type,
        'config': config,
        'seed': seed,
        'n_local_optima': n_opt,
        'n_otg_funnels': n_otg_funnels,
        'n_lon_d1_funnels': n_lon_d1_funnels,
        'n_lon_d3_funnels': n_lon_d3_funnels,
        'ari_otg_d1': ari_otg_d1,
        'ari_otg_d3': ari_otg_d3,
        'ari_d1_d3': ari_d1_d3,
        'nmi_otg_d1': nmi_otg_d1,
        'nmi_otg_d3': nmi_otg_d3,
        'nmi_d1_d3': nmi_d1_d3,
        'n_attractors': n_attractors,
        'n_self_loops': n_self_loops,
        'n_multi_cycles': n_multi_cycles,
        'n_in_multi_cycles': n_in_multi_cycles,
        'frac_in_multi_cycles': frac_in_multi_cycles,
        'n_scc': n_scc,
        'dag_depth': dag_depth,
        'dag_leaves': dag_leaves,
        'dag_roots': dag_roots,
        'otg_mean_rank': float(np.mean(otg_ranks)),
        'otg_top5': float(np.mean([1 if r <= 0.05 else 0 for r in otg_ranks])),
        'd1_mean_rank': float(np.mean(d1_ranks)),
        'd1_top5': float(np.mean([1 if r <= 0.05 else 0 for r in d1_ranks])),
        'd3_mean_rank': float(np.mean(d3_ranks)),
        'd3_top5': float(np.mean([1 if r <= 0.05 else 0 for r in d3_ranks])),
        # Per-optimum data for visualization (only keep for small instances)
        '_vis_data': {
            'otg_edges': {int(k): int(v) for k, v in otg_edges.items()},
            'basins': {int(k): int(v) for k, v in basins.items()},
            'opt_fit': {int(a['opt_idx']): float(a['opt_fitness']) for a in analyses},
            'otg_funnel': {int(k): int(v) for k, v in otg_funnels.items()},
            'global_opt': int(global_opt),
            'cycle_nodes': [n for c in otg_cycles for n in c],
            'multi_cycle_nodes': [n for c in multi_cycles for n in c],
        } if n_opt <= 600 else None,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_otg_funnels(result, out_path):
    """Publication-quality OTG visualization with funnel coloring."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    from matplotlib.colors import to_rgba

    vis = result['_vis_data']
    if vis is None:
        return

    otg_edges = vis['otg_edges']
    basins = vis['basins']
    opt_fit = vis['opt_fit']
    funnel_labels = vis['otg_funnel']
    global_opt = vis['global_opt']
    cycle_nodes = set(vis['cycle_nodes'])
    multi_cycle_nodes = set(vis['multi_cycle_nodes'])

    nodes = sorted(otg_edges.keys())
    n = len(nodes)
    if n < 5:
        return

    # Build networkx graph
    G = nx.DiGraph()
    for src, dst in otg_edges.items():
        if src != dst:
            G.add_edge(src, dst)
        else:
            G.add_node(src)

    # Funnel membership → assign colors
    funnel_sizes = Counter(funnel_labels.values())
    # Sort funnels by size (largest first), assign distinct colors to top funnels
    sorted_funnels = sorted(funnel_sizes.keys(), key=lambda f: -funnel_sizes[f])

    # Colormap: top 12 funnels get distinct colors, rest are gray
    import matplotlib.cm as cm
    n_colored = min(12, len(sorted_funnels))
    cmap = cm.get_cmap('Set3', n_colored)
    funnel_color_map = {}
    for i, f in enumerate(sorted_funnels[:n_colored]):
        funnel_color_map[f] = cmap(i)
    gray = (0.85, 0.85, 0.85, 0.6)
    for f in sorted_funnels[n_colored:]:
        funnel_color_map[f] = gray

    # Global optimum's funnel gets gold
    if global_opt in funnel_labels:
        global_funnel = funnel_labels[global_opt]
        funnel_color_map[global_funnel] = (1.0, 0.84, 0.0, 0.9)

    # Node properties
    node_colors = [funnel_color_map.get(funnel_labels.get(nd, -1), gray) for nd in G.nodes()]
    bsizes = np.array([basins.get(nd, 1) for nd in G.nodes()])
    node_sizes = 40 + 800 * (bsizes / max(bsizes.max(), 1))

    # Edge properties: cycle edges are red, others gray
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u in multi_cycle_nodes and v in multi_cycle_nodes:
            edge_colors.append('firebrick')
            edge_widths.append(2.5)
        else:
            edge_colors.append((0.4, 0.4, 0.4, 0.3))
            edge_widths.append(0.5)

    # Node borders: cycle members have thick dark borders
    edge_lw = [2.0 if nd in cycle_nodes else 0.5 for nd in G.nodes()]
    edge_ec = ['black' if nd in cycle_nodes else (0.3, 0.3, 0.3) for nd in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, seed=42, k=3.0 / np.sqrt(n + 1), iterations=100)

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=8, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.08', alpha=0.7,
        min_source_margin=5, min_target_margin=5,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes, node_color=node_colors,
        edgecolors=edge_ec, linewidths=edge_lw,
    )

    # Highlight global optimum
    if global_opt in G.nodes():
        gidx = list(G.nodes()).index(global_opt)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[global_opt],
            node_size=[node_sizes[gidx] * 2],
            node_color=['gold'],
            edgecolors='black', linewidths=3.0,
            node_shape='*',
        )

    # Legend
    legend_elements = []
    for i, f in enumerate(sorted_funnels[:n_colored]):
        sz = funnel_sizes[f]
        lbl = f"Funnel (n={sz})"
        if f == funnel_labels.get(global_opt, -1):
            lbl += " [global opt]"
        legend_elements.append(mpatches.Patch(
            facecolor=funnel_color_map[f], edgecolor='gray', label=lbl))
    n_gray = len(sorted_funnels) - n_colored
    if n_gray > 0:
        legend_elements.append(mpatches.Patch(
            facecolor=gray, edgecolor='gray',
            label=f"{n_gray} small funnels"))
    legend_elements.append(plt.Line2D(
        [0], [0], marker='*', color='gold', markersize=15,
        linestyle='None', markeredgecolor='black', label='Global optimum'))
    legend_elements.append(plt.Line2D(
        [0], [0], color='firebrick', linewidth=2.5, label='Cycle edges'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.9, ncol=1)

    n_mc = result['n_multi_cycles']
    mc_nodes = result['n_in_multi_cycles']
    ax.set_title(
        f"ORC Transition Graph — {result['label']} seed={result['seed']}\n"
        f"Optima={result['n_local_optima']}, "
        f"Funnels={result['n_otg_funnels']}, "
        f"DAG depth={result['dag_depth']}, "
        f"Cycles={n_mc} ({mc_nodes} nodes), "
        f"Top-5%={result['otg_top5']:.0%}",
        fontsize=12, fontweight='bold',
    )
    ax.axis('off')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {out_path}")


# ---------------------------------------------------------------------------
# Phase transition plots
# ---------------------------------------------------------------------------

def phase_transition_plots(results, out_dir):
    """Plot OTG properties vs ruggedness parameter (K or nu)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Group by type and config
    nk_data = defaultdict(list)
    wm_data = defaultdict(list)
    for r in results:
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

    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.5 * len(metrics)),
                             sharex='col')

    for row, (metric, denom, ylabel, is_ratio) in enumerate(metrics):
        for col, (data, xlabel, title) in enumerate([
            (nk_data, 'K', 'NK Landscapes'),
            (wm_data, 'ν', 'W-Model'),
        ]):
            ax = axes[row, col]
            params = sorted(data.keys())
            means = []
            stds = []
            for p in params:
                rows = data[p]
                if is_ratio and denom:
                    vals = [r[metric] / max(r[denom], 1) for r in rows]
                else:
                    vals = [r[metric] for r in rows]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            ax.errorbar(params, means, yerr=stds, marker='o', capsize=4,
                        linewidth=2, markersize=6, color='steelblue')
            ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            if row == len(metrics) - 1:
                ax.set_xlabel(xlabel, fontsize=11)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'otg_phase_transition.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved phase transition plot: {path}")

    # Also plot ARI comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, (data, xlabel, title) in enumerate([
        (nk_data, 'K', 'NK Landscapes'),
        (wm_data, 'ν', 'W-Model'),
    ]):
        ax = axes2[col]
        params = sorted(data.keys())
        for metric_key, label, color in [
            ('ari_otg_d1', 'OTG vs LON-d1', 'steelblue'),
            ('ari_otg_d3', 'OTG vs LON-d3', 'darkorange'),
            ('ari_d1_d3', 'LON-d1 vs LON-d3', 'seagreen'),
        ]:
            means = [np.mean([r[metric_key] for r in data[p]]) for p in params]
            ax.plot(params, means, marker='o', label=label, color=color,
                    linewidth=2, markersize=6)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Adjusted Rand Index', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig2.tight_layout()
    path2 = os.path.join(out_dir, 'otg_ari_comparison.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved ARI comparison plot: {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='OTG Funnel Analysis')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/otg_funnel_analysis.json')
    parser.add_argument('--fig-dir', default='results/otg_funnel_figures')
    args = parser.parse_args()

    tasks = []
    for K in [2, 4, 6, 8, 12, 15]:
        for seed in range(args.n_instances):
            tasks.append({
                'type': 'NK', 'config': {'N': 16, 'K': K, 'model': 'adjacent'},
                'seed': seed, 'gamma': args.gamma,
            })
    for nu in [3, 4, 6, 8, 16]:
        for seed in range(args.n_instances):
            tasks.append({
                'type': 'WModel', 'config': {'n': 16, 'nu': nu},
                'seed': seed, 'gamma': args.gamma,
            })

    total = len(tasks)
    workers = min(args.workers, total)

    print(f"OTG Funnel Analysis")
    print(f"  Tasks: {total}")
    print(f"  Workers: {workers}")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    with Pool(workers) as pool:
        completed = 0
        for row in pool.imap_unordered(_analyze_funnels_instance, tasks):
            if row is not None:
                results.append(row)
            completed += 1
            if completed % 20 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed  "
                      f"ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.0f}s")

    # Save results (strip visualization data for compact JSON)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = [{k: v for k, v in r.items() if k != '_vis_data'} for r in results]
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,))
                  else float(o) if isinstance(o, (np.floating,)) else None)
    print(f"Saved {len(results)} results to {out_path}")

    # ---- Summary tables ----
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    # Table 1: Funnel Structure
    print(f"\n{'='*130}")
    print(f"OTG Funnel Structure & LON Comparison")
    print(f"{'='*130}")
    print(f"{'Config':<20} {'#Opt':>6} {'OTG':>5} {'LONd1':>6} {'LONd3':>6} "
          f"{'ARI':>5} {'ARI':>5} {'ARI':>5} {'NMI':>5} "
          f"{'DAG':>4} {'#Cyc':>5} {'%InCyc':>7}")
    print(f"{'':20} {'':>6} {'fun':>5} {'fun':>6} {'fun':>6} "
          f"{'O-d1':>5} {'O-d3':>5} {'d1d3':>5} {'O-d3':>5} "
          f"{'dep':>4} {'':>5} {'':>7}")
    print(f"{'-'*130}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        print(f"{label:<20} "
              f"{np.mean([r['n_local_optima'] for r in rows]):>6.0f} "
              f"{np.mean([r['n_otg_funnels'] for r in rows]):>5.0f} "
              f"{np.mean([r['n_lon_d1_funnels'] for r in rows]):>6.0f} "
              f"{np.mean([r['n_lon_d3_funnels'] for r in rows]):>6.0f} "
              f"{np.mean([r['ari_otg_d1'] for r in rows]):>5.2f} "
              f"{np.mean([r['ari_otg_d3'] for r in rows]):>5.2f} "
              f"{np.mean([r['ari_d1_d3'] for r in rows]):>5.2f} "
              f"{np.mean([r['nmi_otg_d3'] for r in rows]):>5.2f} "
              f"{np.mean([r['dag_depth'] for r in rows]):>4.1f} "
              f"{np.mean([r['n_multi_cycles'] for r in rows]):>5.1f} "
              f"{100*np.mean([r['frac_in_multi_cycles'] for r in rows]):>6.1f}%")

    # Table 2: Funnel Quality Comparison
    print(f"\n{'='*100}")
    print(f"Attractor Quality: OTG vs LON-d1 vs LON-d3")
    print(f"  (Mean terminal rank: 0=best, 0.5=random. Top-5%: fraction of optima reaching top-5%.)")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'OTG Rank':>9} {'d1 Rank':>8} {'d3 Rank':>8} "
          f"{'OTG T5%':>8} {'d1 T5%':>7} {'d3 T5%':>7}")
    print(f"{'-'*100}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        print(f"{label:<20} "
              f"{np.mean([r['otg_mean_rank'] for r in rows]):>9.3f} "
              f"{np.mean([r['d1_mean_rank'] for r in rows]):>8.3f} "
              f"{np.mean([r['d3_mean_rank'] for r in rows]):>8.3f} "
              f"{100*np.mean([r['otg_top5'] for r in rows]):>7.1f}% "
              f"{100*np.mean([r['d1_top5'] for r in rows]):>6.1f}% "
              f"{100*np.mean([r['d3_top5'] for r in rows]):>6.1f}%")

    # Table 3: Compression ratio
    print(f"\n{'='*80}")
    print(f"Landscape Compression: #Funnels / #Optima")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'#Opt':>6} {'OTG':>8} {'LON-d1':>8} {'LON-d3':>8}")
    print(f"{'-'*80}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        nopt = np.mean([r['n_local_optima'] for r in rows])
        otg_r = np.mean([r['n_otg_funnels'] / r['n_local_optima'] for r in rows])
        d1_r = np.mean([r['n_lon_d1_funnels'] / r['n_local_optima'] for r in rows])
        d3_r = np.mean([r['n_lon_d3_funnels'] / r['n_local_optima'] for r in rows])
        print(f"{label:<20} {nopt:>6.0f} {otg_r:>7.1%} {d1_r:>7.1%} {d3_r:>7.1%}")

    # ---- Visualizations ----
    print(f"\nGenerating visualizations...")
    vis_results = [r for r in results
                   if r['_vis_data'] is not None
                   and r['n_local_optima'] >= 20
                   and r['n_local_optima'] <= 500
                   and r['seed'] < 3]
    for r in vis_results:
        fname = (f"otg_funnel_{r['label']}_seed{r['seed']}.png"
                 .replace('(', '').replace(')', '').replace(',', '_').replace('=', ''))
        visualize_otg_funnels(r, os.path.join(args.fig_dir, fname))

    # ---- Phase transition plots ----
    print(f"\nGenerating phase transition plots...")
    phase_transition_plots(results, args.fig_dir)


if __name__ == '__main__':
    freeze_support()
    main()

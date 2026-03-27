#!/usr/bin/env python3
"""
ORC Transition Graph (OTG) Analysis.

Builds a deterministic directed graph over all local optima: each optimum
has exactly one outgoing edge to the optimum reached by following its
min-ORC direction and hill-climbing. Analyzes:

  - Number of sinks (optima that are self-loops or cycle members)
  - Whether the global optimum basin acts as a global attractor
  - Funnel structure: how many distinct attractors exist
  - Comparison with LON-style stochastic transitions
  - Visualization for small instances

Usage:
    python3 benchmarks/otg_analysis.py --workers 150
    python3 benchmarks/otg_analysis.py --workers 150 --visualize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _analyze_otg_instance(args: dict) -> dict:
    """Build and analyze OTG for a single landscape instance."""
    import os, sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.orc_discrete import (
        full_landscape_analysis,
        find_all_local_optima,
        hill_climb,
    )

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
        fitness = landscape.fitness
        neighbor_fn = landscape.neighbor_fn
        space_size = landscape.space_size
        global_opt = landscape.global_optimum()
        label = f"NK(N={config['N']},K={config['K']})"
    elif benchmark_type == 'WModel':
        from src.wmodel import WModel
        landscape = WModel(
            n=config['n'], nu=config['nu'],
            gamma=config.get('gamma', 0),
            mu=config.get('mu', 1), seed=seed,
        )
        fitness = landscape.fitness
        neighbor_fn = landscape.neighbor_fn
        space_size = landscape.space_size
        global_opt = landscape.global_optimum()
        label = f"W(n={config['n']},nu={config['nu']})"
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    t0 = time.perf_counter()

    orc_result = full_landscape_analysis(
        space_size, fitness, neighbor_fn, gamma,
        n_random_trials=30, seed=seed,
    )
    orc_time = time.perf_counter() - t0

    local_optima = orc_result['local_optima']
    basins = orc_result['basin_sizes']
    analyses = orc_result['orc_analyses']
    n_opt = len(local_optima)

    # Build the OTG: each optimum -> its ORC destination
    # dest_opt is the optimum reached by following min-ORC direction
    opt_set = set(local_optima)
    global_opt_minimization = int(np.argmin(fitness))

    otg_edges = {}
    for a in analyses:
        src = a['opt_idx']
        dst = a['dest_opt']
        if dst is None:
            dst = src
        otg_edges[src] = dst

    # ---- OTG reachability analysis ----
    # For each optimum, follow OTG edges repeatedly and track:
    #   - The full path until a cycle is detected
    #   - Whether the global optimum is encountered along the path
    #   - How many hops to reach it

    def trace_path(start):
        """Follow OTG edges, return (path, cycle_nodes)."""
        visited_order = []
        visited_set = set()
        current = start
        while current not in visited_set:
            visited_set.add(current)
            visited_order.append(current)
            nxt = otg_edges.get(current, current)
            if nxt == current:
                return visited_order, {current}
            current = nxt
        # current is the start of a cycle
        cycle_start_idx = visited_order.index(current)
        cycle = set(visited_order[cycle_start_idx:])
        return visited_order, cycle

    # Trace from every optimum
    all_cycles = []
    seen_cycles = set()
    opt_paths = {}
    opt_terminal_cycle = {}
    for opt in local_optima:
        path, cycle = trace_path(opt)
        opt_paths[opt] = path
        opt_terminal_cycle[opt] = cycle
        cycle_key = frozenset(cycle)
        if cycle_key not in seen_cycles:
            seen_cycles.add(cycle_key)
            all_cycles.append(cycle)

    sinks = set()
    for cycle in all_cycles:
        sinks.update(cycle)

    n_sinks = len(sinks)
    n_attractors = len(all_cycles)

    # Global optimum reachability: does the path from each optimum
    # pass through the global optimum at any point?
    hops_to_global = []
    reaches_global_count = 0
    for opt in local_optima:
        path = opt_paths[opt]
        if global_opt_minimization in path:
            idx = path.index(global_opt_minimization)
            hops_to_global.append(idx)
            reaches_global_count += 1
        else:
            # Check if the terminal cycle contains the global optimum
            cycle = opt_terminal_cycle[opt]
            if global_opt_minimization in cycle:
                hops_to_global.append(len(path))
                reaches_global_count += 1

    frac_reach_global = reaches_global_count / n_opt if n_opt > 0 else 0.0
    mean_hops_to_global = float(np.mean(hops_to_global)) if hops_to_global else float('nan')
    median_hops_to_global = float(np.median(hops_to_global)) if hops_to_global else float('nan')

    # Also: which attractor does each optimum converge to?
    # (attractor = the cycle it ends up in)
    global_is_sink = global_opt_minimization in sinks

    terminal_counts = defaultdict(int)
    for opt in local_optima:
        # Use the first element of the terminal cycle as representative
        cycle = opt_terminal_cycle[opt]
        rep = min(cycle)
        terminal_counts[rep] += 1

    # Path lengths to reach any sink node
    path_lengths = []
    for opt in local_optima:
        path = opt_paths[opt]
        length = 0
        for node in path:
            if node in sinks:
                break
            length += 1
        path_lengths.append(length)

    # Distribution of hops to global (for histogram data)
    hops_distribution = {}
    if hops_to_global:
        for h in range(max(hops_to_global) + 1):
            hops_distribution[h] = sum(1 for x in hops_to_global if x == h)
    mean_path_length = float(np.mean(path_lengths)) if path_lengths else 0.0

    # ---- Attractor quality analysis ----
    # For each optimum, what is the fitness of its terminal attractor?
    # Terminal node = the sink/cycle node it converges to.
    # For cycles, use the best (min) fitness node in the cycle.
    global_fitness = float(fitness[global_opt_minimization])
    all_opt_fitnesses = np.array([fitness[o] for o in local_optima])
    sorted_fitnesses = np.sort(all_opt_fitnesses)

    # Fitness of the terminal (best node in terminal cycle) for each optimum
    terminal_fitnesses = []
    fitness_improvements = []
    terminal_ranks = []  # percentile rank of terminal among all optima
    for opt in local_optima:
        cycle = opt_terminal_cycle[opt]
        best_in_cycle = min(cycle, key=lambda c: fitness[c])
        term_fit = float(fitness[best_in_cycle])
        terminal_fitnesses.append(term_fit)

        # Fitness improvement from start to terminal
        start_fit = float(fitness[opt])
        improvement = start_fit - term_fit  # positive = improved (minimization)
        fitness_improvements.append(improvement)

        # Rank of terminal among all optima (0 = best, 1 = worst)
        rank = np.searchsorted(sorted_fitnesses, term_fit) / max(n_opt - 1, 1)
        terminal_ranks.append(float(rank))

    mean_terminal_fitness = float(np.mean(terminal_fitnesses))
    mean_improvement = float(np.mean(fitness_improvements))
    mean_terminal_rank = float(np.mean(terminal_ranks))
    frac_terminal_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in terminal_ranks]))
    frac_terminal_top10 = float(np.mean([1 if r <= 0.10 else 0 for r in terminal_ranks]))
    frac_terminal_top25 = float(np.mean([1 if r <= 0.25 else 0 for r in terminal_ranks]))

    # Fitness range for normalization
    fitness_range = float(sorted_fitnesses[-1] - sorted_fitnesses[0]) if n_opt > 1 else 1.0
    normalized_improvement = mean_improvement / fitness_range if fitness_range > 0 else 0.0

    # ---- Path length distribution to terminal attractor ----
    # For each optimum: how many OTG hops to reach the terminal attractor?
    # This is the "geometric diameter" of the landscape through ORC.
    hops_to_terminal = []
    hops_to_terminal_top5 = []  # conditional: only for those reaching top-5%
    for i, opt in enumerate(local_optima):
        path = opt_paths[opt]
        # path[0] = opt itself. The terminal is reached when we first hit a sink node.
        length = 0
        for node in path:
            if node in sinks:
                break
            length += 1
        hops_to_terminal.append(length)
        if terminal_ranks[i] <= 0.05:
            hops_to_terminal_top5.append(length)

    path_to_terminal_stats = {
        'mean': float(np.mean(hops_to_terminal)),
        'median': float(np.median(hops_to_terminal)),
        'p90': float(np.percentile(hops_to_terminal, 90)),
        'p95': float(np.percentile(hops_to_terminal, 95)),
        'max': int(np.max(hops_to_terminal)),
    }
    if hops_to_terminal_top5:
        path_to_top5_stats = {
            'mean': float(np.mean(hops_to_terminal_top5)),
            'median': float(np.median(hops_to_terminal_top5)),
            'p90': float(np.percentile(hops_to_terminal_top5, 90)),
            'p95': float(np.percentile(hops_to_terminal_top5, 95)),
            'max': int(np.max(hops_to_terminal_top5)),
            'count': len(hops_to_terminal_top5),
        }
    else:
        path_to_top5_stats = {'mean': float('nan'), 'median': float('nan'),
                              'p90': float('nan'), 'p95': float('nan'),
                              'max': 0, 'count': 0}

    # OTG vs MinGap comparison
    mingap_edges = {}
    for a in analyses:
        src = a['opt_idx']
        dst = a.get('min_gap_dest', src)
        if dst is None:
            dst = src
        mingap_edges[src] = dst

    # How often do OTG and MinGap agree on destination?
    n_agree = sum(1 for opt in local_optima
                  if otg_edges.get(opt) == mingap_edges.get(opt))
    frac_agree = n_agree / n_opt if n_opt > 0 else 0.0

    # ---- MinGap Transition Graph for comparison ----
    # Build the same graph using MinGap edges instead of ORC edges
    mg_terminal_ranks = []
    def trace_mingap_path(start):
        visited_set = set()
        current = start
        while current not in visited_set:
            visited_set.add(current)
            nxt = mingap_edges.get(current, current)
            if nxt == current:
                return current
            current = nxt
        return current  # cycle node

    for opt in local_optima:
        mg_term = trace_mingap_path(opt)
        term_fit = float(fitness[mg_term])
        rank = np.searchsorted(sorted_fitnesses, term_fit) / max(n_opt - 1, 1)
        mg_terminal_ranks.append(float(rank))

    mg_mean_terminal_rank = float(np.mean(mg_terminal_ranks))
    mg_frac_top5 = float(np.mean([1 if r <= 0.05 else 0 for r in mg_terminal_ranks]))
    mg_frac_top10 = float(np.mean([1 if r <= 0.10 else 0 for r in mg_terminal_ranks]))
    mg_frac_top25 = float(np.mean([1 if r <= 0.25 else 0 for r in mg_terminal_ranks]))

    # MinGap path length distribution
    mg_hops_to_terminal = []
    mg_hops_to_top5 = []
    # Build MinGap sinks
    mg_all_cycles = []
    mg_seen_cycles = set()
    mg_opt_terminal_cycle = {}
    for opt in local_optima:
        visited_order = []
        visited_set = set()
        current = opt
        while current not in visited_set:
            visited_set.add(current)
            visited_order.append(current)
            nxt = mingap_edges.get(current, current)
            if nxt == current:
                mg_opt_terminal_cycle[opt] = {current}
                break
            current = nxt
        else:
            cycle_start_idx = visited_order.index(current)
            cycle = set(visited_order[cycle_start_idx:])
            mg_opt_terminal_cycle[opt] = cycle
            cycle_key = frozenset(cycle)
            if cycle_key not in mg_seen_cycles:
                mg_seen_cycles.add(cycle_key)
                mg_all_cycles.append(cycle)

    mg_sinks = set()
    for cycle in mg_all_cycles:
        mg_sinks.update(cycle)
    # Include self-loops
    for opt in local_optima:
        if mingap_edges.get(opt, opt) == opt:
            mg_sinks.add(opt)

    for i, opt in enumerate(local_optima):
        visited_set = set()
        current = opt
        length = 0
        while current not in mg_sinks and current not in visited_set:
            visited_set.add(current)
            current = mingap_edges.get(current, current)
            length += 1
        mg_hops_to_terminal.append(length)
        if mg_terminal_ranks[i] <= 0.05:
            mg_hops_to_top5.append(length)

    mg_path_stats = {
        'mean': float(np.mean(mg_hops_to_terminal)),
        'median': float(np.median(mg_hops_to_terminal)),
        'p90': float(np.percentile(mg_hops_to_terminal, 90)),
        'max': int(np.max(mg_hops_to_terminal)),
    }
    if mg_hops_to_top5:
        mg_path_to_top5 = {
            'mean': float(np.mean(mg_hops_to_top5)),
            'median': float(np.median(mg_hops_to_top5)),
            'p90': float(np.percentile(mg_hops_to_top5, 90)),
            'max': int(np.max(mg_hops_to_top5)),
            'count': len(mg_hops_to_top5),
        }
    else:
        mg_path_to_top5 = {'mean': float('nan'), 'median': float('nan'),
                           'p90': float('nan'), 'max': 0, 'count': 0}

    return {
        'label': label,
        'type': benchmark_type,
        'config': config,
        'seed': seed,
        'n_local_optima': n_opt,
        'n_sinks': n_sinks,
        'n_attractors': n_attractors,
        'frac_reach_global': frac_reach_global,
        'mean_hops_to_global': mean_hops_to_global,
        'median_hops_to_global': median_hops_to_global,
        'global_is_sink': global_is_sink,
        'mean_path_length': mean_path_length,
        'frac_orc_mingap_agree': frac_agree,
        'frac_leads_to_better': orc_result['frac_leads_to_better'],
        'frac_mingap_leads_to_better': orc_result['frac_mingap_leads_to_better'],
        'frac_random_leads_to_better': orc_result['frac_random_leads_to_better'],
        # Attractor quality metrics
        'mean_terminal_rank': mean_terminal_rank,
        'frac_terminal_top5': frac_terminal_top5,
        'frac_terminal_top10': frac_terminal_top10,
        'frac_terminal_top25': frac_terminal_top25,
        'normalized_improvement': normalized_improvement,
        # Path length distribution
        'path_to_terminal': path_to_terminal_stats,
        'path_to_top5': path_to_top5_stats,
        # MinGap comparison
        'mg_mean_terminal_rank': mg_mean_terminal_rank,
        'mg_frac_top5': mg_frac_top5,
        'mg_frac_top10': mg_frac_top10,
        'mg_frac_top25': mg_frac_top25,
        'mg_path_to_terminal': mg_path_stats,
        'mg_path_to_top5': mg_path_to_top5,
        'terminal_counts': {int(k): v for k, v in terminal_counts.items()},
        'sink_fitnesses': [float(fitness[s]) for s in sinks],
        'attractor_sizes': [len(c) for c in all_cycles],
        'hops_distribution': hops_distribution,
        'orc_time_s': round(orc_time, 3),
        # Per-optimum data for visualization
        'otg_edges': {int(k): int(v) for k, v in otg_edges.items()},
        'basin_sizes': {int(k): v for k, v in basins.items()},
        'opt_fitnesses': {int(a['opt_idx']): a['opt_fitness']
                          for a in analyses},
    }


def visualize_otg(result: dict, out_dir: str):
    """Visualize OTG for a single instance using networkx + matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx

    otg_edges = result['otg_edges']
    basin_sizes = result['basin_sizes']
    opt_fitnesses = result['opt_fitnesses']

    G = nx.DiGraph()
    nodes = list(otg_edges.keys())
    for src, dst in otg_edges.items():
        G.add_edge(src, dst)

    if len(G.nodes()) == 0:
        return

    sizes = np.array([basin_sizes.get(n, 1) for n in G.nodes()])
    sizes = 100 + 2000 * (sizes / max(sizes.max(), 1))

    fitnesses = np.array([opt_fitnesses.get(str(n), opt_fitnesses.get(n, 0))
                          for n in G.nodes()])
    fit_norm = (fitnesses - fitnesses.min()) / (fitnesses.max() - fitnesses.min() + 1e-12)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42, k=2.0/np.sqrt(len(G.nodes())+1))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color='gray', arrows=True,
                           arrowsize=15, width=1.0,
                           connectionstyle='arc3,rad=0.1')

    scatter = nx.draw_networkx_nodes(G, pos, ax=ax,
                                     node_size=sizes,
                                     node_color=fit_norm,
                                     cmap=plt.cm.RdYlGn_r,
                                     edgecolors='black', linewidths=0.5)

    global_opt = min(opt_fitnesses, key=lambda k: opt_fitnesses[k])
    global_opt = int(global_opt)
    if global_opt in G.nodes():
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               nodelist=[global_opt],
                               node_size=[sizes[list(G.nodes()).index(global_opt)] * 1.5],
                               node_color='gold',
                               edgecolors='black', linewidths=2.0,
                               node_shape='*')

    plt.colorbar(scatter, ax=ax, label='Fitness (darker = worse)', shrink=0.7)
    ax.set_title(f"ORC Transition Graph: {result['label']} seed={result['seed']}\n"
                 f"Optima={result['n_local_optima']}, Sinks={result['n_sinks']}, "
                 f"Attractors={result['n_attractors']}, "
                 f"Reach global={result['frac_reach_global']:.1%}")
    ax.axis('off')

    os.makedirs(out_dir, exist_ok=True)
    fname = f"otg_{result['label']}_seed{result['seed']}.png"
    fname = fname.replace('(', '').replace(')', '').replace(',', '_').replace('=', '')
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description='ORC Transition Graph analysis')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out', default='results/otg_analysis.json')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate OTG visualizations for small instances')
    parser.add_argument('--vis-dir', default='results/otg_figures')
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

    print(f"ORC Transition Graph Analysis")
    print(f"  Tasks: {total}")
    print(f"  Workers: {workers}")
    print(flush=True)

    t_start = time.perf_counter()
    results = []

    if workers <= 1:
        for task in tasks:
            results.append(_analyze_otg_instance(task))
    else:
        with Pool(workers) as pool:
            completed = 0
            for row in pool.imap_unordered(_analyze_otg_instance, tasks):
                results.append(row)
                completed += 1
                if completed % 20 == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / completed * (total - completed)
                    print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed  "
                          f"ETA {eta:.0f}s", flush=True)

    # Save results (strip large per-optimum data for compact output)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = []
    for r in results:
        compact = {k: v for k, v in r.items()
                   if k not in ('otg_edges', 'basin_sizes', 'opt_fitnesses',
                                'sink_fitnesses')}
        save_data.append(compact)

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else
                  float(o) if isinstance(o, (np.floating,)) else None)

    print(f"\nSaved {len(results)} results to {out_path}")

    # Print summary table
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    print(f"\n{'='*140}")
    print(f"{'Config':<20} {'#Opt':>6} {'#Sink':>6} {'Sink%':>6} {'#Attr':>6} "
          f"{'%Reach':>7} {'MnHops':>7} {'MdHops':>7} {'MnPath':>7} "
          f"{'%ORC':>6} {'%MG':>5} {'%Rnd':>5} {'%Agr':>5}")
    print(f"{'='*140}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        nopt = np.mean([r['n_local_optima'] for r in rows])
        nsink = np.mean([r['n_sinks'] for r in rows])
        nattr = np.mean([r['n_attractors'] for r in rows])
        fglob = np.mean([r['frac_reach_global'] for r in rows])
        mn_hops = np.nanmean([r['mean_hops_to_global'] for r in rows])
        md_hops = np.nanmean([r['median_hops_to_global'] for r in rows])
        path = np.mean([r['mean_path_length'] for r in rows])
        orc_b = np.mean([r['frac_leads_to_better'] for r in rows])
        mg_b = np.mean([r['frac_mingap_leads_to_better'] for r in rows])
        rnd_b = np.mean([r['frac_random_leads_to_better'] for r in rows])
        agr = np.mean([r['frac_orc_mingap_agree'] for r in rows])
        print(f"{label:<20} {nopt:>6.0f} {nsink:>6.0f} {100*nsink/nopt:>5.1f}% {nattr:>6.0f} "
              f"{100*fglob:>6.1f}% {mn_hops:>7.2f} {md_hops:>7.1f} {path:>7.2f} "
              f"{100*orc_b:>5.1f}% {100*mg_b:>4.1f}% {100*rnd_b:>4.1f}% {100*agr:>4.1f}%")

    # Reachability summary: what fraction reach global within k hops?
    print(f"\n{'='*100}")
    print(f"OTG Reachability to Global Optimum (fraction of optima reaching global within k hops)")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'%Reach':>7} {'≤0hop':>7} {'≤1hop':>7} {'≤2hop':>7} "
          f"{'≤3hop':>7} {'≤5hop':>7} {'MnHops':>7} {'MdHops':>7} {'MaxHops':>8}")
    print(f"{'-'*100}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        fglob = np.mean([r['frac_reach_global'] for r in rows])
        mn_hops = np.nanmean([r['mean_hops_to_global'] for r in rows])
        md_hops = np.nanmean([r['median_hops_to_global'] for r in rows])

        # Compute cumulative fractions from hops_distribution
        fracs_within = {0: [], 1: [], 2: [], 3: [], 5: []}
        max_hops_list = []
        for r in rows:
            dist = r.get('hops_distribution', {})
            n_opt = r['n_local_optima']
            if not dist:
                for k in fracs_within:
                    fracs_within[k].append(0.0)
                max_hops_list.append(0)
                continue
            # dist keys are hop counts, values are number of optima at that hop count
            max_h = max(int(k) for k in dist.keys())
            max_hops_list.append(max_h)
            total_reaching = sum(dist.values())
            for threshold in fracs_within:
                cum = sum(v for k, v in dist.items() if int(k) <= threshold)
                fracs_within[threshold].append(cum / n_opt if n_opt > 0 else 0.0)

        print(f"{label:<20} {100*fglob:>6.1f}% ", end='')
        for k in [0, 1, 2, 3, 5]:
            print(f"{100*np.mean(fracs_within[k]):>6.1f}% ", end='')
        print(f"{mn_hops:>7.2f} {md_hops:>7.1f} {np.mean(max_hops_list):>8.1f}")

    # Attractor quality table
    print(f"\n{'='*110}")
    print(f"OTG Attractor Quality (what percentile rank does following OTG lead to?)")
    print(f"  Rank 0.0 = global optimum, 0.5 = median optimum. Random baseline = 0.5.")
    print(f"{'='*110}")
    print(f"{'Config':<20} {'OTG Rank':>9} {'MG Rank':>8} "
          f"{'OTG Top5%':>10} {'MG Top5%':>9} "
          f"{'OTG Top10%':>11} {'MG Top10%':>10} "
          f"{'OTG Top25%':>11} {'MG Top25%':>10}")
    print(f"{'-'*110}")

    for label in sorted(groups.keys()):
        rows = groups[label]
        otg_rank = np.mean([r['mean_terminal_rank'] for r in rows])
        mg_rank = np.mean([r['mg_mean_terminal_rank'] for r in rows])
        otg_t5 = np.mean([r['frac_terminal_top5'] for r in rows])
        mg_t5 = np.mean([r['mg_frac_top5'] for r in rows])
        otg_t10 = np.mean([r['frac_terminal_top10'] for r in rows])
        mg_t10 = np.mean([r['mg_frac_top10'] for r in rows])
        otg_t25 = np.mean([r['frac_terminal_top25'] for r in rows])
        mg_t25 = np.mean([r['mg_frac_top25'] for r in rows])
        norm_imp = np.mean([r['normalized_improvement'] for r in rows])
        print(f"{label:<20} {otg_rank:>9.3f} {mg_rank:>8.3f} "
              f"{100*otg_t5:>9.1f}% {100*mg_t5:>8.1f}% "
              f"{100*otg_t10:>10.1f}% {100*mg_t10:>9.1f}% "
              f"{100*otg_t25:>10.1f}% {100*mg_t25:>9.1f}%")

    # Path length to terminal attractor (the headline result)
    print(f"\n{'='*130}")
    print(f"OTG Path Length to Terminal Attractor (all optima)")
    print(f"{'='*130}")
    print(f"{'Config':<20}  {'--- OTG ---':^40}  {'--- MinGap ---':^40}")
    print(f"{'':20}  {'Mean':>7} {'Med':>5} {'P90':>5} {'P95':>5} {'Max':>5}   "
          f"{'Mean':>7} {'Med':>5} {'P90':>5} {'Max':>5}")
    print(f"{'-'*130}")
    for label in sorted(groups.keys()):
        rows = groups[label]
        o_mean = np.mean([r['path_to_terminal']['mean'] for r in rows])
        o_med = np.mean([r['path_to_terminal']['median'] for r in rows])
        o_p90 = np.mean([r['path_to_terminal']['p90'] for r in rows])
        o_p95 = np.mean([r['path_to_terminal']['p95'] for r in rows])
        o_max = np.mean([r['path_to_terminal']['max'] for r in rows])
        m_mean = np.mean([r['mg_path_to_terminal']['mean'] for r in rows])
        m_med = np.mean([r['mg_path_to_terminal']['median'] for r in rows])
        m_p90 = np.mean([r['mg_path_to_terminal']['p90'] for r in rows])
        m_max = np.mean([r['mg_path_to_terminal']['max'] for r in rows])
        print(f"{label:<20}  {o_mean:>7.2f} {o_med:>5.1f} {o_p90:>5.1f} {o_p95:>5.1f} {o_max:>5.1f}   "
              f"{m_mean:>7.2f} {m_med:>5.1f} {m_p90:>5.1f} {m_max:>5.1f}")

    # Conditional: path length for optima whose terminal is top-5%
    print(f"\n{'='*100}")
    print(f"Path Length to Top-5% Attractor (only optima converging to top-5% quality)")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'%Top5':>6}  {'Mean':>6} {'Med':>5} {'P90':>5} {'P95':>5} {'Max':>5}  "
          f"{'MG %T5':>7} {'MG Med':>7} {'MG Max':>7}")
    print(f"{'-'*100}")
    for label in sorted(groups.keys()):
        rows = groups[label]
        otg_t5 = np.mean([r['frac_terminal_top5'] for r in rows])
        # Filter rows that have top5 data
        t5_rows = [r for r in rows if r['path_to_top5']['count'] > 0]
        if t5_rows:
            o_mean = np.mean([r['path_to_top5']['mean'] for r in t5_rows])
            o_med = np.mean([r['path_to_top5']['median'] for r in t5_rows])
            o_p90 = np.mean([r['path_to_top5']['p90'] for r in t5_rows])
            o_p95 = np.mean([r['path_to_top5']['p95'] for r in t5_rows])
            o_max = np.mean([r['path_to_top5']['max'] for r in t5_rows])
        else:
            o_mean = o_med = o_p90 = o_p95 = o_max = float('nan')
        mg_t5 = np.mean([r['mg_frac_top5'] for r in rows])
        mg_rows = [r for r in rows if r['mg_path_to_top5']['count'] > 0]
        if mg_rows:
            m_med = np.mean([r['mg_path_to_top5']['median'] for r in mg_rows])
            m_max = np.mean([r['mg_path_to_top5']['max'] for r in mg_rows])
        else:
            m_med = m_max = float('nan')
        print(f"{label:<20} {100*otg_t5:>5.1f}%  {o_mean:>6.2f} {o_med:>5.1f} {o_p90:>5.1f} "
              f"{o_p95:>5.1f} {o_max:>5.1f}  {100*mg_t5:>6.1f}% {m_med:>7.1f} {m_max:>7.1f}")

    # Visualization for selected small instances
    if args.visualize:
        print(f"\nGenerating OTG visualizations...")
        vis_results = [r for r in results
                       if r['n_local_optima'] > 3
                       and r['n_local_optima'] < 500
                       and r['seed'] < 3]
        for r in vis_results:
            visualize_otg(r, args.vis_dir)


if __name__ == '__main__':
    freeze_support()
    main()

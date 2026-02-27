"""
Publication-Quality Visualization for Perelman-Faithful RSS Results.

Generates:
1. Convergence curves with confidence bands
2. Surgery event timeline
3. Ablation bar chart
4. Curvature evolution plot
5. Critical difference diagram (Nemenyi)
6. Weight ratio heatmap over time (flow evolution)
7. Curvature variance evolution with surgery events
8. 2D landscape overlay with agents and neck regions

Usage:
    python benchmarks/visualize.py --results results/
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('Visualize')

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_convergence_histories(results_dir, algorithms=None):
    """Load convergence histories from per-run CSV files."""
    data = defaultdict(lambda: defaultdict(list))

    for fname in os.listdir(results_dir):
        if not fname.endswith('.csv'):
            continue
        if 'summary' in fname or 'table' in fname:
            continue

        parts = fname.replace('.csv', '').split('_')
        if len(parts) < 4:
            continue

        try:
            algo_parts = []
            func_num = dim = run_id = None
            for p in parts:
                if p.startswith('F') and p[1:].isdigit():
                    func_num = int(p[1:])
                elif p.startswith('D') and p[1:].isdigit():
                    dim = int(p[1:])
                elif p.startswith('run') and p[3:].isdigit():
                    run_id = int(p[3:])
                else:
                    algo_parts.append(p)

            if func_num is None or dim is None or run_id is None:
                continue

            algo = '_'.join(algo_parts) if algo_parts else 'Unknown'
            if algorithms and algo not in algorithms:
                continue

            filepath = os.path.join(results_dir, fname)
            history = []
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    history.append(float(row[1]))

            if history:
                data[algo][(func_num, dim)].append(history)
        except (ValueError, IndexError):
            continue

    return dict(data)


def plot_convergence(data, func_num, dim, output_path, pop_size=50):
    """Plot convergence curves with median + IQR confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for algo in sorted(data.keys()):
        key = (func_num, dim)
        if key not in data[algo]:
            continue

        histories = data[algo][key]
        if not histories:
            continue

        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories])

        median = np.median(aligned, axis=0)
        q25 = np.percentile(aligned, 25, axis=0)
        q75 = np.percentile(aligned, 75, axis=0)

        fe_axis = np.arange(min_len) * pop_size

        median_log = np.log10(np.maximum(median, 1e-10))
        q25_log = np.log10(np.maximum(q25, 1e-10))
        q75_log = np.log10(np.maximum(q75, 1e-10))

        color = colors[color_idx % len(colors)]
        ax.plot(fe_axis, median_log, label=algo, color=color, linewidth=2)
        ax.fill_between(fe_axis, q25_log, q75_log,
                        alpha=0.2, color=color)
        color_idx += 1

    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Log$_{10}$(Error)')
    ax.set_title(f'CEC 2022 F{func_num} (D={dim})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Convergence plot saved to {output_path}")


def plot_ablation_bar(ablation_summary_path, output_path):
    """Generate grouped bar chart for ablation study results."""
    if not os.path.exists(ablation_summary_path):
        logger.warning(f"Ablation summary not found: {ablation_summary_path}")
        return

    configs = []
    funcs = set()
    results = defaultdict(dict)

    with open(ablation_summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            func = int(row['function'])
            mean = float(row['mean'])
            std = float(row['std'])

            if config not in configs:
                configs.append(config)
            funcs.add(func)
            results[config][func] = (mean, std)

    funcs = sorted(funcs)
    n_configs = len(configs)
    n_funcs = len(funcs)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_funcs)
    width = 0.8 / n_configs
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for i, config in enumerate(configs):
        means = []
        stds = []
        for func in funcs:
            if func in results[config]:
                m, s = results[config][func]
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=config, color=colors[i], capsize=3, alpha=0.85)

    ax.set_xlabel('Function')
    ax.set_ylabel('Mean Error')
    ax.set_title('Ablation Study: Component Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels([f'F{f}' for f in funcs])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Ablation bar chart saved to {output_path}")


def plot_critical_difference(avg_ranks, output_path, cd=None):
    """Simple critical difference diagram showing average rankings."""
    if not avg_ranks:
        logger.warning("No ranking data for CD diagram.")
        return

    sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
    n = len(sorted_algos)

    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.5)))

    for i, (algo, rank) in enumerate(sorted_algos):
        y = n - i
        ax.plot([rank], [y], 'ko', markersize=8)
        ax.annotate(f'{algo} ({rank:.2f})',
                    xy=(rank, y), xytext=(10, 0),
                    textcoords='offset points',
                    va='center', fontsize=11)

    if cd is not None:
        best_rank = sorted_algos[0][1]
        ax.plot([best_rank, best_rank + cd], [n + 0.5, n + 0.5],
                'r-', linewidth=2)
        ax.annotate(f'CD={cd:.2f}', xy=(best_rank + cd / 2, n + 0.7),
                    ha='center', fontsize=10, color='red')

    ax.set_xlabel('Average Rank')
    ax.set_title('Algorithm Rankings (Friedman)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0.5, n + 0.5)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Critical difference diagram saved to {output_path}")


# ================================================================== #
#  New Phase 8B: Flow Evolution Visualizations                        #
# ================================================================== #

def plot_weight_ratio_evolution(weight_ratio_log, surgery_log, output_path):
    """
    Plot weight ratio statistics over generations.

    Shows how the max and mean w/w0 ratios evolve, with surgery
    events marked as vertical lines. This visualizes the Ricci flow's
    accumulation and neck pinch formation.

    Args:
        weight_ratio_log: List of (gen, max_ratio, mean_ratio,
                          n_developing, n_strong, n_singular)
        surgery_log: List of (gen, n_edges_cut, ...)
        output_path: Path to save figure.
    """
    if not weight_ratio_log:
        logger.warning("No weight ratio data to plot.")
        return

    gens = [r[0] for r in weight_ratio_log]
    max_ratios = [r[1] for r in weight_ratio_log]
    mean_ratios = [r[2] for r in weight_ratio_log]
    n_developing = [r[3] for r in weight_ratio_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: weight ratios
    ax1.plot(gens, max_ratios, 'r-', linewidth=2, label='Max w/w$_0$')
    ax1.plot(gens, mean_ratios, 'b-', linewidth=1.5, label='Mean w/w$_0$')
    ax1.axhline(y=5.0, color='red', linestyle='--', alpha=0.5,
                label='Singularity threshold')
    ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5,
                label='Developing threshold')

    # Mark surgery events
    for log_entry in surgery_log:
        gen = log_entry[0]
        ax1.axvline(x=gen, color='green', linestyle=':', alpha=0.7)

    ax1.set_ylabel('Weight Ratio (w/w$_0$)')
    ax1.set_title('Ricci Flow Weight Evolution')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: number of stretched edges
    ax2.fill_between(gens, 0, n_developing,
                     alpha=0.5, color='orange', label='Developing (>2x)')
    n_strong = [r[4] for r in weight_ratio_log]
    ax2.fill_between(gens, 0, n_strong,
                     alpha=0.5, color='red', label='Strong (>3.5x)')

    for log_entry in surgery_log:
        gen = log_entry[0]
        ax2.axvline(x=gen, color='green', linestyle=':', alpha=0.7,
                    label='Surgery' if log_entry == surgery_log[0] else '')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Stretched Edges')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Weight ratio evolution saved to {output_path}")


def plot_curvature_evolution(curvature_log, surgery_log, output_path):
    """
    Plot curvature statistics over generations.

    Shows min, mean, and variance of Forman-Ricci curvature,
    with surgery events marked.

    Args:
        curvature_log: List of (gen, min_kappa, mean_kappa, var_kappa)
        surgery_log: List of (gen, n_edges_cut, ...)
        output_path: Path to save figure.
    """
    if not curvature_log:
        logger.warning("No curvature data to plot.")
        return

    gens = [r[0] for r in curvature_log]
    min_kappas = [r[1] for r in curvature_log]
    mean_kappas = [r[2] for r in curvature_log]
    var_kappas = [r[3] for r in curvature_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: min and mean curvature
    ax1.plot(gens, min_kappas, 'b-', linewidth=1.5, alpha=0.8,
             label='Min $\\kappa$')
    ax1.plot(gens, mean_kappas, 'k-', linewidth=2,
             label='Mean $\\kappa$')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    for log_entry in surgery_log:
        gen = log_entry[0]
        ax1.axvline(x=gen, color='green', linestyle=':', alpha=0.7)

    ax1.set_ylabel('Forman-Ricci Curvature')
    ax1.set_title('Curvature Evolution under Ricci Flow')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: curvature variance
    ax2.plot(gens, var_kappas, 'purple', linewidth=2,
             label='Var($\\kappa$)')
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5,
                label='Uniformization threshold')

    for log_entry in surgery_log:
        gen = log_entry[0]
        ax2.axvline(x=gen, color='green', linestyle=':', alpha=0.7,
                    label='Surgery' if log_entry == surgery_log[0] else '')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Curvature Variance')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Curvature evolution saved to {output_path}")


def plot_landscape_overlay(agents, fitness, graph, neck_ghosts,
                           output_path, landscape_func=None,
                           bounds=(-100, 100)):
    """
    2D landscape overlay with agents, graph edges, and neck regions.

    Projects high-dimensional agents to 2D via PCA (or uses first 2 dims).

    Args:
        agents: (N, D) agent positions.
        fitness: (N,) fitness values.
        graph: NetworkX graph with edge weights/curvatures.
        neck_ghosts: List of neck ghost dicts with centroid/direction/radius.
        output_path: Path to save figure.
        landscape_func: Optional callable for background contours.
        bounds: Domain bounds for contour plot.
    """
    if agents.shape[1] > 2:
        # PCA projection to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(agents)
    else:
        coords = agents

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background landscape contours (if function provided and 2D)
    if landscape_func is not None and agents.shape[1] == 2:
        x_grid = np.linspace(bounds[0], bounds[1], 100)
        y_grid = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = landscape_func(np.array([X[i, j], Y[i, j]]))
        ax.contourf(X, Y, Z, levels=30, alpha=0.3, cmap='terrain')
        ax.contour(X, Y, Z, levels=15, alpha=0.2, colors='gray')

    # Draw graph edges
    if graph is not None:
        for u, v, data in graph.edges(data=True):
            if u < len(coords) and v < len(coords):
                w0 = data.get('w0', data.get('weight', 1.0))
                w = data.get('weight', w0)
                ratio = w / max(w0, 1e-10)

                # Color by weight ratio
                if ratio > 5.0:
                    color = 'red'
                    alpha = 0.8
                    lw = 2.0
                elif ratio > 2.0:
                    color = 'orange'
                    alpha = 0.6
                    lw = 1.5
                else:
                    color = 'gray'
                    alpha = 0.3
                    lw = 0.5

                ax.plot([coords[u, 0], coords[v, 0]],
                        [coords[u, 1], coords[v, 1]],
                        color=color, alpha=alpha, linewidth=lw)

    # Draw agents colored by fitness
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=fitness, cmap='viridis_r',
                         s=30, edgecolors='black', linewidths=0.5,
                         zorder=5)
    plt.colorbar(scatter, ax=ax, label='Fitness')

    # Draw neck ghost regions
    if neck_ghosts:
        for neck in neck_ghosts:
            centroid = neck['centroid']
            radius = neck.get('radius', 5.0)

            if agents.shape[1] > 2:
                # Project centroid
                centroid_2d = pca.transform(centroid.reshape(1, -1))[0]
            else:
                centroid_2d = centroid[:2]

            circle = plt.Circle(centroid_2d, radius,
                                fill=False, color='red',
                                linewidth=2, linestyle='--',
                                label='Neck region')
            ax.add_patch(circle)

            if neck.get('direction') is not None:
                direction = neck['direction']
                if agents.shape[1] > 2:
                    dir_2d = pca.transform(direction.reshape(1, -1))[0]
                else:
                    dir_2d = direction[:2]

                dir_norm = np.linalg.norm(dir_2d)
                if dir_norm > 1e-10:
                    dir_2d = dir_2d / dir_norm * radius
                    ax.annotate('', xy=centroid_2d + dir_2d,
                                xytext=centroid_2d,
                                arrowprops=dict(arrowstyle='->',
                                                color='red',
                                                lw=2))

    ax.set_xlabel('Dimension 1' if agents.shape[1] <= 2 else 'PC 1')
    ax.set_ylabel('Dimension 2' if agents.shape[1] <= 2 else 'PC 2')
    ax.set_title('Agent Distribution with Ricci Flow State')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Landscape overlay saved to {output_path}")


def plot_flow_evolution_from_rss(rss, output_dir, prefix='rss'):
    """
    Convenience function to generate all flow visualizations
    from a RiemannianSwarm engine's logged data.

    Args:
        rss: RiemannianSwarm instance with populated logs.
        output_dir: Directory to save figures.
        prefix: Filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Weight ratio evolution
    if rss.weight_ratio_log:
        plot_weight_ratio_evolution(
            rss.weight_ratio_log,
            rss.surgery_log,
            os.path.join(output_dir, f'{prefix}_weight_evolution.png')
        )

    # Curvature evolution
    if rss.curvature_log:
        plot_curvature_evolution(
            rss.curvature_log,
            rss.surgery_log,
            os.path.join(output_dir, f'{prefix}_curvature_evolution.png')
        )

    # Landscape overlay (current state)
    if rss.graph is not None and rss.cached_fitness is not None:
        neck_ghosts = []
        if rss.archive is not None and hasattr(rss.archive, 'neck_ghosts'):
            neck_ghosts = rss.archive.neck_ghosts

        plot_landscape_overlay(
            rss.swarm,
            rss.cached_fitness,
            rss.graph,
            neck_ghosts,
            os.path.join(output_dir, f'{prefix}_landscape_overlay.png')
        )


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--func', type=int, nargs='+', default=[12])
    parser.add_argument('--dim', type=int, nargs='+', default=[10])
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load convergence data
    data = load_convergence_histories(args.results)
    if data:
        logger.info(f"Loaded convergence data for: {list(data.keys())}")

        for func in args.func:
            for dim in args.dim:
                out = os.path.join(args.output,
                                   f'convergence_F{func}_D{dim}.png')
                plot_convergence(data, func, dim, out)

    # Ablation bar chart
    abl_path = os.path.join(args.results, 'ablation_summary.csv')
    if os.path.exists(abl_path):
        plot_ablation_bar(
            abl_path,
            os.path.join(args.output, 'ablation_chart.png')
        )

    # CD diagram
    try:
        from benchmarks.statistics import load_results, friedman_test
        stat_data = load_results(args.results)
        if stat_data:
            _, _, avg_ranks = friedman_test(stat_data)
            if avg_ranks:
                plot_critical_difference(
                    avg_ranks,
                    os.path.join(args.output, 'cd_diagram.png')
                )
    except Exception as e:
        logger.warning(f"Could not generate CD diagram: {e}")


if __name__ == "__main__":
    main()

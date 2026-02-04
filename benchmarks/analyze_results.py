"""
Results Analysis and Visualization for CEC 2022 Benchmarks
Generates publication-ready plots and statistical tests
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import glob

def load_latest_results(results_dir='results'):
    """Load the most recent benchmark results."""
    json_files = glob.glob(f'{results_dir}/cec2022_*.json')
    if not json_files:
        print("No results found. Run run_parallel_cec2022.py first.")
        return None
    
    latest = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    return data

def plot_convergence_curves(data, save_path=None):
    """Plot average convergence curves with confidence intervals."""
    archive_types = list(data['results'].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Log-scale convergence
    ax = axes[0]
    for at in archive_types:
        histories = [r['history'] for r in data['results'][at]]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        
        hist_array = np.array(histories)
        mean_hist = np.mean(hist_array, axis=0)
        std_hist = np.std(hist_array, axis=0)
        
        fe_axis = np.arange(min_len) * 50  # Approximate FE per generation
        
        ax.plot(fe_axis, np.log10(mean_hist), label=f'RSS-{at.upper()}', linewidth=2)
        ax.fill_between(fe_axis, 
                        np.log10(mean_hist - std_hist), 
                        np.log10(mean_hist + std_hist), 
                        alpha=0.3)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Log10(Error)', fontsize=12)
    ax.set_title(f"CEC 2022 F{data['config']['func_num']} (D={data['config']['dim']}) Convergence", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of final errors
    ax = axes[1]
    errors = [np.array([r['final_error'] for r in data['results'][at]]) for at in archive_types]
    labels = [f'RSS-{at.upper()}' for at in archive_types]
    
    bp = ax.boxplot(errors, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Final Error (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f"Distribution of Final Errors (n={data['config']['num_trials']})", fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def wilcoxon_test(data):
    """Perform Wilcoxon signed-rank test between sheaf and none."""
    if 'sheaf' not in data['results'] or 'none' not in data['results']:
        print("Need both 'sheaf' and 'none' results for comparison")
        return
    
    sheaf_errors = np.array([r['final_error'] for r in data['results']['sheaf']])
    none_errors = np.array([r['final_error'] for r in data['results']['none']])
    
    # Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(sheaf_errors, none_errors, alternative='less')
    
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 60)
    print(f"H0: Median difference between Sheaf and None is zero")
    print(f"H1: Sheaf has lower median error than None")
    print(f"\nStatistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"\n✅ RESULT: Statistically significant (p < 0.05)")
        print(f"   Sheaf Archive significantly outperforms baseline")
    else:
        print(f"\n⚠️  RESULT: Not statistically significant (p >= 0.05)")
    
    print("=" * 60)

def print_detailed_stats(data):
    """Print detailed statistics table."""
    print("\n" + "=" * 80)
    print(f"DETAILED STATISTICS - CEC 2022 F{data['config']['func_num']} "
          f"(D={data['config']['dim']}, {data['config']['num_trials']} trials)")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Sheaf':>20} {'No Archive':>20} {'Improvement':>15}")
    print("-" * 80)
    
    for at in ['sheaf', 'none']:
        errors = np.array([r['final_error'] for r in data['results'][at]])
        stats_dict = {
            'Mean': np.mean(errors),
            'Std Dev': np.std(errors),
            'Median': np.median(errors),
            'Best (Min)': np.min(errors),
            'Worst (Max)': np.max(errors),
            '25th Percentile': np.percentile(errors, 25),
            '75th Percentile': np.percentile(errors, 75),
            'Success Rate (<160)': np.mean(errors < 160) * 100
        }
        data['stats'][at] = stats_dict
    
    metrics = ['Mean', 'Median', 'Best (Min)', 'Worst (Max)', 'Std Dev', 
               '25th Percentile', '75th Percentile']
    
    for metric in metrics:
        sheaf_val = data['stats']['sheaf'][metric]
        none_val = data['stats']['none'][metric]
        improvement = (none_val - sheaf_val) / none_val * 100 if none_val != 0 else 0
        print(f"{metric:<25} {sheaf_val:>20.4e} {none_val:>20.4e} {improvement:>14.1f}%")
    
    # Success rate
    sheaf_sr = data['stats']['sheaf']['Success Rate (<160)']
    none_sr = data['stats']['none']['Success Rate (<160)']
    print(f"{'Success Rate (<160)':<25} {sheaf_sr:>19.1f}% {none_sr:>19.1f}%")
    
    print("=" * 80)
    
    # Target comparison
    sheaf_mean = data['stats']['sheaf']['Mean']
    print(f"\nTARGET COMPARISON:")
    print(f"  SOTA (EA4Eig) target: < 160")
    print(f"  RSS-Sheaf achieved:   {sheaf_mean:.1f}")
    
    if sheaf_mean < 160:
        print(f"  ✅ TARGET ACHIEVED - Ready for A* submission!")
    else:
        gap = sheaf_mean - 160
        print(f"  ⚠️  Gap to SOTA: {gap:.1f} (need more tuning)")

def plot_surgical_events(data, save_path=None):
    """Plot where surgical events occurred in successful runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for at in ['sheaf', 'none']:
        histories = [r['history'] for r in data['results'][at]]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        
        # Plot individual trajectories with low alpha
        for hist in histories:
            fe_axis = np.arange(len(hist)) * 50
            ax.plot(fe_axis, np.log10(hist), alpha=0.2, color='blue' if at == 'sheaf' else 'orange')
        
        # Plot mean
        hist_array = np.array(histories)
        mean_hist = np.mean(hist_array, axis=0)
        fe_axis = np.arange(len(mean_hist)) * 50
        ax.plot(fe_axis, np.log10(mean_hist), 
                label=f'RSS-{at.upper()} (Mean)', 
                linewidth=3, 
                color='darkblue' if at == 'sheaf' else 'darkorange')
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Log10(Error)', fontsize=12)
    ax.set_title(f"Convergence Trajectories (n={data['config']['num_trials']})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Load results
    data = load_latest_results()
    
    if data is None:
        sys.exit(1)
    
    # Print statistics
    print_detailed_stats(data)
    
    # Statistical test
    wilcoxon_test(data)
    
    # Generate plots
    func_num = data['config']['func_num']
    dim = data['config']['dim']
    timestamp = data.get('timestamp', 'latest')
    
    plot_convergence_curves(data, f'results/convergence_f{func_num}_d{dim}.png')
    plot_surgical_events(data, f'results/trajectories_f{func_num}_d{dim}.png')
    
    print("\n✅ Analysis complete!")

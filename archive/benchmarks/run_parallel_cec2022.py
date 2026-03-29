"""
Parallel CEC 2022 Benchmark Runner
Optimized for Apple Silicon (M2 Ultra) multi-core execution

Runs 30 independent trials of RSS on CEC 2022 F12 (20D, 30k FEs)
Results are saved to CSV for statistical analysis
"""

import sys
import os
import numpy as np
import multiprocessing as mp
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Ensure path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import opfunu
from benchmarks.rss_optimizer import RSSOptimizer

class CEC2022Wrapper:
    """Wrapper for CEC 2022 functions with error metric."""
    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2022")
        self.problem = problem_class(ndim=dim)
        self.bounds = [-100, 100]
        self.f_bias = self.problem.f_bias
        
    def evaluate(self, x):
        return self.problem.evaluate(x) - self.f_bias

def run_single_trial(args):
    """
    Run a single trial. This function must be at module level for multiprocessing.
    
    Args:
        args: tuple (trial_id, func_num, dim, max_fe, seed, archive_type)
    
    Returns:
        dict: Results including final error, history, and metadata
    """
    trial_id, func_num, dim, max_fe, seed, archive_type = args
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create problem instance
    problem = CEC2022Wrapper(func_num, dim)
    
    # Run RSS optimizer
    opt = RSSOptimizer(
        problem,
        pop_size=50,
        dim=dim,
        max_fe=max_fe,
        archive_type=archive_type
    )
    
    # Run optimization and collect history
    history = opt.run()
    
    # Return results
    return {
        'trial_id': trial_id,
        'func_num': func_num,
        'dim': dim,
        'seed': seed,
        'archive_type': archive_type,
        'final_error': history[-1] if len(history) > 0 else float('inf'),
        'best_error': min(history) if len(history) > 0 else float('inf'),
        'history': history,
        'fe_count': opt.fe_count,
        'num_generations': len(history)
    }

def run_parallel_benchmark(
    func_num=12,
    dim=20,
    max_fe=30000,
    num_trials=30,
    archive_types=['sheaf', 'none'],
    n_workers=None
):
    """
    Run parallel benchmark across multiple cores.
    
    Args:
        func_num: CEC 2022 function number (default: 12)
        dim: Dimensionality (default: 20)
        max_fe: Max function evaluations per trial (default: 30000)
        num_trials: Number of independent trials per configuration (default: 30)
        archive_types: List of archive types to test
        n_workers: Number of parallel workers (default: auto-detect)
    """
    print("=" * 70)
    print(f"PARALLEL CEC 2022 BENCHMARK")
    print(f"Function: F{func_num}, Dimension: {dim}, Max FE: {max_fe}")
    print(f"Trials: {num_trials} per configuration")
    print(f"Configurations: {archive_types}")
    print("=" * 70)
    
    # Auto-detect optimal worker count for M2 Ultra
    if n_workers is None:
        n_workers = min(mp.cpu_count(), num_trials * len(archive_types))
    
    print(f"\nUsing {n_workers} parallel workers (detected {mp.cpu_count()} cores)\n")
    
    # Build task list
    tasks = []
    base_seed = 2024
    
    for archive_type in archive_types:
        for trial in range(num_trials):
            # Unique seed for each trial (deterministic)
            seed = base_seed + trial + (1000 * archive_types.index(archive_type))
            task = (trial, func_num, dim, max_fe, seed, archive_type)
            tasks.append(task)
    
    total_tasks = len(tasks)
    print(f"Total tasks to execute: {total_tasks}\n")
    
    # Execute in parallel
    results = {at: [] for at in archive_types}
    
    with mp.Pool(processes=n_workers) as pool:
        # Use imap_unordered for better progress tracking
        for i, result in enumerate(pool.imap_unordered(run_single_trial, tasks)):
            archive_type = result['archive_type']
            results[archive_type].append(result)
            
            # Progress update (reduced frequency for performance)
            progress = (i + 1) / total_tasks * 100
            if i % 5 == 0 or i == total_tasks - 1:  # Print every 5th task + final
                print(f"[{progress:5.1f}%] Trial {result['trial_id']:2d} ({archive_type:5s}): "
                      f"Final Error = {result['final_error']:.4e}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE - SUMMARY STATISTICS")
    print("=" * 70)
    
    # Compute statistics
    summary_stats = {}
    
    for archive_type in archive_types:
        errors = [r['final_error'] for r in results[archive_type]]
        best_errors = [r['best_error'] for r in results[archive_type]]
        
        stats = {
            'mean_final': np.mean(errors),
            'std_final': np.std(errors),
            'median_final': np.median(errors),
            'min_final': np.min(errors),
            'max_final': np.max(errors),
            'mean_best': np.mean(best_errors),
            'std_best': np.std(best_errors),
            'success_rate': sum(1 for e in errors if e < 160) / len(errors) * 100
        }
        summary_stats[archive_type] = stats
        
        print(f"\n{archive_type.upper()} Archive:")
        print(f"  Mean Final Error:  {stats['mean_final']:.4e} ± {stats['std_final']:.4e}")
        print(f"  Median Final:      {stats['median_final']:.4e}")
        print(f"  Best Achieved:     {stats['min_final']:.4e}")
        print(f"  Worst Achieved:    {stats['max_final']:.4e}")
        print(f"  Success Rate (<160): {stats['success_rate']:.1f}%")
    
    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results as JSON
    json_path = f'{output_dir}/cec2022_f{func_num}_d{dim}_{timestamp}.json'
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for at in archive_types:
            json_results[at] = []
            for r in results[at]:
                jr = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                      for k, v in r.items()}
                json_results[at].append(jr)
        
        json.dump({
            'config': {
                'func_num': func_num,
                'dim': dim,
                'max_fe': max_fe,
                'num_trials': num_trials,
                'archive_types': archive_types
            },
            'summary_stats': summary_stats,
            'results': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Save CSV for easy analysis
    csv_path = f'{output_dir}/cec2022_f{func_num}_d{dim}_{timestamp}.csv'
    with open(csv_path, 'w') as f:
        f.write('trial_id,archive_type,seed,final_error,best_error\n')
        for archive_type in archive_types:
            for r in results[archive_type]:
                f.write(f"{r['trial_id']},{archive_type},{r['seed']},"
                       f"{r['final_error']:.6e},{r['best_error']:.6e}\n")
    
    print(f"CSV saved to: {csv_path}")
    
    return results, summary_stats

if __name__ == "__main__":
    # For macOS, use 'spawn' start method to avoid issues
    mp.set_start_method('spawn', force=True)
    
    # Configuration
    FUNC_NUM = 12      # CEC 2022 F12 (Composition Function 4 - Russian Doll)
    DIM = 20           # 20 dimensions
    MAX_FE = 200000    # 200,000 function evaluations (CEC 2022 standard budget)
    NUM_TRIALS = 30    # 30 independent runs
    
    print(f"\n{'='*70}")
    print(f"⚠️  CEC 2022 STANDARD BUDGET: {MAX_FE:,} FEs")
    print(f"   (Previous 30k was too low for convergence)")
    print(f"{'='*70}\n")
    
    # Run benchmark
    results, stats = run_parallel_benchmark(
        func_num=FUNC_NUM,
        dim=DIM,
        max_fe=MAX_FE,
        num_trials=NUM_TRIALS,
        archive_types=['sheaf', 'none'],  # Compare Sheaf vs No Archive
        n_workers=None  # Auto-detect (will use all cores on M2 Ultra)
    )
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    sheaf_mean = stats['sheaf']['mean_final']
    none_mean = stats['none']['mean_final']
    
    if sheaf_mean < none_mean:
        improvement = (none_mean - sheaf_mean) / none_mean * 100
        print(f"✅ Sheaf Archive WINS by {improvement:.1f}% improvement over baseline")
    else:
        print("⚠️  Sheaf did not outperform baseline in this run")
    
    if sheaf_mean < 160:
        print(f"✅ TARGET ACHIEVED: Mean error {sheaf_mean:.1f} < 160 (SOTA level)")
    else:
        print(f"⚠️  TARGET NOT MET: Mean error {sheaf_mean:.1f} > 160 (need more tuning)")
    
    print("=" * 70)

"""
Comprehensive CEC 2022 Benchmark - All Functions
Tests RSS on all 12 CEC 2022 functions at 20D, 30k FEs
Optimized for parallel execution on Apple Silicon
"""

import sys
import os
import numpy as np
import multiprocessing as mp
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import opfunu
from benchmarks.rss_optimizer import RSSOptimizer

class CEC2022Wrapper:
    """Wrapper for CEC 2022 functions."""
    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        try:
            problem_class = getattr(opfunu.cec_based, f"F{func_num}2022")
            self.problem = problem_class(ndim=dim)
            self.bounds = [-100, 100]
            self.f_bias = self.problem.f_bias
            self.valid = True
        except Exception as e:
            print(f"Warning: Could not load F{func_num}2022: {e}")
            self.valid = False
            
    def evaluate(self, x):
        if not self.valid:
            return float('inf')
        return self.problem.evaluate(x) - self.f_bias

def run_single_experiment(args):
    """Run a single trial on a single function."""
    trial_id, func_num, dim, max_fe, seed, archive_type = args
    
    np.random.seed(seed)
    
    problem = CEC2022Wrapper(func_num, dim)
    if not problem.valid:
        return {
            'trial_id': trial_id,
            'func_num': func_num,
            'dim': dim,
            'seed': seed,
            'archive_type': archive_type,
            'final_error': float('inf'),
            'best_error': float('inf'),
            'history': [],
            'fe_count': 0,
            'error': 'Function load failed'
        }
    
    try:
        opt = RSSOptimizer(
            problem,
            pop_size=50,
            dim=dim,
            max_fe=max_fe,
            archive_type=archive_type
        )
        
        history = opt.run()
        
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
    except Exception as e:
        return {
            'trial_id': trial_id,
            'func_num': func_num,
            'dim': dim,
            'seed': seed,
            'archive_type': archive_type,
            'final_error': float('inf'),
            'best_error': float('inf'),
            'history': [],
            'fe_count': 0,
            'error': str(e)
        }

def run_comprehensive_benchmark(
    func_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    dim=20,
    max_fe=30000,
    num_trials=30,
    archive_types=['sheaf', 'none'],
    n_workers=None
):
    """Run comprehensive benchmark across all CEC 2022 functions."""
    
    print("=" * 80)
    print("COMPREHENSIVE CEC 2022 BENCHMARK - ALL FUNCTIONS")
    print("=" * 80)
    print(f"Functions: {func_nums}")
    print(f"Dimension: {dim}D")
    print(f"Max FE: {max_fe:,}")
    print(f"Trials per function: {num_trials}")
    print(f"Archive types: {archive_types}")
    print("=" * 80)
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(func_nums) * num_trials * len(archive_types))
    
    print(f"\nUsing {n_workers} parallel workers\n")
    
    # Build task list
    tasks = []
    base_seed = 2024
    
    for func_num in func_nums:
        for archive_type in archive_types:
            for trial in range(num_trials):
                seed = base_seed + trial + func_num * 100 + archive_types.index(archive_type) * 1000
                task = (trial, func_num, dim, max_fe, seed, archive_type)
                tasks.append(task)
    
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}\n")
    
    # Execute
    results = {fn: {at: [] for at in archive_types} for fn in func_nums}
    
    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_experiment, tasks)):
            func_num = result['func_num']
            archive_type = result['archive_type']
            results[func_num][archive_type].append(result)
            
            progress = (i + 1) / total_tasks * 100
            if i % 10 == 0 or i == total_tasks - 1:
                print(f"[{progress:5.1f}%] F{result['func_num']:02d} Trial {result['trial_id']:2d} ({archive_type:5s}): "
                      f"Error = {result['final_error']:.4e}")
    
    # Compute statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = {}
    
    for func_num in func_nums:
        print(f"\n--- Function F{func_num:02d} ---")
        summary[func_num] = {}
        
        for archive_type in archive_types:
            errors = [r['final_error'] for r in results[func_num][archive_type] 
                     if 'error' not in r]
            
            if len(errors) == 0:
                continue
                
            stats = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'success_rate': np.mean(np.array(errors) < 100) * 100
            }
            summary[func_num][archive_type] = stats
            
            print(f"  {archive_type.upper():5s}: Mean={stats['mean']:10.4e} ± {stats['std']:8.2e}  "
                  f"Best={stats['min']:10.4e}  Median={stats['median']:10.4e}")
    
    # Overall comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON (Sheaf vs None)")
    print("=" * 80)
    
    sheaf_means = []
    none_means = []
    
    for func_num in func_nums:
        if func_num in summary:
            if 'sheaf' in summary[func_num]:
                sheaf_means.append(summary[func_num]['sheaf']['mean'])
            if 'none' in summary[func_num]:
                none_means.append(summary[func_num]['none']['mean'])
    
    if sheaf_means and none_means:
        overall_sheaf = np.mean(sheaf_means)
        overall_none = np.mean(none_means)
        improvement = (overall_none - overall_sheaf) / overall_none * 100
        
        print(f"\nOverall Mean Error (across all functions):")
        print(f"  Sheaf: {overall_sheaf:.4e}")
        print(f"  None:  {overall_none:.4e}")
        print(f"  Improvement: {improvement:.2f}%")
        
        if overall_sheaf < overall_none:
            print(f"\n✅ Sheaf wins by {improvement:.2f}%")
        else:
            print(f"\n⚠️  Sheaf did not outperform")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = f'{output_dir}/cec2022_all_funcs_d{dim}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json_results = {}
        for fn in func_nums:
            json_results[fn] = {}
            for at in archive_types:
                json_results[fn][at] = []
                for r in results[fn][at]:
                    jr = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in r.items()}
                    json_results[fn][at].append(jr)
        
        json.dump({
            'config': {
                'func_nums': func_nums,
                'dim': dim,
                'max_fe': max_fe,
                'num_trials': num_trials,
                'archive_types': archive_types
            },
            'summary': summary,
            'results': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    return results, summary

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Test ALL CEC 2022 functions (1-12)
    # Note: Some functions may have dimension restrictions
    ALL_FUNCTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    results, summary = run_comprehensive_benchmark(
        func_nums=ALL_FUNCTIONS,
        dim=20,
        max_fe=30000,
        num_trials=30,
        archive_types=['sheaf', 'none'],
        n_workers=None
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

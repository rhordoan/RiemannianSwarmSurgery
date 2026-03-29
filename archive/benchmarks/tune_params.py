"""
Parameter Tuning Script for RSS Optimizer
Tests different combinations to find optimal configuration.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import opfunu
from benchmarks.rss_optimizer import RSSOptimizer
from src.riemannian_swarm import RiemannianSwarm

class CEC2022Wrapper:
    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2022")
        self.problem = problem_class(ndim=dim)
        self.bounds = [-100, 100]
        self.f_bias = self.problem.f_bias
        
    def evaluate(self, x):
        return self.problem.evaluate(x) - self.f_bias

def test_config(k_neighbors, learning_rate, surgery_interval, pop_size, trials=3, max_fe=30000):
    """Test a specific parameter configuration."""
    errors = []
    
    for t in range(trials):
        np.random.seed(t + 2024)
        
        problem = CEC2022Wrapper(12, 10)
        
        # Temporarily override RiemannianSwarm defaults
        original_init = RiemannianSwarm.__init__
        def patched_init(self, agents, dimension, k_neighbors_=k_neighbors, learning_rate_=learning_rate, archive_type='sheaf'):
            original_init(self, agents, dimension, k_neighbors_, learning_rate_, archive_type)
        RiemannianSwarm.__init__ = patched_init
        
        opt = RSSOptimizer(
            problem, 
            pop_size=pop_size, 
            dim=10, 
            max_fe=max_fe, 
            archive_type='sheaf'
        )
        opt.surgery_interval = surgery_interval
        
        # Restore
        RiemannianSwarm.__init__ = original_init
        
        history = opt.run()
        errors.append(history[-1])
    
    return np.mean(errors), np.min(errors), np.std(errors)

def run_tuning():
    print("=" * 60)
    print("RSS Parameter Tuning - CEC 2022 F12")
    print("=" * 60)
    
    # Parameter grid
    configs = [
        # (k_neighbors, learning_rate, surgery_interval, pop_size, label)
        (3, 2.0, 10, 50, "baseline"),
        (3, 2.0, 15, 50, "surgery_15"),
        (3, 2.0, 20, 50, "surgery_20"),
        (3, 1.0, 10, 50, "lr_1.0"),
        (3, 3.0, 10, 50, "lr_3.0"),
        (4, 2.0, 10, 50, "k_4"),
        (5, 2.0, 10, 50, "k_5"),
        (3, 2.0, 10, 80, "pop_80"),
        (3, 2.0, 10, 100, "pop_100"),
    ]
    
    results = []
    
    for k, lr, si, pop, label in configs:
        print(f"\nTesting: {label} (k={k}, lr={lr}, si={si}, pop={pop})")
        mean_err, min_err, std_err = test_config(k, lr, si, pop, trials=3, max_fe=30000)
        results.append((label, mean_err, min_err, std_err))
        print(f"  Mean: {mean_err:.2f}, Best: {min_err:.2f}, Std: {std_err:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (sorted by best error)")
    print("=" * 60)
    results.sort(key=lambda x: x[2])  # Sort by min error
    
    for label, mean, best, std in results:
        print(f"{label:15s}: Best={best:8.2f}, Mean={mean:8.2f}, Std={std:6.2f}")
    
    print(f"\nWINNER: {results[0][0]} with best error {results[0][2]:.2f}")

if __name__ == "__main__":
    run_tuning()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import opfunu
import copy
from benchmarks.rss_optimizer import RSSOptimizer

class CEC2022Wrapper:
    def __init__(self, func_num, dim):
        self.func_num = func_num
        self.dim = dim
        # Instantiate specific CEC class, e.g., F122022
        problem_class = getattr(opfunu.cec_based, f"F{func_num}2022")
        self.problem = problem_class(ndim=dim)
        self.bounds = [-100, 100] # CEC2022 is [-100, 100]
        self.f_bias = self.problem.f_bias  # Store optimal value
        
    def evaluate(self, x):
        # Return ERROR (distance from optimum), not raw value
        return self.problem.evaluate(x) - self.f_bias

def run_cec_experiment(func_num=12, dim=10, trials=3):
    print(f"Starting CEC 2022 F{func_num} Benchmark (Dim={dim}, Trials={trials})")
    
    max_fe = 30000
    modes = ['sheaf', 'none']
    results = {m: [] for m in modes}
    
    for mode in modes:
        print(f"\nRunning Mode: {mode}")
        for t in range(trials):
            # Seed reproducibility
            np.random.seed(t + 2022)
            
            problem = CEC2022Wrapper(func_num, dim)
            
            opt = RSSOptimizer(
                problem, 
                pop_size=50, 
                dim=dim, 
                max_fe=max_fe, 
                archive_type=mode
            )
            
            history = opt.run()
            results[mode].append(history)
            
            final_error = history[-1]
            print(f"  Trial {t}: Final Error = {final_error:.4e}")
            
    # Plotting
    plt.figure(figsize=(10, 6))
    for mode in modes:
        min_len = min(len(h) for h in results[mode])
        avg_hist = np.mean([h[:min_len] for h in results[mode]], axis=0)
        
        fe_axis = np.arange(len(avg_hist)) * 50 # pop_size
        plt.plot(fe_axis, np.log10(np.maximum(avg_hist, 1e-10)), label=f"RSS-{mode.upper()}")
        
    plt.xlabel('Function Evaluations')
    plt.ylabel('Log Error')
    plt.title(f'Convergence on CEC 2022 F{func_num} (D={dim})')
    plt.legend()
    plt.grid(True)
    
    output_path = f'results/cec2022_f{func_num}_d{dim}.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    # F12 is the Composition Function (Russian Doll like)
    run_cec_experiment(func_num=12, dim=10, trials=3)

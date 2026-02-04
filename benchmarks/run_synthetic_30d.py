import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.synthetic_functions import RussianDollFunction
from benchmarks.rss_optimizer import RSSOptimizer

def run_synthetic_30d_experiment(dim=30, trials=3):
    print(f"Starting Synthetic Russian Doll Benchmark (Dim={dim}, Trials={trials})")
    
    problem = RussianDollFunction(dimension=dim)
    max_fe = 30000 
    
    modes = ['sheaf', 'none']
    results = {m: [] for m in modes}
    
    for mode in modes:
        print(f"\nRunning Mode: {mode}")
        for t in range(trials):
            # Seed reproducibility
            np.random.seed(t + 30)
            
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
        
        fe_axis = np.arange(len(avg_hist)) * 50 
        plt.plot(fe_axis, np.log10(np.maximum(avg_hist, 1e-10)), label=f"RSS-{mode.upper()}")
        
    plt.xlabel('Function Evaluations')
    plt.ylabel('Log Error')
    plt.title(f'Convergence on Synthetic Russian Doll (D={dim})')
    plt.legend()
    plt.grid(True)
    
    output_path = f'results/synthetic_d{dim}_30k.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_synthetic_30d_experiment()

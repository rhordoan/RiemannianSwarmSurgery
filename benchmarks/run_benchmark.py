import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.synthetic_functions import RussianDollFunction
from benchmarks.rss_optimizer import RSSOptimizer

def run_experiment(dim=10, trials=5):
    print(f"Starting Benchmark (Dim={dim}, Trials={trials})")
    
    problem = RussianDollFunction(dimension=dim)
    max_fe = 20000 
    
    modes = ['sheaf', 'tabu', 'none']
    results = {m: [] for m in modes}
    
    for mode in modes:
        print(f"\nRunning Mode: {mode}")
        for t in range(trials):
            # Seed reproducibility
            np.random.seed(t + 100)
            
            opt = RSSOptimizer(
                problem, 
                pop_size=30, 
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
        # Average history across trials
        # History lengths might slightly vary due to fe_count check
        min_len = min(len(h) for h in results[mode])
        avg_hist = np.mean([h[:min_len] for h in results[mode]], axis=0)
        
        # Plot
        # Scale x-axis to approx FE (roughly pop_size per step)
        fe_axis = np.arange(len(avg_hist)) * 30 
        plt.plot(fe_axis, np.log10(avg_hist), label=f"RSS-{mode.upper()}")
        
    plt.xlabel('Function Evaluations')
    plt.ylabel('Log Error')
    plt.title(f'Convergence on Russian Doll Synthetic Function (D={dim})')
    plt.legend()
    plt.grid(True)
    
    output_path = f'results/convergence_d{dim}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_experiment(dim=10, trials=5)

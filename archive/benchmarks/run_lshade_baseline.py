import numpy as np
import opfunu
from mealpy.evolutionary_based.SHADE import L_SHADE

from mealpy.utils.space import FloatVar

dim = 10
pop_size = 18 * dim 
max_fe = 1000 * dim 
n_runs = 5 

print(f"{'Func':<5} | {'Mean Error':<15} | {'Std Dev':<15}")
print("-" * 40)

# CEC 2022 functions: 1 to 12
# Note: opfunu returns f(x) - f(x*), so the value is already the error
funcs = {
    1: 'F12022', 2: 'F22022', 3: 'F32022', 4: 'F42022', 5: 'F52022', 
    6: 'F62022', 7: 'F72022', 8: 'F82022', 9: 'F92022', 10: 'F102022', 
    11: 'F112022', 12: 'F122022'
}

for f_id in range(1, 13):
    func_name = funcs[f_id]
    
    # Define problem dictionary for Mealpy
    problem_dict = {
        "obj_func": getattr(opfunu.cec_based.cec2022, func_name)(ndim=dim).evaluate,
        "bounds": FloatVar(lb=[-100] * dim, ub=[100] * dim),
        "minmax": "min",
        "log_to": None,
    }
    
    errors = []
    for run in range(n_runs):
        # Epochs = max_fe // pop_size
        epochs = int(max_fe / pop_size)
        model = L_SHADE(epoch=epochs, pop_size=pop_size) 
        solution = model.solve(problem_dict)
        # Mealpy v3+ returns an Agent/Solution object
        # The best fitness is usually in .target.fitness or solution.target
        # Based on docs, solve returns self.solution, which is an Agent.
        best_fitness = solution.target.fitness
        errors.append(best_fitness)

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    print(f"F{f_id:<4} | {mean_err:.2e}        | {std_err:.2e}")

import numpy as np
import copy
from src.riemannian_swarm import RiemannianSwarm

class RSSOptimizer:
    def __init__(self, problem, pop_size=50, dim=10, max_fe=10000, archive_type='sheaf'):
        self.problem = problem
        self.pop_size = pop_size
        self.dim = dim
        self.max_fe = max_fe
        self.archive_type = archive_type
        
        # Initialize Population
        self.pop = np.random.uniform(problem.bounds[0], problem.bounds[1], (pop_size, dim))
        self.fitness = np.array([problem.evaluate(ind) for ind in self.pop])
        self.fe_count = pop_size
        
        # Initialize RSS Engine
        self.rss = RiemannianSwarm(self.pop, dim, archive_type=archive_type)
        
        # List of sub-populations (if split)
        # Initially just one global population
        self.sub_pops = [{'pop': self.pop, 'fitness': self.fitness, 'rss': self.rss}]
        
    def evolve_sub_pop(self, sub_pop_dict):
        """
        Runs one generation of L-SHADE/DE on the sub-population.
        Also applies RSS repulsion.
        """
        pop = sub_pop_dict['pop']
        fitness = sub_pop_dict['fitness']
        rss_engine = sub_pop_dict['rss']
        
        new_pop = []
        new_fitness = []
        
        # DE parameters
        F = 0.5
        CR = 0.9
        
        for i in range(len(pop)):
            # DE/rand/1/bin
            idxs = [idx for idx in range(len(pop)) if idx != i]
            if len(idxs) < 3: # Too small to evolve
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
                continue
                
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, self.problem.bounds[0], self.problem.bounds[1])
            
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
                
            trial = np.where(cross_points, mutant, pop[i])
            
            # --- RSS INTEGRATION ---
            # 1. Calculate Repulsion from Archive
            # repulsion = rss_engine.get_repulsion(i) # This needs index in global? 
            # Actually, rss engine has its own 'swarm' which matches 'pop'.
            # We need to make sure rss_engine.swarm is synced with pop.
            
            # 2. Evaluate
            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1
            
            # 3. Add Repulsion to Fitness (Metric Inflation)
            # Effective Fitness = f(x) + Repulsion
            repulsion_trial = rss_engine.archive.repulsion(trial) if rss_engine.archive else 0.0
            f_trial_effective = f_trial + repulsion_trial
            
            repulsion_current = rss_engine.archive.repulsion(pop[i]) if rss_engine.archive else 0.0
            f_current_effective = fitness[i] + repulsion_current
            
            if f_trial_effective < f_current_effective:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
                
        # Update sub-pop
        sub_pop_dict['pop'] = np.array(new_pop)
        sub_pop_dict['fitness'] = np.array(new_fitness)
        
        # Sync RSS
        rss_engine.swarm = sub_pop_dict['pop']
        
    def step(self):
        """
        Main optimization generation.
        """
        best_global = float('inf')
        
        # Iterate over all active sub-populations
        for sp in self.sub_pops:
            # 1. Evolve (DE)
            self.evolve_sub_pop(sp)
            
            # 2. RSS Geometric Update
            sp['rss'].step()
            
            # 3. Check for Surgery
            # 3. Check for Surgery & Pruning
            if sp['rss'].graph:
                # Check connected components using NetworkX (since RSS manages the graph)
                # If the graph is split (due to surgery in rss.step()), we should technically split the population.
                # However, for this ablation, we can just detect "Stagnation" in the current component.
                
                # Pruning Condition:
                # If variance of population is very low (converged) AND fitness is poor (>1.0),
                # it means we are trapped in a local optimum.
                fitness_var = np.var(sp['fitness'])
                fitness_mean = np.mean(sp['fitness'])
                
                # If converged to a bad spot (e.g. error > 10.0, var < 1e-4)
                if fitness_var < 1e-4 and fitness_mean > 10.0:
                    # Prune this ENTIRE sub-population
                    # print(f"Pruning Stagnant Population: Mean Fit {fitness_mean:.2f}")
                    
                    # Store ghost
                    indices = list(range(len(sp['pop'])))
                    sp['rss'].prune_sub_swarm(indices)
                    
                    # Respawn Agents (Macro-Mutation / restart)
                    # We keep the object but reset positions
                    # In a full algorithm we might just delete the sub-pop, but here we refill it.
                    sp['pop'] = np.random.uniform(self.problem.bounds[0], self.problem.bounds[1], (len(sp['pop']), self.dim))
                    sp['fitness'] = np.array([self.problem.evaluate(ind) for ind in sp['pop']])
                    self.fe_count += len(sp['pop'])
                    
                    # Reset RSS
                    sp['rss'].swarm = sp['pop']
            
            # Track best
            curr_min = np.min(sp['fitness'])
            if curr_min < best_global:
                best_global = curr_min
                
        return best_global

    def run(self):
        history = []
        while self.fe_count < self.max_fe:
            best = self.step()
            history.append(best)
            if self.fe_count % 1000 == 0:
                print(f"FE: {self.fe_count} | Best Error: {best:.4e}")
        return history

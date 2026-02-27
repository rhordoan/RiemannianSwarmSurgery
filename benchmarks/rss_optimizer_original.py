ï»¿import numpy as np
import copy
import networkx as nx
from src.riemannian_swarm_original import RiemannianSwarm

# Optional CMA-ES for Hunter Mode hybrid
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

# Optional for adaptive strategy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class RSSOptimizer:
    def __init__(self, problem, pop_size=50, dim=10, max_fe=10000, archive_type='sheaf'):
        self.problem = problem
        self.pop_size = pop_size
        self.pop_size_init = pop_size  # LPSR: Initial population size
        self.pop_size_min = 4          # LPSR: Minimum population size (reverted for exploitation)
        self.dim = dim
        self.max_fe = max_fe
        self.archive_type = archive_type
        
        # Surgery Frequency: Only check every N generations
        self.surgery_interval = 15  # Check surgery every 15 generations (tuned)
        self.generation = 0
        
        # ELITE: Calculate max generations for adaptive threshold
        self.max_generations = max_fe // pop_size
        
        # Initialize Population
        self.pop = np.random.uniform(problem.bounds[0], problem.bounds[1], (pop_size, dim))
        self.fitness = np.array([problem.evaluate(ind) for ind in self.pop])
        self.fe_count = pop_size
        
        # Track Best Ever (with solution)
        best_idx = np.argmin(self.fitness)
        self.best_found = self.fitness[best_idx]
        self.best_solution = self.pop[best_idx].copy()  # ELITISM: Store best solution
        
        # Initialize RSS Engine (with max_generations for adaptive threshold)
        self.rss = RiemannianSwarm(self.pop, dim, archive_type=archive_type)
        self.rss.max_generations = self.max_generations
        
        # List of sub-populations (if split)
        # Initially just one global population
        self.sub_pops = [{'pop': self.pop, 'fitness': self.fitness, 'rss': self.rss, 'last_surgery_gen': 0}]
        
    def evolve_sub_pop(self, sub_pop_dict):
        """
        Runs one generation of L-SHADE/DE.
        [UPDATED] Includes Hunter Mode for small surgical squads.
        """
        pop = sub_pop_dict['pop']
        fitness = sub_pop_dict['fitness']
        rss_engine = sub_pop_dict['rss']
        
        # --- HUNTER MODE DETECTION ---
        # CHANGE: Increase threshold from 6 to 15 (30% of initial pop)
        # Any group smaller than this is likely a surgical fragment inside a basin.
        is_hunter_squad = len(pop) <= 15
        
        new_pop = []
        new_fitness = []
        
        # Initialize params if missing
        if 'F' not in sub_pop_dict:
            sub_pop_dict['F'] = np.ones(len(pop)) * 0.5
            sub_pop_dict['CR'] = np.ones(len(pop)) * 0.9
            
        current_F = sub_pop_dict['F']
        current_CR = sub_pop_dict['CR']
        
        updated_F = current_F.copy()
        updated_CR = current_CR.copy()
        
        # Pre-calculate best for Hunter Mode
        best_idx_in_pop = np.argmin(fitness)
        best_agent = pop[best_idx_in_pop]
        
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            
            # Skip if population too small for DE
            if len(idxs) < 3:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
                continue
            
            # --- STRATEGY SWITCHING ---
            if is_hunter_squad:
                # PHASE 1 ENHANCEMENT: CMA-ES Hybrid for Hunter Mode
                # Run CMA-ES once per generation for small populations
                if CMA_AVAILABLE and len(pop) >= 4 and len(new_pop) == 0:
                    try:
                        es = cma.CMAEvolutionStrategy(
                            best_agent, 
                            0.3, 
                            {'verbose': -9, 'verb_log': 0, 'verb_disp': 0, 'maxiter': 5}
                        )
                        # Run for 5 iterations
                        for _ in range(5):
                            if self.fe_count >= self.max_fe:
                                break
                            solutions = es.ask()
                            fitnesses = [self.problem.evaluate(x) for x in solutions]
                            self.fe_count += len(solutions)
                            es.tell(solutions, fitnesses)
                            
                            # Update best
                            best_cma_idx = np.argmin(fitnesses)
                            if fitnesses[best_cma_idx] < self.best_found:
                                self.best_found = fitnesses[best_cma_idx]
                                self.best_solution = solutions[best_cma_idx].copy()
                        
                        # Generate new population from CMA-ES
                        new_solutions = es.ask(len(pop))
                        for sol in new_solutions:
                            sol_clipped = np.clip(sol, self.problem.bounds[0], self.problem.bounds[1])
                            new_pop.append(sol_clipped)
                            new_fitness.append(self.problem.evaluate(sol_clipped))
                            self.fe_count += 1
                        
                        # Apply elitism: keep best individual
                        best_idx = np.argmin(new_fitness)
                        if self.best_found < new_fitness[best_idx]:
                            worst_idx = np.argmax(new_fitness)
                            new_pop[worst_idx] = self.best_solution.copy()
                            new_fitness[worst_idx] = self.best_found
                        
                        sub_pop_dict['pop'] = np.array(new_pop)
                        sub_pop_dict['fitness'] = np.array(new_fitness)
                        rss_engine.swarm = sub_pop_dict['pop']
                        return
                    except:
                        pass  # Fallback to DE if CMA fails
                
        # [ELITE] TOPOLOGICAL SELECTION PROBABILITIES
        # Agents in positive curvature regions (basins) are preferred as parents
        kappas = np.array([rss_engine.get_agent_curvature(idx) for idx in range(len(pop))])
        # Probabilities: higher kappa -> higher weight
        # Use tanh/exp for stable weighting
        probs = np.exp(np.clip(kappas, -5.0, 5.0) / 2.0)
        selection_probs = probs / np.sum(probs)

        # Iterate through population
        for i in range(len(pop)):
            idxs = [j for j in range(len(pop)) if j != i]
            
            if is_hunter_squad:
                # STRATEGY: DE/best/1/bin with Low F (Drilling)
                F_hunt = 0.2  # Fine tuning
                # Need 2 random agents (weighted selection)
                if len(idxs) >= 2:
                    # Adjust selection_probs for the indices available
                    sp_idxs = selection_probs[idxs]
                    sp_idxs = sp_idxs / np.sum(sp_idxs)
                    r1_idx, r2_idx = np.random.choice(idxs, 2, replace=False, p=sp_idxs)
                    r1, r2 = pop[r1_idx], pop[r2_idx]
                    mutant = best_agent + F_hunt * (r1 - r2)
                else:
                    mutant = pop[i] # Fallback
                
                # High CR to preserve good genes
                CRi = 0.9 
                Fi = F_hunt
            else:
                # STRATEGY: DE/rand/1/bin (Explorer - Standard)
                if np.random.rand() < 0.1:
                    Fi = 0.1 + np.random.rand() * 0.9
                else:
                    Fi = current_F[i]
                
                # [ELITE] CURVATURE-AWARE F SCALING
                kappa_i = kappas[i]
                kappa_normalized = np.clip(kappa_i, -10.0, 10.0)
                curvature_boost = 1.0 + 0.2 * np.tanh(-kappa_normalized / 5.0)
                Fi = Fi * curvature_boost
                Fi = np.clip(Fi, 0.1, 1.2)
                
                if np.random.rand() < 0.1:
                    CRi = np.random.rand()
                else:
                    CRi = current_CR[i]
                    
                # Weighted parent selection
                sp_idxs = selection_probs[idxs]
                sp_idxs = sp_idxs / np.sum(sp_idxs)
                a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False, p=sp_idxs)
                a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]
                mutant = a + Fi * (b - c)

            # Boundary handling
            mutant = np.clip(mutant, self.problem.bounds[0], self.problem.bounds[1])
            
            # Crossover
            cross_points = np.random.rand(self.dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            # Evaluate with Repulsion
            f_trial = self.problem.evaluate(trial)
            self.fe_count += 1
            
            repulsion = rss_engine.archive.repulsion(trial) if rss_engine.archive else 0.0
            current_repulsion = rss_engine.archive.repulsion(pop[i]) if rss_engine.archive else 0.0
            
            # Selection (using effective fitness)
            if (f_trial + repulsion) < (fitness[i] + current_repulsion):
                new_pop.append(trial)
                new_fitness.append(f_trial)
                updated_F[i] = Fi
                updated_CR[i] = CRi
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
                # Only keep old params if not hunter
                if not is_hunter_squad:
                    updated_F[i] = current_F[i]
                    updated_CR[i] = current_CR[i]
                    
        sub_pop_dict['F'] = updated_F
        sub_pop_dict['CR'] = updated_CR
        sub_pop_dict['pop'] = np.array(new_pop)
        sub_pop_dict['fitness'] = np.array(new_fitness)
        rss_engine.swarm = sub_pop_dict['pop']
        
    def step(self):
        """
        Main optimization generation with Clean Surgery Logic.
        """
        self.generation += 1
        next_gen_pops = [] # New list to replace self.sub_pops

        # 1. Evolve all current sub-pops
        for sp in self.sub_pops:
            # Check if pop is dead (empty or too small)
            if len(sp['pop']) < 4: # DE needs at least 4 agents for mutation
                 # Archive and skip
                 if sp['rss'].archive: 
                     sp['rss'].prune_sub_swarm(range(len(sp['pop'])))
                 continue

            self.evolve_sub_pop(sp)
            sp['rss'].step() # Ricci Flow

            # Check for splits (handled internally by RSS engine with cooldown/consensus)
            num_components = nx.number_connected_components(sp['rss'].graph) if sp['rss'].graph else 1
            
            if num_components > 1:
                # Reduced print for performance
                if self.generation % 50 == 0:
                    print(f"  [SPLIT @ Gen {self.generation}] Graph has {num_components} components!")
                components = list(nx.connected_components(sp['rss'].graph))
                
                # Keep ALL components as independent sub-populations
                for comp in components:
                    comp_indices = list(comp)
                    
                    # CRITICAL: Validate indices are within bounds
                    max_idx = max(comp_indices) if comp_indices else -1
                    if max_idx >= len(sp['pop']):
                        print(f"  [WARNING] Index {max_idx} >= pop size {len(sp['pop'])}. Skipping split.")
                        next_gen_pops.append(sp)
                        break
                    
                    # Skip very small components (< 4 agents can't do DE)
                    if len(comp_indices) < 4:
                        # Archive tiny component
                        if self.rss.archive:
                            self.rss.archive.store(sp['pop'][comp_indices])
                        continue
                    
                    comp_pop = sp['pop'][comp_indices].copy()  # Defensive copy
                    comp_fitness = sp['fitness'][comp_indices].copy()
                    
                    # Validate fitness values
                    if np.any(comp_fitness < 0):
                        print(f"  [WARNING] Negative fitness detected: {np.min(comp_fitness)}")
                    
                    # Create new RSS engine for this sub-swarm
                    new_rss = RiemannianSwarm(comp_pop, self.dim, archive_type=self.archive_type)
                    new_rss.archive = self.rss.archive # Share global memory
                    
                    new_sp = {
                        'pop': comp_pop, 
                        'fitness': comp_fitness, 
                        'rss': new_rss,
                        'F': sp['F'][comp_indices].copy() if 'F' in sp and sp['F'] is not None else None,
                        'CR': sp['CR'][comp_indices].copy() if 'CR' in sp and sp['CR'] is not None else None,
                        'last_surgery_gen': self.generation # Set cooldown
                    }
                    next_gen_pops.append(new_sp)

            else:
                # No surgery, keep population (check pruning)
                
                # ACTION 1: Loosened Gap Threshold
                # Original: fitness_var < 1.0 (too conservative)
                # New: More sensitive stagnation detection
                fitness_var = np.var(sp['fitness'])
                fitness_mean = np.mean(sp['fitness'])
                fitness_std = np.std(sp['fitness'])
                
                # Loosened threshold: trigger when variance drops below mean - 1.0*std
                # This is more aggressive than the original mean - 2.0*std
                stagnation_detected = (fitness_var < 1.0) or (fitness_std < fitness_mean * 0.01)
                
                if stagnation_detected and fitness_mean > 50.0:
                    # Print removed for performance
                    pass
                    
                    # --- THE LAST STAND ---
                    # Before geometric detonation, check if the best agent is close to a breakthrough.
                    best_idx = np.argmin(sp['fitness'])
                    best_agent = sp['pop'][best_idx]
                    best_val = sp['fitness'][best_idx]
                    
                    # Quick Local Search (50 aggressive local moves)
                    breach_detected = False
                    for _ in range(50):
                        candidate = best_agent + np.random.randn(self.dim) * 0.1
                        candidate = np.clip(candidate, self.problem.bounds[0], self.problem.bounds[1])
                        val = self.problem.evaluate(candidate)
                        self.fe_count += 1
                        
                        if val < best_val:
                            best_agent = candidate
                            best_val = val
                            if best_val < 50.0:
                                # Print removed for performance
                                # print(f"  [LAST STAND] Breached Inner Basin! Error: {best_val:.4e}")
                                self.best_found = min(self.best_found, best_val)
                                self.best_solution = best_agent.copy()
                                sp['pop'][best_idx] = best_agent
                                sp['fitness'][best_idx] = best_val
                                breach_detected = True
                                break
                    
                    if breach_detected:
                        next_gen_pops.append(sp)
                        continue
                    # ---------------------------
                    
                    # ACTION 2: Dynamic Ricci Boost + ACTION 3: Metric Detonation
                    # Instead of random respawn (non-geometric), use inverse Ricci flow
                    # to explosively expand the metric, pushing agents apart
                    
                    # Reduced print for performance
                    # print(f"  [METRIC DETONATION] Triggering Big Bang Protocol...")
                    
                    # Archive the failed region (Ghost Topology)
                    indices = list(range(len(sp['pop'])))
                    sp['rss'].prune_sub_swarm(indices)
                    
                    # GEOMETRIC SOLUTION: Metric Inversion Event
                    # Inject massive negative curvature to force metric expansion
                    # This is a conformal transformation, not random teleportation
                    if sp['rss'].graph is not None and len(sp['rss'].graph.edges()) > 0:
                        # ACTION 2: Ricci Boost - temporarily increase learning rate
                        boost_lr = sp['rss'].learning_rate * 5.0
                        
                        # ACTION 3: Inverse Flow - force all edges to expand (hyperbolic phase)
                        for u, v, data in sp['rss'].graph.edges(data=True):
                            # Artificial negative curvature (-10.0) forces expansion
                            # w_new = w_old * (1 - lambda * (-10.0)) = w_old * (1 + 10*lambda)
                            expansion_factor = 1.0 + boost_lr * 10.0
                            data['weight'] *= expansion_factor
                        
                        # Reduced print for performance
                        # print(f"  [METRIC DETONATION] Space expanded by factor {expansion_factor:.2f}")
                        
                        # The swarm positions remain the same, but the graph metric is now inflated
                        # This effectively makes the optimizer think agents are far apart,
                        # causing DE to make larger moves (geometric mutation)
                        
                        # Reset the surgical memory for this new phase
                        sp['rss'].surgically_cut_edges.clear()
                    
                    # Slight perturbation to help escape (minimal, geometric)
                    # This is NOT random respawn - it's a small nudge after metric expansion
                    perturbation = np.random.randn(len(sp['pop']), self.dim) * 0.5
                    sp['pop'] = sp['pop'] + perturbation
                    sp['pop'] = np.clip(sp['pop'], self.problem.bounds[0], self.problem.bounds[1])
                    sp['fitness'] = np.array([self.problem.evaluate(ind) for ind in sp['pop']])
                    self.fe_count += len(sp['pop'])
                    sp['rss'].swarm = sp['pop']
                    
                    # Reset DE params for exploration phase
                    if 'F' in sp: sp['F'] = np.ones(len(sp['pop'])) * 0.9  # Higher F for exploration
                    if 'CR' in sp: sp['CR'] = np.ones(len(sp['pop'])) * 0.5
                
                next_gen_pops.append(sp)

        self.sub_pops = next_gen_pops
        
        # Aggregation of best result (with sanity check and elitism)
        for sp in self.sub_pops:
            if len(sp['fitness']) > 0:
                best_idx = np.argmin(sp['fitness'])
                b = sp['fitness'][best_idx]
                # Sanity check: errors should be non-negative for CEC functions
                if b >= 0 and b < self.best_found:
                    self.best_found = b
                    self.best_solution = sp['pop'][best_idx].copy()
        
        # ELITISM: Inject best solution into the largest population
        # This ensures we never lose our best discovery
        if len(self.sub_pops) > 0 and self.best_solution is not None:
            largest_sp = max(self.sub_pops, key=lambda x: len(x['pop']))
            if len(largest_sp['pop']) > 1:
                # Replace worst individual with best known
                worst_idx = np.argmax(largest_sp['fitness'])
                if largest_sp['fitness'][worst_idx] > self.best_found:
                    largest_sp['pop'][worst_idx] = self.best_solution.copy()
                    largest_sp['fitness'][worst_idx] = self.best_found
            
        return self.best_found

    def run(self):
        history = []
        while self.fe_count < self.max_fe:
            best = self.step()
            history.append(best)
            
            # LPSR: Linear Population Size Reduction
            # Target population decreases linearly from pop_size_init to pop_size_min
            progress = self.fe_count / self.max_fe
            target_pop = int(self.pop_size_init - (self.pop_size_init - self.pop_size_min) * progress)
            target_pop = max(target_pop, self.pop_size_min)
            
            # Apply LPSR to each sub-population
            for sp in self.sub_pops:
                current_size = len(sp['pop'])
                if current_size > target_pop and current_size > self.pop_size_min:
                    # Remove worst individuals
                    n_remove = current_size - target_pop
                    if n_remove > 0:
                        # Sort by fitness (ascending = best first)
                        sorted_indices = np.argsort(sp['fitness'])
                        keep_indices = sorted_indices[:target_pop]
                        
                        sp['pop'] = sp['pop'][keep_indices]
                        sp['fitness'] = sp['fitness'][keep_indices]
                        sp['rss'].swarm = sp['pop']
                        
                        # Also trim F and CR if present
                        if 'F' in sp and sp['F'] is not None:
                            sp['F'] = sp['F'][keep_indices]
                        if 'CR' in sp and sp['CR'] is not None:
                            sp['CR'] = sp['CR'][keep_indices]
            
            # Reduced print for performance (every 5000 FEs instead of 1000)
            if self.fe_count % 5000 == 0:
                total_agents = sum(len(sp['pop']) for sp in self.sub_pops)
                print(f"FE: {self.fe_count} | Best: {best:.4e} | Pop: {total_agents}")
        return history

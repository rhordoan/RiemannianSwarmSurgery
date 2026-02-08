import numpy as np
import networkx as nx
from scipy.spatial import KDTree

try:
    import gudhi
except ImportError:
    print("Warning: Gudhi not found. Topological scouting will be disabled.")
    gudhi = None

# Note: GraphRicciCurvature needs to be installed. 
# Depending on the version, the import might differ, but we follow the PDF:
try:
    from GraphRicciCurvature.FormanRicci import FormanRicci
except ImportError:
    # Fallback to manual implementation
    FormanRicci = None

from src.sheaf_archive import SheafArchive, TabuArchive

class RiemannianSwarm:
    # CHANGE: k=3 (Fragile), learning_rate=2.5 (Explosive)
    def __init__(self, agents: np.ndarray, dimension: int, k_neighbors: int = 3, learning_rate: float = 2.5, archive_type: str = 'sheaf', multiscale: bool = True):
        """
        initializes the Riemannian Swarm Optimizer.
        
        Args:
            agents (np.ndarray): Initial positions of agents (N x D).
            dimension (int): Dimensionality of the search space.
            k_neighbors (int): Number of neighbors for k-NN graph.
            learning_rate (float): Step size (lambda) for Ricci flow.
            archive_type (str): 'sheaf', 'tabu', or 'none'.
            multiscale (bool): Use multi-scale curvature estimation.
        """
        self.swarm = agents
        self.dimension = dimension
        self.k = k_neighbors
        self.learning_rate = learning_rate
        self.archive_type = archive_type
        self.multiscale = multiscale
        
        if archive_type == 'sheaf':
            self.archive = SheafArchive()
        elif archive_type == 'tabu':
            self.archive = TabuArchive()
        else:
            self.archive = None
            
        self.graph = None
        
        # SURGICAL MEMORY: Track edges that have been permanently cut by surgery
        # This prevents the graph from "healing" itself in subsequent generations
        # Keys are (min_node_id, max_node_id) tuples to ensure consistent ordering
        self.surgically_cut_edges = set()
        
        # MULTISCALE: Store graphs at multiple scales for robust curvature
        self.multiscale_graphs = {}
        self.multiscale_kappas = {}
        
        # ADAPTIVE THRESHOLD: Generation tracking for dynamic surgery threshold
        self.generation = 0
        self.max_generations = 1000  # Default, will be updated by optimizer
        
        # SURGERY COOLDOWN: Prevent rapid-fire surgery for consistency
        self.last_surgery_gen = -100
        self.surgery_cooldown = 30  # Initial cooldown in generations
        self.surgery_count = 0      # Used for exponential backoff
        
    def build_knn_graph(self, points: np.ndarray) -> nx.Graph:

        """
        Builds a k-Nearest Neighbor graph with Euclidean edge weights.
        Respects surgically cut edges - they will NOT be re-added.
        """
        N = len(points)
        k = min(self.k, N - 1)
        tree = KDTree(points)
        
        # Query k+1 because the point itself is included
        distances, indices = tree.query(points, k=k+1)
        
        G = nx.Graph()
        
        # Add nodes
        for i in range(N):
            G.add_node(i, pos=points[i])
            
        # Add edges (respecting surgical cuts)
        for i in range(N):
            for j_idx in range(1, k+1): # Skip self (index 0)
                neighbor = indices[i][j_idx]
                
                # Check if this edge has been surgically cut
                # Use canonical ordering (min, max) for consistent lookup
                edge_key = (min(i, neighbor), max(i, neighbor))
                if edge_key in self.surgically_cut_edges:
                    continue  # Skip this edge - it was surgically severed
                
                dist = max(distances[i][j_idx], 1e-6)  # Clamp to avoid zero weights
                # We use specific attribute for GraphRicciCurvature if needed, 
                # but 'weight' is standard for NetworkX
                G.add_edge(i, neighbor, weight=dist)
                
        return G

    def step(self):
        """
        Main execution step for one generation of RSS.
        """
        # Increment generation for adaptive threshold
        self.generation += 1
        
        # [ELITE] ADAPTIVE k: Neighborhood size decays over time
        # Early: High k (global view, stable curvature)
        # Late: Low k (fine structure, precise surgery)
        progress = min(self.generation / max(self.max_generations, 1), 1.0)
        k_max = 12
        k_min = 3
        self.k = int(k_max - progress * (k_max - k_min))
        
        # 1. Build k-NN Graph
        self.graph = self.build_knn_graph(self.swarm)
        
        # 2. Compute Forman-Ricci Curvature (Multi-scale if enabled)
        if self.multiscale:
            self.build_multiscale_graphs(self.swarm)
            self.compute_multiscale_curvature()
        else:
            self.compute_manual_curvature()
        
        # DEBUG: Print curvature stats less frequently
        kappas = [d.get('ricciCurvature', 0) for u, v, d in self.graph.edges(data=True)]
        # if kappas:
        #     print(f"  [CURVATURE] Min: {np.min(kappas):.4f}, Max: {np.max(kappas):.4f}, Mean: {np.mean(kappas):.4f}")

        
        # 3. Discrete Ricci Flow (Metric Evolution)
        # Iterate over edges and update weights based on curvature
        for u, v, data in self.graph.edges(data=True):
            if 'ricciCurvature' in data:
                kappa = data['ricciCurvature']
                
                # Metric update rule: w_{t+1} = w_t * (1 - lambda * kappa)
                # If kappa < 0 (bridge): factor > 1 -> weight increases (distance expands)
                factor = 1.0 - self.learning_rate * kappa
                
                # Clamp factor to avoid negative weights or explosion
                factor = max(0.01, factor) 
                
                data['weight'] *= factor
                
        # 4. Topological Scouting (Persistent Homology)
        if gudhi is not None:
            # 5. Surgery Check (The "Cut")
            # Enforce cooldown to let sub-swarms stabilize
            is_cooled_down = (self.generation - self.last_surgery_gen) >= self.surgery_cooldown
            
            if is_cooled_down:
                # Detect loops (H1) and necks (H0 merges)
                rips = gudhi.RipsComplex(points=self.swarm)
                simplex_tree = rips.create_simplex_tree(max_dimension=2)
                barcode = simplex_tree.persistence()
                
                if self.detect_singularity(barcode, self.graph):
                    sub_swarms = self.perform_surgery(self.graph)
                    self.manage_sub_swarms(sub_swarms)

    def compute_manual_curvature(self):
        """
        Vectorized-ish implementation of Weighted Forman-Ricci.
        O(E) instead of O(E*k) in Python loops.
        """
        G = self.graph
        # Pre-fetch weights to avoid dictionary lookups in loop
        edge_weights = nx.get_edge_attributes(G, 'weight')
        
        curvatures = {}
        
        # Iterate edges (this loop is inevitable without adjacency matrix ops)
        for (u, v), w_e in edge_weights.items():
            # Approx 2: Weighted (Your implementation, optimized)
            sum_u = 0.0
            sum_v = 0.0
            
            # Use G[u] iterator which is faster than G.neighbors(u)
            for nbr, attr in G[u].items():
                if nbr != v:
                    sum_u += np.sqrt(w_e / attr['weight'])
            
            for nbr, attr in G[v].items():
                if nbr != u:
                    sum_v += np.sqrt(w_e / attr['weight'])
            
            f_e = 2.0 - sum_u - sum_v
            
            # AFRC (Triangles)
            tris = len(list(nx.common_neighbors(G, u, v)))
            f_e += 3.0 * tris
            
            curvatures[(u, v)] = f_e

        nx.set_edge_attributes(G, curvatures, 'ricciCurvature')
    
    def get_agent_curvature(self, agent_idx: int) -> float:
        """
        [ELITE] Returns the mean curvature of edges connected to an agent.
        
        Used for curvature-aware DE mutation:
        - Negative curvature (bridge): Agent is on a bottleneck → boost F to escape
        - Positive curvature (basin): Agent is in a good region → lower F to refine
        
        Args:
            agent_idx: Index of the agent in the swarm
            
        Returns:
            Mean curvature of incident edges, or 0.0 if no edges
        """
        if self.graph is None or agent_idx not in self.graph:
            return 0.0
        
        kappas = []
        for neighbor in self.graph.neighbors(agent_idx):
            edge_data = self.graph.get_edge_data(agent_idx, neighbor)
            if edge_data and 'ricciCurvature' in edge_data:
                kappas.append(edge_data['ricciCurvature'])
        
        if not kappas:
            return 0.0
        
        return np.mean(kappas)
    
    def build_multiscale_graphs(self, points: np.ndarray, scales: list = None):
        """
        Build k-NN graphs at multiple scales for robust curvature estimation.
        Helps with high-frequency landscapes like Rastrigin (F09).
        """
        if scales is None:
            scales = [3, 5, 10]  # Small, medium, large neighborhoods
        
        self.multiscale_graphs = {}
        for k in scales:
            k_adj = min(k, len(points) - 1)
            if k_adj > 0:
                # Temporarily set k for build
                orig_k = self.k
                self.k = k_adj
                self.multiscale_graphs[k] = self.build_knn_graph(points)
                self.k = orig_k
    
    def compute_multiscale_curvature(self):
        """
        Compute curvature at multiple scales and combine weighted by scale.
        [ELITE] Also stores per-scale curvatures for consensus-based surgery.
        """
        if not self.multiscale or not self.multiscale_graphs:
            self.compute_manual_curvature()
            return
        
        # Compute curvature for each scale
        all_curvatures = {}
        self.multiscale_kappas = {}  # Reset per-scale storage
        weights = {3: 0.5, 5: 0.3, 10: 0.2}  # Emphasize local structure
        
        for k, G in self.multiscale_graphs.items():
            edge_weights = nx.get_edge_attributes(G, 'weight')
            curvatures = {}
            scale_kappas = {}  # Per-scale curvature for this k
            
            for (u, v), w_e in edge_weights.items():
                sum_u = 0.0
                sum_v = 0.0
                
                for nbr, attr in G[u].items():
                    if nbr != v:
                        sum_u += np.sqrt(w_e / attr['weight'])
                
                for nbr, attr in G[v].items():
                    if nbr != u:
                        sum_v += np.sqrt(w_e / attr['weight'])
                
                f_e = 2.0 - sum_u - sum_v
                tris = len(list(nx.common_neighbors(G, u, v)))
                f_e += 3.0 * tris
                
                key = (min(u, v), max(u, v))
                scale_kappas[key] = f_e  # Store for consensus
                
                if key not in all_curvatures:
                    all_curvatures[key] = {}
                all_curvatures[key][k] = f_e
            
            self.multiscale_kappas[k] = scale_kappas  # Store per-scale kappas
        
        # Combine curvatures weighted by scale
        final_curvatures = {}
        for edge, kappas in all_curvatures.items():
            combined = 0.0
            total_weight = 0.0
            for k, kappa in kappas.items():
                w = weights.get(k, 0.1)
                combined += kappa * w
                total_weight += w
            if total_weight > 0:
                final_curvatures[edge] = combined / total_weight
        
        # Apply to main graph
        nx.set_edge_attributes(self.graph, final_curvatures, 'ricciCurvature')


        
    def detect_singularity(self, barcode, graph):
        """
        Refined Detection: Triggers if we have significant negative curvature
        relative to the graph's average.
        """
        # 1. Metric Singularity (Dynamic Curvature check)
        kappas = [d['ricciCurvature'] for u, v, d in graph.edges(data=True) if 'ricciCurvature' in d]
        
        if not kappas: 
            return False
        
        # Calculate statistics
        min_k = np.min(kappas)
        # avg_k = np.mean(kappas)
        
        # TRIGGER CONDITION:
        # If the most negative edge is significantly lower than the average (an outlier bridge)
        # OR if we just have raw negative curvature accumulation.
        
        # Fix: Lower the barrier. If we have edges < -1.0, we probably have a neck.
        if min_k < -1.0:
            # Reduced print for performance
            # print(f"  [DEBUG] Singularity Detected: Min Kappa {min_k:.4f}")
            return True
            
        # 2. Topological Loop (Keep existing)
        for dim, (birth, death) in barcode:
            if dim == 1:
                lifespan = death - birth
                if death == float('inf'):
                    return True
                if lifespan > 1.0:
                    return True
                    
        return False

    def perform_surgery(self, graph):
        """
        [ELITE] Adaptive Surgery with Multi-Scale Consensus.
        
        Two key enhancements:
        1. ADAPTIVE THRESHOLD: Percentile-based, generation-aware decay
           - Early (exploration): aggressive cuts (bottom 5%)
           - Late (exploitation): conservative cuts (bottom 20%)
        
        2. MULTI-SCALE CONSENSUS: Cut only if edge is stressed at ALL scales
           - Requires agreement from k=3, k=5, and k=10 graphs
           - Eliminates false positives from single-scale noise
        
        [CRITICAL] Tracks surgically cut edges to prevent graph "healing".
        """
        # Get all curvatures from main graph
        edges = list(graph.edges(data=True))
        kappas = np.array([d.get('ricciCurvature', 0.0) for u, v, d in edges])
        
        if len(kappas) == 0:
            return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        
        # === ADAPTIVE THRESHOLD ===
        # Progress: 0.0 (start) → 1.0 (end)
        progress = min(self.generation / max(self.max_generations, 1), 1.0)
        
        # Percentile decays from 5% (aggressive) → 20% (conservative)
        percentile = 5 + progress * 15
        
        # Calculate dynamic threshold from percentile
        base_threshold = np.percentile(kappas, percentile)
        
        # Floor: Never cut positive curvature edges
        cut_threshold = min(base_threshold, -0.01)
        
        # === MULTI-SCALE CONSENSUS ===
        # Find edges that are stressed at ALL scales (intersection)
        consensus_edges = None
        
        if self.multiscale and self.multiscale_kappas:
            for k, scale_kappas in self.multiscale_kappas.items():
                # Get edges stressed at this scale
                stressed_at_k = set()
                if scale_kappas:
                    # Use same percentile-based threshold for each scale
                    scale_vals = list(scale_kappas.values())
                    if scale_vals:
                        scale_threshold = np.percentile(scale_vals, percentile)
                        scale_threshold = min(scale_threshold, -0.01)
                        
                        for edge, kappa in scale_kappas.items():
                            if kappa < scale_threshold:
                                stressed_at_k.add(edge)
                
                # Intersection: edge must be stressed at ALL scales
                if consensus_edges is None:
                    consensus_edges = stressed_at_k
                else:
                    consensus_edges = consensus_edges & stressed_at_k
        
        # Build edges_to_cut list
        edges_to_cut = []
        
        if consensus_edges is not None and len(consensus_edges) > 0:
            # USE CONSENSUS: Only cut edges that ALL scales agree on
            for u, v, data in edges:
                edge_key = (min(u, v), max(u, v))
                if edge_key in consensus_edges:
                    edges_to_cut.append((u, v))
        else:
            # FALLBACK: Use adaptive threshold on combined curvature
            for u, v, data in edges:
                if data.get('ricciCurvature', 0.0) < cut_threshold:
                    edges_to_cut.append((u, v))
                
        if edges_to_cut:
            # TRIAL CUT: Test if the resulting components are large enough
            test_graph = graph.copy()
            test_graph.remove_edges_from(edges_to_cut)
            components = list(nx.connected_components(test_graph))
            
            valid_components_nodes = []
            MIN_AGENTS = 10  # Viability Threshold
            
            for comp in components:
                if len(comp) >= MIN_AGENTS:
                    valid_components_nodes.append(comp)
            
            # Surgery is only valid if we have at least 2 viable sub-swarms
            if len(valid_components_nodes) >= 2:
                graph.remove_edges_from(edges_to_cut)
                
                # CRITICAL: Remember these edges are surgically cut
                for u, v in edges_to_cut:
                    edge_key = (min(u, v), max(u, v))
                    self.surgically_cut_edges.add(edge_key)
                
                # [ELITE] BACKOFF: Update last surgery gen and backoff cooldown
                self.last_surgery_gen = self.generation
                self.surgery_count += 1
                # Exponential backoff: cooldown grows with each surgery
                # 30 -> 60 -> 120 -> 240...
                self.surgery_cooldown = 30 * (2 ** (self.surgery_count - 1))
                # Cap cooldown at 500 generations
                self.surgery_cooldown = min(self.surgery_cooldown, 500)
                
                # Debug output (enabled for elite mode)
                print(f"  [ELITE SURGERY] Gen {self.generation}: Cut {len(edges_to_cut)} edges")
                print(f"    Cooldown now: {self.surgery_cooldown} gens, Consensus: {consensus_edges is not None}")
                return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
            else:
                # Cancel surgery: fragments too small
                return [graph]
            
        return [graph]
        
    def manage_sub_swarms(self, sub_swarms):
        """
        'Cap' protocol: Re-initializes metric for sub-swarms.
        """
        # In this architecture, 'self.graph' represents the global state.
        # If surgery split the graph, 'self.graph' is now disconnected.
        # We don't necessarily need to replace self.graph with a list of graphs 
        # unless we explicitly parallelize the loop object-oriented style.
        # For now, we update the global graph's edge weights in the new components.
        
        # Re-stabilize: Reset edge weights to Euclidean distance to stop hyperbolic expansion?
        # PDF: "Metric Re-initialization... prevents the sub-swarm from trying to cross the now-severed bridge."
        # Basically, we just reset the weights of the *remaining* edges to their current Euclidean distance, 
        # effectively forgetting the warped history, so they can start fresh in the basin.
        
        # Since self.graph is already modified (edges cut), we iterate over remaining edges.
        
        positions = self.swarm # Global indices mapping
        
        for sub_g in sub_swarms:
            # Re-calculate weights for this component
            for u, v in sub_g.edges():
                # We need original indices to get positions
                # NetworkX nodes are integers if we built it that way
                p1 = self.swarm[u]
                p2 = self.swarm[v]
                dist = np.linalg.norm(p1 - p2)
                
                if self.graph.has_edge(u, v):
                    self.graph[u][v]['weight'] = dist

    def prune_sub_swarm(self, sub_swarm_nodes):
        """
        Archives a sub-swarm (e.g., if it converged to a poor local optimum)
        and removes it from the active population.
        """
        if self.archive is None:
            return
            
        # Get actual coordinates
        # NetworkX nodes are indices 0..N
        points = self.swarm[list(sub_swarm_nodes)]
        self.archive.store(points)
    
    def get_surgical_memory(self):
        """Returns the set of surgically cut edges (for transfer to sub-swarms)."""
        return self.surgically_cut_edges.copy()
    
    def set_surgical_memory(self, cut_edges):
        """Sets the surgical cut memory (used when creating sub-swarms)."""
        self.surgically_cut_edges = cut_edges.copy()
        
        # In a real optimizer, we would now respawn these agents or mark them inactive.
        # For now, we just store the ghost to ensure the archive logic works.
        
    def get_repulsion(self, agent_idx):
        """
        Calculates repulsion penalty for a specific agent based on archive.
        """
        if self.archive is None:
            return 0.0
            
        pos = self.swarm[agent_idx]
        return self.archive.repulsion(pos)



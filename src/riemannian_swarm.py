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
    # Fallback or specific handling if the library isn't standard in the environment
    print("Warning: GraphRicciCurvature not found. Curvature computation will fail.")
    FormanRicci = None

from src.sheaf_archive import SheafArchive

class RiemannianSwarm:
    def __init__(self, agents: np.ndarray, dimension: int, k_neighbors: int = 10, learning_rate: float = 0.1):
        """
        initializes the Riemannian Swarm Optimizer.
        
        Args:
            agents (np.ndarray): Initial positions of agents (N x D).
            dimension (int): Dimensionality of the search space.
            k_neighbors (int): Number of neighbors for k-NN graph.
            learning_rate (float): Step size (lambda) for Ricci flow.
        """
        self.swarm = agents
        self.dimension = dimension
        self.k = k_neighbors
        self.learning_rate = learning_rate
        
        self.archive = SheafArchive()
        self.graph = None
        
    def build_knn_graph(self, points: np.ndarray) -> nx.Graph:
        """
        Builds a k-Nearest Neighbor graph with Euclidean edge weights.
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
            
        # Add edges
        for i in range(N):
            for j_idx in range(1, k+1): # Skip self (index 0)
                neighbor = indices[i][j_idx]
                dist = distances[i][j_idx]
                # We use specific attribute for GraphRicciCurvature if needed, 
                # but 'weight' is standard for NetworkX
                G.add_edge(i, neighbor, weight=dist)
                
        return G

    def step(self):
        """
        Main execution step for one generation of RSS.
        """
        # 1. Build k-NN Graph
        self.graph = self.build_knn_graph(self.swarm)
        
        # 2. Compute Forman-Ricci Curvature
        if FormanRicci is not None:
             # Uses combinatorial formula from library if available
            orc = FormanRicci(self.graph)
            orc.compute_ricci_curvature()
        else:
            # Fallback: Manual implementation matches Sreejith et al. formula (PDF Pg 6)
            self.compute_manual_curvature()
        
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
            # Detect loops (H1) and necks (H0 merges)
            rips = gudhi.RipsComplex(points=self.swarm)
            simplex_tree = rips.create_simplex_tree(max_dimension=2)
            barcode = simplex_tree.persistence()
            
            # 5. Surgery Check (The "Cut")
            # if self.detect_singularity(barcode, self.graph):
            #     sub_swarms = self.perform_surgery(self.graph)
            #     self.manage_sub_swarms(sub_swarms)

    def compute_manual_curvature(self):
        """
        Manual implementation of Weighted Forman-Ricci Curvature.
        Formula (assuming node weights = 1):
        F(e) = 2 - sum_{e1~v1, e1!=e} sqrt(w_e / w_e1) - sum_{e2~v2, e2!=e} sqrt(w_e / w_e2)
        """
        # Pre-calculate curvature for all edges
        curvatures = {}
        for u, v, data in self.graph.edges(data=True):
            w_e = data['weight']
            
            # Sum for u side
            sum_u = 0.0
            for neighbor in self.graph.neighbors(u):
                if neighbor == v: continue
                w_e1 = self.graph[u][neighbor]['weight']
                sum_u += np.sqrt(w_e / w_e1)
                
            # Sum for v side
            sum_v = 0.0
            for neighbor in self.graph.neighbors(v):
                if neighbor == u: continue
                w_e2 = self.graph[v][neighbor]['weight']
                sum_v += np.sqrt(w_e / w_e2)
            
            # F(e) = 4 - deg(u) - deg(v) in unweighted case, but here we use weighted form
            # The simplified weighted form derived above:
            f_e = 2.0 - sum_u - sum_v
            
            # Augmented Forman-Ricci Curvature (AFRC): Add contributions from triangles
            # 3 * Number of triangles containing e
            # A triangle (u, v, w) exists if w is a neighbor of both u and v
            common_neighbors = len(list(nx.common_neighbors(self.graph, u, v)))
            f_e += 3.0 * common_neighbors
            
            curvatures[(u, v)] = f_e
            
        # Apply to graph
        nx.set_edge_attributes(self.graph, curvatures, 'ricciCurvature')


        
    def detect_singularity(self, barcode, graph):
        """
        Detects topological singularities requiring surgery.
        Criterion 1: Persistent Beta_1 loop (lifespan > threshold)
        Criterion 2: Bridge edge with highly negative curvature
        """
        # 1. Topological Loop Detection (Beta 1)
        # Barcode format: list of (dimension, (birth, death))
        for dim, (birth, death) in barcode:
            if dim == 1:
                lifespan = death - birth
                # Threshold logic: if loop persists relative to scale
                # For now, simple constant or relative to diameter necessary
                # PDF mentions death/birth > T_loop, but death can be infinity
                if death == float('inf'):
                    return True # Permanent feature?
                if lifespan > 1.0: # Simplistic threshold, should be dynamic
                    # print(f"Detected loop with lifespan {lifespan}")
                    return True

        # 2. Metric Singularity (Curvature check)
        # Check if any edge has extremely negative curvature
        # This is implicitly handled by perform_surgery looking for candidates,
        # but here we return True to trigger the attempt.
        min_kappa = float('inf')
        for u, v, data in graph.edges(data=True):
             if 'ricciCurvature' in data:
                 min_kappa = min(min_kappa, data['ricciCurvature'])
        
        # Threshold from PDF is conceptual, let's pick a robust one
        if min_kappa < -5.0: 
            return True
            
        return False
        
    def perform_surgery(self, graph):
        """
        Executes 'Cut' protocol: Severs edges with highly negative curvature.
        Returns list of connected components (sub-graphs).
        """
        # Identify edges to cut
        edges_to_cut = []
        threshold = -5.0 # This should be dynamic based on distribution
        
        for u, v, data in graph.edges(data=True):
            if 'ricciCurvature' in data and data['ricciCurvature'] < threshold:
                edges_to_cut.append((u, v))
                
        # Cut edges
        if edges_to_cut:
            graph.remove_edges_from(edges_to_cut)
            # print(f"Surgery Performed: Severed {len(edges_to_cut)} edges.")
            
        # Return connected components
        # We return the subgraph views or copies? 
        # Copies are safer for independent evolution if we want to modify them separately.
        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        
        return sub_graphs
        
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
                
                # Reset weight in the main graph
                if self.graph.has_edge(u, v):
                    self.graph[u][v]['weight'] = dist


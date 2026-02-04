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
        # Stub
        return False
        
    def perform_surgery(self, graph):
        # Stub
        return []
        
    def manage_sub_swarms(self, sub_swarms):
        # Stub
        pass

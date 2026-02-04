import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
from src.riemannian_swarm import RiemannianSwarm

def test_dumbbell_curvature():
    print("Testing Dumbbell Topology Curvature...")
    
    # Create two clusters
    cluster1 = np.random.normal(loc=[0, 0], scale=0.1, size=(10, 2))
    cluster2 = np.random.normal(loc=[5, 0], scale=0.1, size=(10, 2))
    
    # Bridge points
    bridge = np.array([[2.5, 0]])
    
    agents = np.vstack([cluster1, bridge, cluster2])
    
    swarm = RiemannianSwarm(agents, dimension=2, k_neighbors=5)
    
    print("Building Graph...")
    swarm.step()
    
    if swarm.graph is None:
        print("Error: Graph not built.")
        return

    print(f"Graph built with {len(swarm.graph.nodes)} nodes and {len(swarm.graph.edges)} edges.")
    
    # Check if curvature was computed (if library exists)
    try:
        if 'ricciCurvature' in list(swarm.graph.edges(data=True))[0][2]:
            print("Curvature computed successfully.")
            # Verify negative curvature on the bridge?
            # It's hard to pick the specific edge without ID, but we can check the range of curvatures.
            curvatures = [d['ricciCurvature'] for u, v, d in swarm.graph.edges(data=True)]
            print(f"Curvature Range: {min(curvatures):.2f} to {max(curvatures):.2f}")
        else:
            print("Curvature key not found (Library missing?).")
    except IndexError:
        print("Graph has no edges?")

if __name__ == "__main__":
    test_dumbbell_curvature()

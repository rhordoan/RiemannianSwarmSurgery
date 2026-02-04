import sys
import os
import numpy as np
import networkx as nx
# Ensure path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.riemannian_swarm import RiemannianSwarm

def create_ring_topology(n_points=20, radius=5.0):
    """Creates points arranged in a circle."""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])

def test_ring_topology_loop_detection():
    print("Testing Ring Topology Loop Detection...")
    agents = create_ring_topology(n_points=30, radius=5.0)
    
    # Needs k enough to connect reliable ring but not fill it
    swarm = RiemannianSwarm(agents, dimension=2, k_neighbors=4)
    
    # 1. Build Graph
    swarm.graph = swarm.build_knn_graph(swarm.swarm)
    print(f"Ring Graph: {len(swarm.graph.nodes)} nodes, {len(swarm.graph.edges)} edges.")
    
    # 2. Scouting (Persistent Homology)
    # We invoke private methods of gudhi logic or simulated step if gudhi missing
    # Since we implemented 'step', let's trace it carefully or just check detect_singularity logic
    try:
        import gudhi
        rips = gudhi.RipsComplex(points=swarm.swarm)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        barcode = simplex_tree.persistence()
        
        # Check manual detection
        is_singular = swarm.detect_singularity(barcode, swarm.graph)
        print(f"Singularity Detected on Ring? {is_singular}")
        
        if is_singular:
            print("SUCCESS: Loop detected.")
        else:
            print("FAILURE: Loop NOT detected (check threshold).")
            
    except ImportError:
        print("Skipping Loop Test (Gudhi not found)")

def test_surgery_cutting():
    print("\nTesting Surgery on Dumbbell...")
    # Create dumbbell
    c1 = np.random.normal([0, 0], 0.1, (10, 2))
    c2 = np.random.normal([5, 0], 0.1, (10, 2))
    bridge = np.array([[2.5, 0]]) # Will be connected to both
    agents = np.vstack([c1, bridge, c2])
    
    swarm = RiemannianSwarm(agents, dimension=2, k_neighbors=5)
    swarm.graph = swarm.build_knn_graph(swarm.swarm)
    
    # Manually inject negative curvature on bridge edges to simulate effect of Ricci Flow
    # We know bridge point is index 10 (0-9 are c1, 10 is bridge, 11-20 are c2)
    # Let's find edges connected to node 10
    bridge_node = 10
    edges_to_negatively_curve = list(swarm.graph.edges(bridge_node))
    
    # Inject curvatures
    nx.set_edge_attributes(swarm.graph, 0.0, 'ricciCurvature')
    for u, v in edges_to_negatively_curve:
        swarm.graph[u][v]['ricciCurvature'] = -10.0 # Highly negative
        
    print(f"Injected negative curvature on {len(edges_to_negatively_curve)} edges.")
    
    # Perform Surgery
    sub_swarms = swarm.perform_surgery(swarm.graph)
    
    print(f"Sub-swarms returned: {len(sub_swarms)}")
    
    # Diagnostic
    if len(sub_swarms) > 1:
        print("SUCCESS: Graph split into components.")
        # Check metric reset
        swarm.manage_sub_swarms(sub_swarms)
        print("Metric managed/reset.")
    else:
        print("FAILURE: Graph did not split.")

if __name__ == "__main__":
    test_ring_topology_loop_detection()
    test_surgery_cutting()

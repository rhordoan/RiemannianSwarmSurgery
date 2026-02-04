import sys
import os
import numpy as np
# Ensure path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.riemannian_swarm import RiemannianSwarm

def test_archive_repulsion():
    print("Testing Archive Repulsion (Sheaf & Tabu)...")
    
    # 1. Setup Swarm
    # Create agents in a cluster
    cluster = np.random.normal([10, 10], 1.0, (10, 2))
    # Agent near the cluster
    near_agent = np.array([[10, 10]])
    # Agent far away
    far_agent = np.array([[0, 0]])
    
    # Combined population for indices
    # indices 0-9: cluster
    # index 10: near
    # index 11: far
    agents = np.vstack([cluster, near_agent, far_agent])
    
    # Test Tabu
    print("\n--- Tabu Archive ---")
    swarm_tabu = RiemannianSwarm(agents, dimension=2, archive_type='tabu')
    # Use indices 0-9 to prune
    cluster_indices = list(range(10))
    swarm_tabu.prune_sub_swarm(cluster_indices)
    
    rep_near = swarm_tabu.get_repulsion(10)
    rep_far = swarm_tabu.get_repulsion(11)
    
    print(f"Repulsion Near (10,10): {rep_near:.4f}")
    print(f"Repulsion Far (0,0): {rep_far:.4f}")
    
    if rep_near > 0.0 and rep_far == 0.0:
        print("SUCCESS: Tabu repulsion working correctly.")
    else:
        print("FAILURE: Tabu repulsion logic error.")
        
    # Test Sheaf
    print("\n--- Sheaf Archive ---")
    swarm_sheaf = RiemannianSwarm(agents, dimension=2, archive_type='sheaf')
    swarm_sheaf.prune_sub_swarm(cluster_indices)
    
    rep_near_s = swarm_sheaf.get_repulsion(10)
    rep_far_s = swarm_sheaf.get_repulsion(11)
    
    print(f"Repulsion Near (10,10): {rep_near_s:.4f}")
    print(f"Repulsion Far (0,0): {rep_far_s:.4f}")
    
    if rep_near_s > 0.0 and rep_far_s == 0.0:
        print("SUCCESS: Sheaf repulsion working correctly.")
    else:
        print("FAILURE: Sheaf repulsion logic error.")

if __name__ == "__main__":
    test_archive_repulsion()

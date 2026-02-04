import numpy as np

class ArchiveBase:
    def store(self, region_points):
        """Stores a representation of a visited/pruned region."""
        raise NotImplementedError
    
    def repulsion(self, agent_pos):
        """Returns repulsion strength (metric inflation factor) if close to stored region."""
        raise NotImplementedError

class TabuArchive(ArchiveBase):
    """
    A simple distance-based Tabu List.
    Stores centroids and radii of pruned regions.
    """
    def __init__(self):
        self.ghosts = [] # List of (centroid, radius)

    def store(self, region_points):
        centroid = np.mean(region_points, axis=0)
        # Radius is max distance from centroid to any point in region
        dists = np.linalg.norm(region_points - centroid, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 0.0
        self.ghosts.append((centroid, radius))
        # print(f"TabuArchive: Blocked region at {centroid} with radius {radius:.2f}")

    def repulsion(self, agent_pos):
        penalty = 0.0
        for centroid, radius in self.ghosts:
            dist = np.linalg.norm(agent_pos - centroid)
            if dist < radius * 1.5: # 1.5x buffer zone
                # Exponential penalty as we get closer to centroid
                # If inside, return high value
                penalty += np.exp(-(dist**2) / (radius**2 + 1e-6)) * 10.0
        return penalty

class SheafArchive(ArchiveBase):
    """
    Topological Replay Buffer using Cellular Sheaves.
    Stores 'Ghost Topologies' and uses restriction maps for consistency checking.
    
    In this simplified implementation:
    - Base Space: Discretized grid/hash map of the search space.
    - Stalks: Local curvature/gradient information (or just binary 'visited' state for now).
    - Consistency: If an agent enters a region with a stored 'Ghost', 
      it checks if its local topological state matches the ghost's state (failed basin).
    """
    def __init__(self, resolution=1.0):
        self.ghosts = []
        # In a full implementation, this would be a hash map of open sets.
        # Here we store the point cloud signatures directly to compute consistency.

    def store(self, region_points):
        """
        Stores the point cloud of the failed sub-swarm as a 'Ghost Section'.
        """
        # We store representing points (e.g., using k-means or just the raw points if small)
        # For efficiency, let's store the centroid and a sparse set of boundary points
        centroid = np.mean(region_points, axis=0)
        dists = np.linalg.norm(region_points - centroid, axis=1)
        radius = np.max(dists)
        
        # 'Section' data: The topological signature (centroid, radius, maybe curvature profile)
        section_data = {
            'centroid': centroid,
            'radius': radius,
            'points': region_points  # Store reference for consistency check
        }
        self.ghosts.append(section_data)

    def repulsion(self, agent_pos):
        """
        Computes consistency error via restriction map.
        Restriction Map: rho_uv(s_u) -> s_v
        Checks if agent_pos is 'consistent' with being in a ghost region.
        """
        penalty = 0.0
        for ghost in self.ghosts:
            dist = np.linalg.norm(agent_pos - ghost['centroid'])
            
            # CHANGE: Reduced buffer from 1.2 to 0.9
            # We want to block the CENTER of the trap, but allow grazing the edges
            # because the next "Russian Doll" entrance is often on the edge.
            effective_radius = ghost['radius'] * 0.9 
            
            # Restriction condition: Are we in the support of this section?
            if dist < effective_radius:
                # Consistency Error: In Sheaf theory, this is the Laplacian energy.
                # If we are effectively in the same state that failed, repulsion is high.
                # For this implementation, we treat 'being inside the radius' as 
                # highly consistent with the failed section.
                
                # We can add a "gradient consistency" check here if we had gradients.
                # For now, spatial consistency (proximity) drives the repulsion.
                
                # Metric Inflation: "artificially inflating the metric g_ij -> infinity"
                # We return a multiplier that will divide the step size or increase distance.
                
                # Repulsion strength
                local_strength = (1.0 - (dist / effective_radius)) * 1000.0 # Massive penalty
                penalty += max(0, local_strength)
                
        return penalty

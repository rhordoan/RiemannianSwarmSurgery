"""
Sheaf-Theoretic Archive for Riemannian Swarm Surgery.

Stores two types of ghost regions:
1. Basin ghosts: Failed basins archived when sub-populations are pruned.
   Uses gradient-consistency checking as a simplified restriction map.
2. Neck ghosts: Surgical neck regions archived after Perelman surgery.
   Uses directional repulsion based on the neck's principal direction.

Ghost deduplication prevents redundant archival of the same region.

Mathematical Framing (Cellular Sheaf):
- Base Space: The search domain R^D, covered by open balls around ghosts.
- Stalks: Gradient signature (mean gradient direction) at each ghost.
- Restriction Maps: Cosine similarity between agent gradient and ghost
  gradient. High consistency => same basin => repel.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ArchiveBase:
    """Abstract base for archive strategies."""

    def store(self, region_points, region_fitness=None, region_gradients=None):
        """Store a representation of a visited/pruned region."""
        raise NotImplementedError

    def store_neck(self, neck_info):
        """Store a surgical neck region."""
        pass  # Optional: not all archives support this

    def repulsion(self, agent_pos, agent_gradient=None):
        """Return repulsion strength if agent is in a stored ghost region."""
        raise NotImplementedError

    def num_ghosts(self):
        """Return number of stored ghost regions."""
        raise NotImplementedError


class TabuArchive(ArchiveBase):
    """
    Simple distance-based Tabu List.
    Stores centroids and radii with exponential repulsion.
    Used as an ablation baseline (no gradient consistency, no necks).
    """

    def __init__(self, domain_width=200.0):
        self.ghosts = []
        self.domain_width = domain_width

    def store(self, region_points, region_fitness=None, region_gradients=None):
        centroid = np.mean(region_points, axis=0)
        dists = np.linalg.norm(region_points - centroid, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 0.0
        radius = max(radius, self.domain_width * 0.05)

        # Deduplication
        for existing in self.ghosts:
            dist = np.linalg.norm(centroid - existing[0])
            if dist < max(radius, existing[1]) * 0.5:
                existing[1] = max(existing[1], dist + radius)
                return

        self.ghosts.append([centroid, radius])

    def store_neck(self, neck_info):
        centroid = neck_info['centroid']
        radius = neck_info['radius']
        # Deduplication
        for existing in self.ghosts:
            dist = np.linalg.norm(centroid - existing[0])
            if dist < max(radius, existing[1]) * 0.5:
                existing[1] = max(existing[1], dist + radius)
                return
        self.ghosts.append([centroid, radius])

    def repulsion(self, agent_pos, agent_gradient=None):
        penalty = 0.0
        for centroid, radius in self.ghosts:
            dist = np.linalg.norm(agent_pos - centroid)
            if dist < radius * 1.2:
                penalty += np.exp(-(dist ** 2) / (radius ** 2 + 1e-6)) * 10.0
        return penalty

    def num_ghosts(self):
        return len(self.ghosts)


class SheafArchive(ArchiveBase):
    """
    Topological memory with gradient consistency and neck geometry.

    Stores two types of ghosts:
    - Basin ghosts: From pruned sub-populations (gradient-consistent repulsion)
    - Neck ghosts: From surgical cuts (directional repulsion along neck axis)

    Ghost deduplication prevents storing the same region multiple times.

    Args:
        domain_width: Width of the search domain (default 200 for [-100,100]).
        min_radius_fraction: Minimum ghost radius as fraction of domain width.
        merge_overlap: Overlap fraction for deduplication (default 0.5).
    """

    def __init__(self, domain_width=200.0, min_radius_fraction=0.05,
                 merge_overlap=0.5):
        self.basin_ghosts = []   # Basin ghosts (pruned sub-populations)
        self.neck_ghosts = []    # Neck ghosts (surgical cut regions)
        self.domain_width = domain_width
        self.min_radius = domain_width * min_radius_fraction
        self.merge_overlap = merge_overlap

    def store(self, region_points, region_fitness=None, region_gradients=None):
        """
        Store a basin ghost with optional gradient signature.
        Deduplicates against existing basin ghosts.
        """
        centroid = np.mean(region_points, axis=0)
        dists = np.linalg.norm(region_points - centroid, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 0.0
        radius = max(radius, self.min_radius)

        # Deduplication: check overlap with existing basin ghosts
        for existing in self.basin_ghosts:
            dist = np.linalg.norm(centroid - existing['centroid'])
            merge_dist = max(radius, existing['radius']) * self.merge_overlap
            if dist < merge_dist:
                # Merge: expand existing ghost
                existing['radius'] = max(existing['radius'], dist + radius)
                if region_fitness is not None and len(region_fitness) > 0:
                    new_best = float(np.min(region_fitness))
                    if existing['best_fitness'] is None \
                            or new_best < existing['best_fitness']:
                        existing['best_fitness'] = new_best
                logger.debug(
                    f"Merged basin ghost at {centroid[:2]}... "
                    f"into existing (R={existing['radius']:.1f})"
                )
                return

        # Compute gradient signature
        gradient_signature = None
        if region_gradients is not None and len(region_gradients) > 0:
            mean_grad = np.mean(region_gradients, axis=0)
            norm = np.linalg.norm(mean_grad)
            if norm > 1e-10:
                gradient_signature = mean_grad / norm

        best_fit = None
        if region_fitness is not None and len(region_fitness) > 0:
            best_fit = float(np.min(region_fitness))

        section = {
            'centroid': centroid,
            'radius': radius,
            'gradient_signature': gradient_signature,
            'best_fitness': best_fit,
        }
        self.basin_ghosts.append(section)

        logger.info(
            f"Stored basin ghost at {centroid[:2]}... "
            f"(R={radius:.1f}, "
            f"grad={'yes' if gradient_signature is not None else 'no'}). "
            f"Total basin ghosts: {len(self.basin_ghosts)}"
        )

    def store_neck(self, neck_info):
        """
        Store a neck ghost from a surgical cut.
        Deduplicates against existing neck ghosts.

        neck_info: dict with 'centroid', 'direction', 'radius'
        """
        centroid = neck_info['centroid']
        radius = neck_info.get('radius', self.min_radius)

        # Deduplication
        for existing in self.neck_ghosts:
            dist = np.linalg.norm(centroid - existing['centroid'])
            merge_dist = max(radius, existing['radius']) * self.merge_overlap
            if dist < merge_dist:
                existing['radius'] = max(existing['radius'], dist + radius)
                logger.debug(
                    f"Merged neck ghost at {centroid[:2]}... "
                    f"into existing (R={existing['radius']:.1f})"
                )
                return

        self.neck_ghosts.append({
            'centroid': centroid.copy(),
            'direction': neck_info.get('direction', None),
            'radius': radius,
        })

        logger.info(
            f"Stored neck ghost at {centroid[:2]}... "
            f"(R={radius:.1f}, "
            f"dir={'yes' if neck_info.get('direction') is not None else 'no'}). "
            f"Total neck ghosts: {len(self.neck_ghosts)}"
        )

    def repulsion(self, agent_pos, agent_gradient=None):
        """
        Compute total repulsion from both basin and neck ghosts.

        Basin ghosts: spatial * gradient_consistency * strength
        Neck ghosts: spatial * directional_factor * strength
        """
        penalty = 0.0

        # Basin ghost repulsion
        for ghost in self.basin_ghosts:
            dist = np.linalg.norm(agent_pos - ghost['centroid'])
            if dist >= ghost['radius']:
                continue

            spatial = 1.0 - (dist / ghost['radius'])

            if (agent_gradient is not None
                    and ghost['gradient_signature'] is not None):
                agent_grad_norm = np.linalg.norm(agent_gradient)
                if agent_grad_norm > 1e-10:
                    cos_sim = np.dot(
                        agent_gradient / agent_grad_norm,
                        ghost['gradient_signature']
                    )
                    grad_consistency = max(0.0, cos_sim)
                else:
                    grad_consistency = 0.5
            else:
                grad_consistency = 0.5

            penalty += spatial * grad_consistency * 50.0

        # Neck ghost repulsion (directional)
        for neck in self.neck_ghosts:
            dist = np.linalg.norm(agent_pos - neck['centroid'])
            if dist >= neck['radius']:
                continue

            spatial = 1.0 - (dist / neck['radius'])

            # Directional factor: stronger repulsion along neck direction
            if neck['direction'] is not None:
                to_agent = agent_pos - neck['centroid']
                to_agent_norm = np.linalg.norm(to_agent)
                if to_agent_norm > 1e-10:
                    # How aligned is approach direction with neck direction?
                    alignment = abs(np.dot(
                        to_agent / to_agent_norm,
                        neck['direction']
                    ))
                    # High alignment = approaching along neck = repel strongly
                    dir_factor = 0.3 + 0.7 * alignment
                else:
                    dir_factor = 1.0
            else:
                dir_factor = 1.0

            penalty += spatial * dir_factor * 30.0

        return max(0.0, penalty)

    def num_ghosts(self):
        return len(self.basin_ghosts) + len(self.neck_ghosts)

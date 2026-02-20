"""
Topological Saddle Archive (TSA).

Stores inter-basin saddle points identified by the Ollivier-Ricci Oracle and
provides exploration-directed injection vectors for population restarts.

Key design decisions (informed by D=10 CEC 2022 ablation results):

  1. EXPLORATION-DIRECTED DESCENT:
     When the oracle detects a saddle edge (u, v) with f(u) < f(v), u is
     in the known-good basin and v is on the boundary of an unexplored region.
     The injection direction points FROM the better agent TOWARD the worse one
     (i.e., into the unexplored basin).  This is the opposite of gradient
     descent -- the goal is to DISCOVER new basins, not exploit known ones.

     descent = (x_worse - x_better) / ||x_worse - x_better||

     Rationale: the optimizer already handles exploitation.  TMI's value
     is in exploration -- placing agents where the optimizer hasn't been.

  2. ADAPTIVE STEP SIZE:
     Each saddle stores the actual Euclidean distance between its endpoint
     agents.  Injection step size is proportional to this distance rather
     than a fixed fraction of the domain.  This ensures the injected agent
     lands in the next basin (not 30 units away in a 200-unit domain when
     the basins are only 5 units apart).

  3. SADDLE EXPIRY:
     Saddles older than max_age generations are automatically pruned.
     Population geometry changes as the optimizer progresses; a saddle
     detected 500 generations ago likely describes extinct structure.

  4. DEDUPLICATION + CAPACITY:  Unchanged from v1.
"""

import numpy as np


class SaddleArchive:

    def __init__(self,
                 domain_width: float = 200.0,
                 min_sep_fraction: float = 0.05,
                 max_saddles: int = 30,
                 max_age: int = 300):
        self.domain_width = domain_width
        self.min_sep = domain_width * min_sep_fraction
        self.max_saddles = max_saddles
        self.max_age = max_age
        self.saddles: list = []

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_saddle(self,
                     x_u: np.ndarray,
                     x_v: np.ndarray,
                     f_u: float,
                     f_v: float,
                     generation: int = 0,
                     nbr_centroid_explore: np.ndarray = None) -> bool:
        """
        Store a detected inter-basin saddle.

        Args:
            x_u, x_v:   Positions of the two agents on each side of the saddle.
            f_u, f_v:   Fitness values (lower = better).
            generation: Current generation.
            nbr_centroid_explore: Centroid of the neighborhood on the
                        *unexplored* (worse) side.  If provided, used as the
                        injection target instead of the crude descent vector.

        Returns:
            True if stored, False if duplicate/degenerate.
        """
        midpoint = (x_u + x_v) * 0.5

        for s in self.saddles:
            if np.linalg.norm(midpoint - s['center']) < self.min_sep:
                return False

        # Exploration-directed: point AWAY from the better agent,
        # INTO the unexplored basin.
        if f_u <= f_v:
            x_better, x_worse = x_u, x_v
        else:
            x_better, x_worse = x_v, x_u

        raw = x_worse - x_better
        edge_dist = float(np.linalg.norm(raw))
        if edge_dist < 1e-12:
            return False

        explore_dir = raw / edge_dist

        # If the caller provided the neighborhood centroid on the unexplored
        # side, use it as a more precise injection target.
        if nbr_centroid_explore is not None:
            raw_nbr = nbr_centroid_explore - x_better
            nbr_norm = float(np.linalg.norm(raw_nbr))
            if nbr_norm > 1e-12:
                explore_dir = raw_nbr / nbr_norm
                edge_dist = nbr_norm

        self.saddles.append({
            'center': midpoint.copy(),
            'explore_dir': explore_dir.copy(),
            'edge_dist': edge_dist,
            'best_fitness': min(f_u, f_v),
            'generation': generation,
        })

        if len(self.saddles) > self.max_saddles:
            self.saddles.sort(key=lambda s: s['best_fitness'])
            self.saddles = self.saddles[:self.max_saddles]

        return True

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def get_injection_points(self,
                             n: int,
                             step_size: float,
                             lb: float,
                             ub: float,
                             rng: np.random.Generator = None,
                             current_gen: int = 0) -> np.ndarray | None:
        """
        Generate n injection positions aimed at unexplored basins.

        Step size for each saddle is adaptive: max(step_size, 0.5 * edge_dist),
        ensuring the injected agent clears the saddle barrier and lands in the
        neighboring basin.  Points are spread across multiple distinct saddles
        for diversity.

        Args:
            n:           Number of injection points.
            step_size:   Base step size (fallback if edge_dist is unavailable).
            lb, ub:      Domain bounds.
            rng:         Optional numpy Generator.
            current_gen: Current generation (used for age-based expiry).
        """
        # Expire old saddles
        if self.max_age and current_gen > 0:
            self.saddles = [s for s in self.saddles
                           if (current_gen - s['generation']) <= self.max_age]

        if not self.saddles or n <= 0:
            return None

        if rng is None:
            rng = np.random.default_rng()

        dim = len(self.saddles[0]['center'])

        # Rank by best_fitness (best first), but cycle through ALL saddles
        # to ensure diversity.  Don't put all eggs in one basket.
        ordered = sorted(self.saddles, key=lambda s: s['best_fitness'])

        points = np.empty((n, dim))
        for i in range(n):
            saddle = ordered[i % len(ordered)]

            # Adaptive step: at least half the edge distance, at most step_size
            effective_step = max(step_size, 0.5 * saddle.get('edge_dist', step_size))

            noise_std = max(effective_step * 0.15, 1e-6)
            noise = rng.normal(0.0, noise_std, dim)

            pt = saddle['center'] + effective_step * saddle['explore_dir'] + noise
            points[i] = np.clip(pt, lb, ub)

        return points

    def clear(self):
        self.saddles.clear()

    @property
    def num_saddles(self) -> int:
        return len(self.saddles)

    @property
    def best_saddle_fitness(self) -> float | None:
        if not self.saddles:
            return None
        return min(s['best_fitness'] for s in self.saddles)

    def summary(self) -> str:
        return (f"SaddleArchive: {self.num_saddles}/{self.max_saddles} saddles, "
                f"best_fitness={self.best_saddle_fitness}")

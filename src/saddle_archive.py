"""
Topological Saddle Archive (TSA).

Stores inter-basin saddle points identified by the Ollivier-Ricci Oracle and
provides fitness-guided injection vectors for population restarts.

Design principles:
  1. FITNESS-GUIDED DESCENT: The descent direction is derived solely from
     fitness comparisons between two agents on opposite sides of a detected
     saddle edge.  No gradient, Hessian, or extra function evaluations needed.

     If agent u has f(u) < f(v)  (u is in the better basin):
         descent = (x_u - x_v) / ||x_u - x_v||   (point toward the better basin)

  2. DEDUPLICATION: Saddles closer than min_sep in search space are treated as
     the same boundary and not stored twice.  This prevents wasting injection
     budget on redundant restarts.

  3. CAPACITY: Only the best max_saddles saddles (by fitness of the better
     agent) are retained.  This ensures injections are always pointed toward
     the most promising discovered region.

  4. INJECTION SAMPLING: Injection points are placed at
         p = center + step_size * descent + small_noise
     and clipped to [lb, ub].  The step_size controls how far past the saddle
     midpoint the injected agent lands -- it should be a fraction of domain width
     (default 0.15) so the new agent enters the target basin rather than landing
     exactly on the boundary.
"""

import numpy as np


class SaddleArchive:
    """
    Archive of topological saddle points with fitness-guided descent vectors.

    Args:
        domain_width:    Diameter of the search domain (ub - lb).  Used to
                         set the minimum separation distance for deduplication.
        min_sep_fraction: Saddles within this fraction of domain_width of an
                         existing saddle are discarded as duplicates.
        max_saddles:     Maximum number of saddles to retain (keeps best by fitness).
    """

    def __init__(self,
                 domain_width: float = 200.0,
                 min_sep_fraction: float = 0.05,
                 max_saddles: int = 30):
        self.domain_width = domain_width
        self.min_sep = domain_width * min_sep_fraction
        self.max_saddles = max_saddles
        self.saddles: list = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store_saddle(self,
                     x_u: np.ndarray,
                     x_v: np.ndarray,
                     f_u: float,
                     f_v: float,
                     generation: int = 0) -> bool:
        """
        Attempt to store a new saddle point.

        Args:
            x_u, x_v:   Positions of the two agents on each side of the saddle.
            f_u, f_v:   Fitness values (lower = better).
            generation:  Current generation (for bookkeeping).

        Returns:
            True if the saddle was stored, False if it was a duplicate or degenerate.
        """
        midpoint = (x_u + x_v) * 0.5

        # Deduplicate: skip if too close to an existing archived saddle
        for s in self.saddles:
            if np.linalg.norm(midpoint - s['center']) < self.min_sep:
                return False

        # Fitness-guided descent: point from the worse basin toward the better one.
        # f is a minimization objective; lower f = better.
        if f_u <= f_v:
            raw_descent = x_u - x_v  # toward agent u (better basin)
        else:
            raw_descent = x_v - x_u  # toward agent v (better basin)

        norm = float(np.linalg.norm(raw_descent))
        if norm < 1e-12:
            return False

        descent = raw_descent / norm
        best_fitness = min(f_u, f_v)

        self.saddles.append({
            'center': midpoint.copy(),
            'descent': descent.copy(),
            'best_fitness': best_fitness,
            'generation': generation,
        })

        # Prune to max_saddles, keeping those with lowest (best) fitness
        if len(self.saddles) > self.max_saddles:
            self.saddles.sort(key=lambda s: s['best_fitness'])
            self.saddles = self.saddles[:self.max_saddles]

        return True

    def get_injection_points(self,
                             n: int,
                             step_size: float,
                             lb: float,
                             ub: float,
                             rng: np.random.Generator = None) -> np.ndarray | None:
        """
        Generate n injection positions from the best archived saddles.

        Each position is placed just past the saddle midpoint in the descent
        direction, with small Gaussian noise for diversity.

        Args:
            n:          Number of injection points to generate.
            step_size:  How far past the saddle to inject (in search-space units).
                        Typically INJECT_STEP_SIZE_FRAC * domain_width.
            lb, ub:     Domain bounds (scalar, applied uniformly per dimension).
            rng:        Optional numpy Generator for reproducibility.

        Returns:
            Array of shape (n, dim), or None if archive is empty.
        """
        if not self.saddles or n <= 0:
            return None

        if rng is None:
            rng = np.random.default_rng()

        dim = len(self.saddles[0]['center'])
        noise_std = max(step_size * 0.1, 1e-6)

        # Best-first ordering (already sorted after pruning, but re-sort for safety)
        ordered = sorted(self.saddles, key=lambda s: s['best_fitness'])

        points = np.empty((n, dim))
        for i in range(n):
            saddle = ordered[i % len(ordered)]
            noise = rng.normal(0.0, noise_std, dim)
            pt = saddle['center'] + step_size * saddle['descent'] + noise
            points[i] = np.clip(pt, lb, ub)

        return points

    def clear(self):
        """Empty the archive."""
        self.saddles.clear()

    # ------------------------------------------------------------------
    # Properties / info
    # ------------------------------------------------------------------

    @property
    def num_saddles(self) -> int:
        """Number of archived saddle points."""
        return len(self.saddles)

    @property
    def best_saddle_fitness(self) -> float | None:
        """Fitness of the best saddle endpoint discovered so far."""
        if not self.saddles:
            return None
        return min(s['best_fitness'] for s in self.saddles)

    def summary(self) -> str:
        return (f"SaddleArchive: {self.num_saddles}/{self.max_saddles} saddles, "
                f"best_fitness={self.best_saddle_fitness}")

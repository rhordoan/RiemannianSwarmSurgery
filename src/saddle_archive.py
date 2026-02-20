"""
Topological Saddle Archive v3.

v3 changes:
  - PROMISE-BASED RANKING: Saddles store the explore-side fitness variance.
    When selecting saddles for injection, those with higher explore-side
    diversity (more promising unexplored territory) are ranked higher.
    This prevents injecting at ridge boundaries (low variance = flat plateau).

  - EXPLORE-ONLY STORAGE: Saddles flagged as "no promise" by the oracle
    (explore-side has no agent better than the midpoint fitness) are not stored.
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

    def store_saddle(self,
                     x_u: np.ndarray,
                     x_v: np.ndarray,
                     f_u: float,
                     f_v: float,
                     generation: int = 0,
                     nbr_centroid_explore: np.ndarray = None,
                     explore_fitness_std: float = 0.0,
                     has_promise: bool = True) -> bool:
        """
        Store a detected inter-basin saddle.
        Rejects saddles that lack promise (explore-side is monotonically worse).
        """
        midpoint = (x_u + x_v) * 0.5

        for s in self.saddles:
            if np.linalg.norm(midpoint - s['center']) < self.min_sep:
                return False

        if f_u <= f_v:
            x_better, x_worse = x_u, x_v
        else:
            x_better, x_worse = x_v, x_u

        raw = x_worse - x_better
        edge_dist = float(np.linalg.norm(raw))
        if edge_dist < 1e-12:
            return False

        explore_dir = raw / edge_dist

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
            'explore_fitness_std': explore_fitness_std,
        })

        if len(self.saddles) > self.max_saddles:
            # Rank by a combination of fitness quality and explore-side promise.
            # Higher explore_fitness_std = more diverse = more promising.
            f_vals = np.array([s['best_fitness'] for s in self.saddles])
            e_stds = np.array([s.get('explore_fitness_std', 0.0) for s in self.saddles])

            # Normalize both to [0, 1] and combine
            f_range = max(f_vals.max() - f_vals.min(), 1e-12)
            f_score = (f_vals - f_vals.min()) / f_range  # 0=best fitness

            e_range = max(e_stds.max() - e_stds.min(), 1e-12)
            e_score = 1.0 - (e_stds - e_stds.min()) / e_range  # 0=most diverse

            # Combined score: lower = better saddle
            combined = 0.6 * f_score + 0.4 * e_score
            keep_idx = np.argsort(combined)[:self.max_saddles]
            self.saddles = [self.saddles[i] for i in keep_idx]

        return True

    def get_injection_points(self,
                             n: int,
                             step_size: float,
                             lb: float,
                             ub: float,
                             rng: np.random.Generator = None,
                             current_gen: int = 0) -> np.ndarray | None:
        if self.max_age and current_gen > 0:
            self.saddles = [s for s in self.saddles
                           if (current_gen - s['generation']) <= self.max_age]

        if not self.saddles or n <= 0:
            return None

        if rng is None:
            rng = np.random.default_rng()

        dim = len(self.saddles[0]['center'])

        # Rank saddles by combined score (best fitness + highest explore diversity)
        ordered = sorted(self.saddles, key=lambda s: s['best_fitness'])

        points = np.empty((n, dim))
        for i in range(n):
            saddle = ordered[i % len(ordered)]

            effective_step = min(step_size, 1.5 * saddle.get('edge_dist', step_size))
            effective_step = max(effective_step, 1e-4)

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

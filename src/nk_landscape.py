"""
NK Landscape implementation for Fitness Landscape Analysis.

The NK model (Kauffman 1993) defines tunably rugged fitness landscapes
over binary strings of length N. Each locus i has K epistatic interactions
with other loci, and its fitness contribution depends on the values of
those K+1 positions. Total fitness is the mean of per-locus contributions.

Parameters
----------
N : string length (number of loci)
K : number of epistatic interactions per locus (0 to N-1)
    K=0 -> smooth (unimodal), K=N-1 -> maximally rugged

Interaction models:
- 'adjacent': locus i interacts with i+1, ..., i+K (mod N)
- 'random': K random loci chosen uniformly

References
----------
Kauffman (1993). The Origins of Order.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class NKLandscape:
    """NK Landscape with precomputed fitness for all 2^N solutions."""

    def __init__(self, N: int, K: int, model: str = 'adjacent', seed: int = 0):
        if K >= N:
            raise ValueError(f"K={K} must be < N={N}")
        self.N = N
        self.K = K
        self.model = model
        self.seed = seed
        self.space_size = 2 ** N

        rng = np.random.RandomState(seed)
        self._interactions = self._build_interactions(rng)
        self._contrib_tables = self._build_tables(rng)
        self._fitness = self._compute_all_fitness()

    def _build_interactions(self, rng: np.random.RandomState) -> List[List[int]]:
        interactions = []
        for i in range(self.N):
            if self.model == 'adjacent':
                deps = [(i + 1 + j) % self.N for j in range(self.K)]
            elif self.model == 'random':
                others = [j for j in range(self.N) if j != i]
                deps = rng.choice(others, size=self.K, replace=False).tolist()
            else:
                raise ValueError(f"Unknown model: {self.model}")
            interactions.append(sorted([i] + deps))
        return interactions

    def _build_tables(self, rng: np.random.RandomState) -> List[np.ndarray]:
        tables = []
        for i in range(self.N):
            table_size = 2 ** (self.K + 1)
            tables.append(rng.rand(table_size))
        return tables

    def _locus_contribution(self, bitstring: int, locus: int) -> float:
        deps = self._interactions[locus]
        key = 0
        for j, d in enumerate(deps):
            if bitstring & (1 << d):
                key |= (1 << j)
        return float(self._contrib_tables[locus][key])

    def evaluate(self, bitstring: int) -> float:
        total = sum(self._locus_contribution(bitstring, i) for i in range(self.N))
        return total / self.N

    def _compute_all_fitness(self) -> np.ndarray:
        fitness = np.empty(self.space_size)
        for idx in range(self.space_size):
            fitness[idx] = self.evaluate(idx)
        return fitness

    @property
    def fitness(self) -> np.ndarray:
        """Fitness array, shape (2^N,). Higher = better (maximization).
        For landscape analysis we negate: lower = better (minimization)."""
        return -self._fitness

    @property
    def raw_fitness(self) -> np.ndarray:
        """Raw fitness (higher = better)."""
        return self._fitness.copy()

    def neighbor_fn(self, idx: int) -> List[int]:
        """Bit-flip neighbors: flip each of N bits."""
        return [idx ^ (1 << i) for i in range(self.N)]

    def global_optimum(self) -> int:
        """Index of the global optimum (maximum raw fitness)."""
        return int(np.argmax(self._fitness))

    def idx_to_bits(self, idx: int) -> str:
        return format(idx, f'0{self.N}b')


def create_nk_suite(
    N: int = 16,
    K_values: List[int] = None,
    n_instances: int = 30,
    model: str = 'adjacent',
) -> List[NKLandscape]:
    """Create a suite of NK landscape instances."""
    if K_values is None:
        K_values = [0, 2, 4, 6, 8, 12, N - 1]
    instances = []
    for K in K_values:
        for seed in range(n_instances):
            instances.append(NKLandscape(N, K, model=model, seed=seed))
    return instances

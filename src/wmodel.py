"""
W-Model implementation for Fitness Landscape Analysis.

The W-Model (Weise & Wu, 2014) is a tunable benchmark for discrete
optimization over binary strings. It applies a chain of transformations
to create landscapes with controllable neutrality, epistasis, and
ruggedness/deceptiveness.

Pipeline: x -> neutrality(mu) -> epistasis(nu) -> ruggedness(gamma) -> fitness

Base problem: number of ones (OneMax-like, shifted to target string).

Parameters
----------
n     : string length
mu    : neutrality degree (1 = none, higher = more neutral)
nu    : epistasis block size (2 = none, n = maximum)
gamma : ruggedness/deceptiveness (0 = none, up to n*(n-1)/2 = maximum)

References
----------
Weise & Wu (2014). Difficult Features of Combinatorial Optimization
    Problems and Algorithm Selection with the W-Model.
"""

from __future__ import annotations

from typing import List

import numpy as np


class WModel:
    """W-Model with precomputed fitness for all 2^n solutions."""

    def __init__(self, n: int, mu: int = 1, nu: int = 2, gamma: int = 0,
                 seed: int = 0):
        self.n = n
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.seed = seed

        self._n_effective = n * mu
        self.space_size = 2 ** n

        rng = np.random.RandomState(seed)
        self._target = rng.randint(0, 2, size=n).tolist()
        self._ruggedness_perm = self._build_ruggedness_permutation()
        self._fitness = self._compute_all_fitness()

    def _neutrality_reduction(self, bits: List[int]) -> List[int]:
        """Reduce representation via majority vote in blocks of mu."""
        if self.mu <= 1:
            return bits
        reduced = []
        for i in range(0, len(bits), self.mu):
            block = bits[i:i + self.mu]
            reduced.append(1 if sum(block) > len(block) // 2 else 0)
        return reduced

    def _epistasis_transform(self, bits: List[int]) -> List[int]:
        """Apply epistasis via overlapping XOR blocks of size nu."""
        if self.nu <= 2:
            return bits
        n = len(bits)
        result = []
        for i in range(n):
            block_start = max(0, i - self.nu + 1)
            val = 0
            for j in range(block_start, i + 1):
                val ^= bits[j]
            result.append(val)
        return result

    def _build_ruggedness_permutation(self) -> List[int]:
        """Build the ruggedness permutation on {0, ..., n}."""
        n = self.n
        g = self.gamma
        perm = list(range(n + 1))
        if g == 0:
            return perm

        perm_out = [0] * (n + 1)
        for v in range(n + 1):
            if g >= n * (n - 1) // 2:
                perm_out[v] = n - v
            else:
                remaining = g
                mapped = v
                sign = 1
                delta = n
                current_low = 0
                current_high = n

                while remaining > 0 and current_low < current_high:
                    step = min(remaining, delta)
                    if sign > 0:
                        if v <= current_low + step and v >= current_low:
                            mapped = current_high - (v - current_low)
                            break
                        current_low_new = current_low + step
                    else:
                        if v >= current_high - step and v <= current_high:
                            mapped = current_low + (current_high - v)
                            break
                        current_high_new = current_high - step

                    remaining -= step
                    delta -= 1
                    if sign > 0:
                        current_low = current_low_new
                    else:
                        current_high = current_high_new
                    sign *= -1

                perm_out[v] = mapped

        return perm_out

    def evaluate_bits(self, bits: List[int]) -> float:
        """Evaluate a bit list through the full W-model pipeline."""
        if self.mu > 1:
            extended = []
            for b in bits:
                extended.extend([b] * self.mu)
            working = self._neutrality_reduction(extended)
        else:
            working = list(bits)

        working = self._epistasis_transform(working)

        n_match = sum(1 for a, b in zip(working, self._target) if a == b)

        fitness_val = self._ruggedness_perm[n_match]
        return float(fitness_val) / self.n

    def evaluate(self, idx: int) -> float:
        bits = [(idx >> i) & 1 for i in range(self.n)]
        return self.evaluate_bits(bits)

    def _compute_all_fitness(self) -> np.ndarray:
        fitness = np.empty(self.space_size)
        for idx in range(self.space_size):
            fitness[idx] = self.evaluate(idx)
        return fitness

    @property
    def fitness(self) -> np.ndarray:
        """Fitness array (lower = better, minimization). Shape (2^n,)."""
        return -self._fitness

    @property
    def raw_fitness(self) -> np.ndarray:
        """Raw fitness (higher = better)."""
        return self._fitness.copy()

    def neighbor_fn(self, idx: int) -> List[int]:
        """Bit-flip neighbors."""
        return [idx ^ (1 << i) for i in range(self.n)]

    def global_optimum(self) -> int:
        return int(np.argmax(self._fitness))

    def idx_to_bits(self, idx: int) -> str:
        return format(idx, f'0{self.n}b')


def create_wmodel_suite(
    n: int = 16,
    nu_values: List[int] = None,
    n_instances: int = 30,
    mu: int = 1,
    gamma: int = 0,
) -> List[WModel]:
    """Create a suite of W-model instances with varying epistasis (nu)."""
    if nu_values is None:
        nu_values = [1, 3, 4, 6, 8, n]
    instances = []
    for nu in nu_values:
        for seed in range(n_instances):
            instances.append(WModel(n, mu=mu, nu=nu, gamma=gamma, seed=seed))
    return instances

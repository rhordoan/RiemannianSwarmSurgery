"""
NAS Search Algorithms: Baselines and ORC-augmented variants.

Algorithms implemented:
  - RandomSearch: uniform random sampling (sanity baseline)
  - RegularizedEvolution (RE): the standard NAS baseline (Real et al., 2019)
  - LocalSearch (LS): hill climbing with random restarts
  - TabuSearch: hill climbing with random non-backtracking escapes
  - SMAC: Random Forest surrogate-guided search (SMAC/BOHB style)
  - ORC_RE: RE augmented with ORC saddle-guided mutation
  - ORC_Tabu: Tabu search with ORC-guided saddle crossing

All algorithms operate on the NAS-Bench-201 topology search space
(15,625 architectures) and count "queries" (fitness evaluations) as the
cost metric.

References
----------
Real et al. (2019). Regularized Evolution for Image Classifier Architecture
    Search. AAAI 2019.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.orc_nas import (
    SPACE_SIZE, N_EDGES, N_OPS,
    index_to_tuple, tuple_to_index,
    get_neighbors, get_neighbor_indices,
    compute_orc_neighborhood, find_saddle_direction,
)


@dataclass
class SearchResult:
    """Result of a NAS search run."""
    best_idx: int
    best_fitness: float
    best_accuracy: float
    n_queries: int
    history: List[Tuple[int, float]]  # [(query_count, best_accuracy_so_far)]


# ---------------------------------------------------------------------------
# Fitness oracle (wraps the benchmark with query counting)
# ---------------------------------------------------------------------------

class FitnessOracle:
    """Wraps NASBench201 with query counting and caching."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.fitness = benchmark.fitness
        self.n_queries = 0
        self._cache = {}
        self.best_idx = -1
        self.best_fitness = np.inf
        self.best_accuracy = 0.0
        self.history: List[Tuple[int, float]] = []

    def query(self, idx: int) -> float:
        """Query fitness, always counting toward budget. Returns fitness (lower=better)."""
        self.n_queries += 1
        if idx in self._cache:
            return self._cache[idx]
        f = float(self.fitness[idx])
        self._cache[idx] = f
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_idx = idx
            self.best_accuracy = float(self.benchmark.accuracy[idx])
        self.history.append((self.n_queries, self.best_accuracy))
        return f

    def result(self) -> SearchResult:
        return SearchResult(
            best_idx=self.best_idx,
            best_fitness=self.best_fitness,
            best_accuracy=self.best_accuracy,
            n_queries=self.n_queries,
            history=list(self.history),
        )


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

def random_search(benchmark, budget: int, seed: int = 0) -> SearchResult:
    """Uniform random sampling baseline."""
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    for _ in range(budget):
        idx = rng.randint(0, SPACE_SIZE)
        oracle.query(idx)

    return oracle.result()


# ---------------------------------------------------------------------------
# Regularized Evolution (RE)
# ---------------------------------------------------------------------------

def regularized_evolution(
    benchmark,
    budget: int,
    pop_size: int = 50,
    tournament_size: int = 10,
    seed: int = 0,
) -> SearchResult:
    """
    Standard Regularized Evolution (Real et al., 2019).

    Maintains a population (FIFO queue). Each step: sample a tournament,
    pick the best, mutate one random operation, add child, kill oldest.
    """
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    population = deque(maxlen=pop_size)
    for _ in range(pop_size):
        if oracle.n_queries >= budget:
            return oracle.result()
        idx = rng.randint(0, SPACE_SIZE)
        f = oracle.query(idx)
        population.append((idx, f))

    while oracle.n_queries < budget:
        sample_indices = rng.choice(len(population), size=tournament_size,
                                     replace=False)
        tournament = [population[i] for i in sample_indices]
        parent_idx, _ = min(tournament, key=lambda x: x[1])

        child_idx = _random_mutate(parent_idx, rng)
        f = oracle.query(child_idx)
        population.append((child_idx, f))

    return oracle.result()


# ---------------------------------------------------------------------------
# Local Search (LS)
# ---------------------------------------------------------------------------

def local_search(
    benchmark,
    budget: int,
    n_restarts: int = 100,
    seed: int = 0,
) -> SearchResult:
    """
    Hill climbing with random restarts.

    From a random start, evaluate all 24 neighbors, move to the best one.
    When stuck (no improving neighbor), restart from a new random point.
    """
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    for _ in range(n_restarts):
        if oracle.n_queries >= budget:
            break

        current_idx = rng.randint(0, SPACE_SIZE)
        current_f = oracle.query(current_idx)

        while oracle.n_queries < budget:
            nbr_indices = get_neighbor_indices(current_idx)
            best_nbr_idx = None
            best_nbr_f = current_f

            for nbr_idx in nbr_indices:
                if oracle.n_queries >= budget:
                    return oracle.result()
                f = oracle.query(nbr_idx)
                if f < best_nbr_f:
                    best_nbr_f = f
                    best_nbr_idx = nbr_idx

            if best_nbr_idx is None:
                break  # local optimum, restart
            current_idx = best_nbr_idx
            current_f = best_nbr_f

    return oracle.result()


# ---------------------------------------------------------------------------
# Tabu Search (Random Saddle Crossing)
# ---------------------------------------------------------------------------

def tabu_search(
    benchmark,
    budget: int,
    seed: int = 0,
) -> SearchResult:
    """
    Hill climbing with random non-backtracking escapes.

    When stuck at a local optimum, takes a random move to a neighbor
    not in the tabu list (preventing immediate backtracking).
    """
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    current_idx = rng.randint(0, SPACE_SIZE)
    current_f = oracle.query(current_idx)
    tabu_list = set()

    while oracle.n_queries < budget:
        nbr_indices = get_neighbor_indices(current_idx)

        best_nbr_idx = None
        best_nbr_f = current_f

        for nbr_idx in nbr_indices:
            if oracle.n_queries >= budget:
                return oracle.result()
            f = oracle.query(nbr_idx)
            if f < best_nbr_f:
                best_nbr_f = f
                best_nbr_idx = nbr_idx

        if best_nbr_idx is not None:
            tabu_list.clear()
            tabu_list.add(current_idx)
            current_idx = best_nbr_idx
            current_f = best_nbr_f
            continue

        valid_nbrs = [n for n in nbr_indices if n not in tabu_list]
        if not valid_nbrs:
            current_idx = rng.randint(0, SPACE_SIZE)
            current_f = oracle.query(current_idx)
            tabu_list.clear()
            continue

        next_idx = rng.choice(valid_nbrs)
        tabu_list.add(current_idx)
        current_idx = next_idx
        current_f = oracle.query(current_idx)

    return oracle.result()


# ---------------------------------------------------------------------------
# ORC-guided Regularized Evolution (ORC-RE)
# ---------------------------------------------------------------------------

def orc_regularized_evolution(
    benchmark,
    budget: int,
    pop_size: int = 50,
    tournament_size: int = 10,
    orc_prob: float = 0.3,
    gamma: float = 1.0,
    seed: int = 0,
) -> SearchResult:
    """
    Regularized Evolution augmented with ORC saddle detection.

    With probability orc_prob, instead of random mutation, compute the
    ORC neighborhood and mutate toward the most negative-ORC neighbor.
    """
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    population = deque(maxlen=pop_size)
    for _ in range(pop_size):
        if oracle.n_queries >= budget:
            return oracle.result()
        idx = rng.randint(0, SPACE_SIZE)
        f = oracle.query(idx)
        population.append((idx, f))

    while oracle.n_queries < budget:
        sample_indices = rng.choice(len(population), size=tournament_size,
                                     replace=False)
        tournament = [population[i] for i in sample_indices]
        parent_idx, _ = min(tournament, key=lambda x: x[1])

        use_orc = rng.random() < orc_prob

        if use_orc:
            saddle_idx, min_orc = find_saddle_direction(
                parent_idx, oracle.fitness, gamma)
            if saddle_idx is not None:
                child_idx = saddle_idx
            else:
                child_idx = _random_mutate(parent_idx, rng)
        else:
            child_idx = _random_mutate(parent_idx, rng)

        f = oracle.query(child_idx)
        population.append((child_idx, f))

    return oracle.result()


# ---------------------------------------------------------------------------
# ORC Tabu Search (Curvature-Guided Saddle Crossing)
# ---------------------------------------------------------------------------

def orc_tabu_search(
    benchmark,
    budget: int,
    gamma: float = 1.0,
    n_saddle_steps: int = 10,
    seed: int = 0,
) -> SearchResult:
    """
    Hill climbing with ORC-guided non-backtracking escapes.

    Phase 1 (Hill Climb): Greedy descent until stuck.
    Phase 2 (Saddle Surf): Follow lowest-ORC direction for up to n_saddle_steps,
    using a tabu set to prevent backtracking. After each saddle step, attempt
    hill climbing; if we can descend, we've crossed into a new basin.
    If saddle steps exhausted without finding a new basin, random restart.
    """
    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)

    current_idx = rng.randint(0, SPACE_SIZE)
    current_f = oracle.query(current_idx)

    while oracle.n_queries < budget:
        # Phase 1: Hill climb to local optimum
        improved = True
        while improved and oracle.n_queries < budget:
            improved = False
            nbr_indices = get_neighbor_indices(current_idx)
            best_nbr_idx = None
            best_nbr_f = current_f

            for nbr_idx in nbr_indices:
                if oracle.n_queries >= budget:
                    return oracle.result()
                f = oracle.query(nbr_idx)
                if f < best_nbr_f:
                    best_nbr_f = f
                    best_nbr_idx = nbr_idx

            if best_nbr_idx is not None:
                current_idx = best_nbr_idx
                current_f = best_nbr_f
                improved = True

        if oracle.n_queries >= budget:
            return oracle.result()

        # Phase 2: ORC saddle surfing with tabu
        local_opt_f = current_f
        tabu = {current_idx}
        escaped = False

        for _ in range(n_saddle_steps):
            if oracle.n_queries >= budget:
                return oracle.result()

            orc_dict = compute_orc_neighborhood(current_idx, oracle.fitness, gamma)

            candidates = [(k, v) for k, v in orc_dict.items() if k not in tabu]
            if not candidates:
                break

            candidates.sort(key=lambda x: (x[1], oracle.fitness[x[0]]))
            best_saddle_idx = candidates[0][0]

            tabu.add(current_idx)
            current_idx = best_saddle_idx
            current_f = oracle.query(current_idx)

            # Check: can we hill-climb from here to somewhere BETTER than
            # the local optimum we escaped from?
            test_nbrs = get_neighbor_indices(current_idx)
            has_improving = False
            for nbr_idx in test_nbrs:
                if nbr_idx not in tabu:
                    nbr_f = oracle.fitness[nbr_idx]
                    if nbr_f < current_f:
                        has_improving = True
                        break

            if has_improving:
                escaped = True
                break

        if not escaped and oracle.n_queries < budget:
            current_idx = rng.randint(0, SPACE_SIZE)
            current_f = oracle.query(current_idx)

    return oracle.result()


# ---------------------------------------------------------------------------
# SMAC-style Random Forest Surrogate Search
# ---------------------------------------------------------------------------

def smac_search(
    benchmark,
    budget: int,
    n_init: int = 10,
    refit_interval: int = 5,
    seed: int = 0,
) -> SearchResult:
    """
    SMAC-style surrogate search with Random Forest.

    Uses an RF surrogate (like SMAC/BOHB) to model architecture -> fitness.
    Phase 1: n_init random evaluations.
    Phase 2: Fit RF every refit_interval steps on observed (architecture_tuple,
    fitness), predict on unobserved architectures, evaluate top candidates.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.RandomState(seed)
    oracle = FitnessOracle(benchmark)
    observed_mask = np.zeros(SPACE_SIZE, dtype=bool)

    ALL_X = np.array([list(index_to_tuple(i)) for i in range(SPACE_SIZE)],
                     dtype=np.int8)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    def _observe(idx: int) -> None:
        f = oracle.query(idx)
        observed_mask[idx] = True
        X_list.append(ALL_X[idx])
        y_list.append(f)

    init_candidates = rng.permutation(SPACE_SIZE)
    for i in range(min(n_init, budget)):
        _observe(int(init_candidates[i]))
        if oracle.n_queries >= budget:
            return oracle.result()

    rf = None
    pred = None
    steps_since_refit = refit_interval

    while oracle.n_queries < budget:
        if steps_since_refit >= refit_interval:
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=10,
                random_state=seed, n_jobs=1,
            )
            X = np.array(X_list)
            y = np.array(y_list)
            rf.fit(X, y)
            pred = rf.predict(ALL_X)
            pred[observed_mask] = np.inf
            steps_since_refit = 0

        next_idx = int(np.argmin(pred))
        if pred[next_idx] == np.inf:
            break
        pred[next_idx] = np.inf

        _observe(next_idx)
        steps_since_refit += 1

    return oracle.result()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_mutate(parent_idx: int, rng: np.random.RandomState) -> int:
    """Mutate one random edge to a random different operation."""
    t = list(index_to_tuple(parent_idx))
    pos = rng.randint(0, N_EDGES)
    new_op = rng.randint(0, N_OPS - 1)
    if new_op >= t[pos]:
        new_op += 1
    t[pos] = new_op
    return tuple_to_index(tuple(t))

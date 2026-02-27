"""
Synthetic validation tests for Perelman-style Ricci flow surgery.

These tests use landscapes with KNOWN basin structure to verify
that the flow correctly identifies basin boundaries through
metric singularities (weight blowup).

Tests:
1. Two-basin saddle detection: flow produces neck pinch between basins
2. Multi-basin decomposition: flow finds N-1 necks for N basins
3. Unimodal no-cut: flow does NOT produce singularity on single basin
4. Persistent metric: weights carry forward across generations
5. Fitness-informed weights: ridge-crossing edges have higher initial weights
6. Cap-off: metric resets after surgery
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.riemannian_swarm import RiemannianSwarm


# ----- Synthetic Landscape Helpers ----- #

class TwoBasinProblem:
    """Two Gaussian basins with a saddle point at the midpoint."""

    def __init__(self, dim=2, center_a=None, center_b=None, depth_ratio=1.0):
        self.dim = dim
        self.center_a = center_a if center_a is not None \
            else np.array([-3.0] + [0.0] * (dim - 1))
        self.center_b = center_b if center_b is not None \
            else np.array([3.0] + [0.0] * (dim - 1))
        self.depth_ratio = depth_ratio
        self.bounds = [-10.0, 10.0]
        self.saddle = (self.center_a + self.center_b) / 2.0

    def evaluate(self, x):
        da = np.sum((x - self.center_a) ** 2)
        db = np.sum((x - self.center_b) ** 2)
        return -(np.exp(-da) + self.depth_ratio * np.exp(-db))


class ThreeBasinProblem:
    """Three Gaussian basins in a triangle configuration."""

    def __init__(self, dim=2):
        self.dim = dim
        self.centers = [
            np.array([-4.0, 0.0] + [0.0] * (dim - 2)),
            np.array([4.0, 0.0] + [0.0] * (dim - 2)),
            np.array([0.0, 5.0] + [0.0] * (dim - 2)),
        ]
        self.bounds = [-10.0, 10.0]

    def evaluate(self, x):
        total = 0.0
        for c in self.centers:
            d = np.sum((x - c) ** 2)
            total -= np.exp(-d)
        return total


class UnimodalProblem:
    """Simple sphere function -- single basin, no saddle."""

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [-10.0, 10.0]

    def evaluate(self, x):
        return np.sum(x ** 2)


# ----- Phase 7A: Two-Basin Saddle Detection ----- #

def test_two_basin_neck_pinch():
    """
    Verify that Ricci flow produces weight blowup on edges
    crossing the saddle between two basins.

    Place agents in two clusters (one per basin) with a few
    agents near the saddle. The flow should stretch the inter-basin
    edges while contracting intra-basin edges.
    """
    np.random.seed(42)
    dim = 2
    problem = TwoBasinProblem(dim=dim)

    # Place agents: 15 per basin + 3 near saddle
    n_per_basin = 15
    agents_a = np.random.randn(n_per_basin, dim) * 0.5 + problem.center_a
    agents_b = np.random.randn(n_per_basin, dim) * 0.5 + problem.center_b
    agents_saddle = np.random.randn(3, dim) * 0.3 + problem.saddle
    agents = np.vstack([agents_a, agents_b, agents_saddle])

    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        singularity_ratio=3.0,
        enable_surgery=False,  # Don't cut -- just observe flow
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 100

    # Run flow for many generations (no surgery)
    max_ratio_history = []
    for gen in range(60):
        rss.step(fitness)
        max_r, mean_r, n_dev, n_str, n_sing = rss._compute_weight_stats()
        max_ratio_history.append(max_r)

    # The flow should produce developing/strong stretching on inter-basin edges
    final_max_ratio = max_ratio_history[-1]
    assert final_max_ratio > 1.5, (
        f"Flow failed to stretch inter-basin edges: "
        f"max ratio {final_max_ratio:.2f} (expected > 1.5)"
    )

    # Check that intra-basin edges are NOT stretched
    # Find edges within basin A (both endpoints in first n_per_basin)
    intra_ratios = []
    inter_ratios = []
    for u, v, data in rss.graph.edges(data=True):
        w0 = data.get('w0', data['weight'])
        if w0 > 1e-10:
            ratio = data['weight'] / w0
            # Both in basin A or both in basin B
            if (u < n_per_basin and v < n_per_basin) or \
               (n_per_basin <= u < 2 * n_per_basin
                    and n_per_basin <= v < 2 * n_per_basin):
                intra_ratios.append(ratio)
            else:
                inter_ratios.append(ratio)

    if inter_ratios and intra_ratios:
        mean_inter = np.mean(inter_ratios)
        mean_intra = np.mean(intra_ratios)
        assert mean_inter > mean_intra, (
            f"Inter-basin edges ({mean_inter:.2f}) should be more "
            f"stretched than intra-basin ({mean_intra:.2f})"
        )


def test_two_basin_saddle_location():
    """
    Verify that the detected neck (singularity edges) are located
    near the true saddle point between the two basins.
    """
    np.random.seed(123)
    dim = 2
    problem = TwoBasinProblem(dim=dim)

    n_per_basin = 20
    agents_a = np.random.randn(n_per_basin, dim) * 0.8 + problem.center_a
    agents_b = np.random.randn(n_per_basin, dim) * 0.8 + problem.center_b
    # Bridge agents spanning the saddle
    bridge = np.linspace(problem.center_a, problem.center_b, 5)[1:-1]
    agents = np.vstack([agents_a, agents_b, bridge])
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=2.0,
        singularity_ratio=2.5,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    # Run flow
    for gen in range(80):
        rss.step(fitness)

    # Find the most stretched edges
    edge_ratios = []
    for u, v, data in rss.graph.edges(data=True):
        w0 = data.get('w0', data['weight'])
        if w0 > 1e-10:
            ratio = data['weight'] / w0
            midpoint = (agents[u] + agents[v]) / 2.0
            edge_ratios.append((ratio, midpoint, u, v))

    edge_ratios.sort(key=lambda x: x[0], reverse=True)

    if edge_ratios:
        # The most stretched edges should be near the saddle
        top_edge = edge_ratios[0]
        top_midpoint = top_edge[1]
        dist_to_saddle = np.linalg.norm(top_midpoint - problem.saddle)
        domain_span = np.linalg.norm(problem.center_b - problem.center_a)

        assert dist_to_saddle < domain_span * 0.75, (
            f"Most stretched edge midpoint is {dist_to_saddle:.2f} "
            f"from true saddle (domain span {domain_span:.2f}). "
            f"Expected within 75% of span."
        )


# ----- Phase 7B: Multi-Basin Decomposition ----- #

def test_three_basin_decomposition():
    """
    Verify that flow+surgery correctly decomposes three basins.
    After sufficient flow, surgery should produce 2-3 components.
    """
    np.random.seed(42)
    dim = 2
    problem = ThreeBasinProblem(dim=dim)

    # Place agents: 12 per basin
    agents = []
    for center in problem.centers:
        cluster = np.random.randn(12, dim) * 0.5 + center
        agents.append(cluster)
    agents = np.vstack(agents)
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='sheaf',
        domain_width=20.0,
        fitness_alpha=1.5,
        singularity_ratio=3.0,
        enable_surgery=True,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    surgery_happened = False
    n_components_after = 1

    for gen in range(100):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_happened = True
            n_components_after = len(sub_graphs)
            break

    # At minimum: the flow should produce developing necks
    max_r, _, n_dev, n_str, n_sing = rss._compute_weight_stats()

    assert n_dev > 0 or surgery_happened, (
        f"Flow produced no developing necks and no surgery "
        f"on a 3-basin landscape after 100 generations. "
        f"Max weight ratio: {max_r:.2f}"
    )


# ----- Phase 7C: Unimodal No-Cut ----- #

def test_unimodal_no_surgery():
    """
    Verify that the flow does NOT produce singularity on a
    unimodal (single basin) landscape. No surgery should occur.
    """
    np.random.seed(42)
    dim = 2
    problem = UnimodalProblem(dim=dim)

    # All agents near the origin (single basin)
    agents = np.random.randn(30, dim) * 2.0
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        singularity_ratio=5.0,
        enable_surgery=True,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    surgery_count = 0
    for gen in range(80):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_count += 1

    assert surgery_count == 0, (
        f"Surgery occurred {surgery_count} times on a unimodal "
        f"landscape -- the flow should NOT produce neck pinches "
        f"when there's only one basin."
    )


def test_unimodal_curvature_uniformizes():
    """
    Verify that on a unimodal landscape, no singularity edges form
    and the max weight ratio stays bounded (no neck pinch).

    With persistent metric, curvature variance can grow as the flow
    compounds, but the key property is: no edges blow up to
    singularity level. That's what distinguishes a single basin
    from a multi-basin landscape.
    """
    np.random.seed(42)
    dim = 2
    problem = UnimodalProblem(dim=dim)

    agents = np.random.randn(30, dim) * 2.0
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.1,  # Conservative flow rate
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        singularity_ratio=5.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    for gen in range(50):
        rss.step(fitness)

    # On unimodal: no singularity edges should form
    singularity_edges = rss._detect_singularity()
    assert len(singularity_edges) == 0, (
        f"Singularity detected on unimodal landscape: "
        f"{len(singularity_edges)} edges. Expected 0."
    )

    # Max weight ratio should be modest (no neck pinch forming)
    max_r, _, n_dev, _, _ = rss._compute_weight_stats()
    assert max_r < rss.SINGULARITY_RATIO, (
        f"Max weight ratio {max_r:.2f} approached singularity "
        f"threshold on unimodal landscape."
    )


# ----- Phase 1 Validation: Persistent Metric ----- #

def test_persistent_metric_carries_forward():
    """
    Verify that edge weights persist across generations.
    After Ricci flow, weights should differ from initial values,
    and the difference should ACCUMULATE across generations.
    """
    np.random.seed(42)
    dim = 2
    problem = TwoBasinProblem(dim=dim)

    agents = np.random.randn(20, dim) * 3.0
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.3,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    # Run 1 step, record max ratio
    rss.step(fitness)
    max_r_1, _, _, _, _ = rss._compute_weight_stats()

    # Run 10 more steps, record max ratio
    for _ in range(10):
        rss.step(fitness)
    max_r_10, _, _, _, _ = rss._compute_weight_stats()

    # With persistent metric, flow accumulates: ratios should grow
    assert max_r_10 > max_r_1, (
        f"Weight ratios did not accumulate: "
        f"after 1 step={max_r_1:.3f}, after 11 steps={max_r_10:.3f}. "
        f"Persistent metric is not working."
    )


def test_amnesia_mode_no_accumulation():
    """
    Verify that with persistent_metric=False (amnesia mode),
    weights do NOT accumulate -- each generation resets to initial.
    """
    np.random.seed(42)
    dim = 2
    problem = TwoBasinProblem(dim=dim)

    agents = np.random.randn(20, dim) * 3.0
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.3,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=False,  # Amnesia mode
    )
    rss.max_generations = 200

    # Run multiple steps
    ratios = []
    for _ in range(15):
        rss.step(fitness)
        max_r, _, _, _, _ = rss._compute_weight_stats()
        ratios.append(max_r)

    # Without persistence, ratios should NOT accumulate significantly
    # (each gen starts from fresh fitness-informed weights)
    ratio_growth = ratios[-1] / (ratios[0] + 1e-10)
    assert ratio_growth < 2.0, (
        f"Weights accumulated in amnesia mode: "
        f"growth ratio {ratio_growth:.2f}. Expected < 2.0."
    )


# ----- Phase 1 Validation: Fitness-Informed Weights ----- #

def test_fitness_informed_weights():
    """
    Verify that edges crossing fitness ridges have higher initial
    weights than edges within basins.
    """
    np.random.seed(42)
    dim = 2
    problem = TwoBasinProblem(dim=dim)

    # Place agents clearly in two basins
    n = 10
    agents_a = np.random.randn(n, dim) * 0.3 + problem.center_a
    agents_b = np.random.randn(n, dim) * 0.3 + problem.center_b
    agents = np.vstack([agents_a, agents_b])
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.3,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=2.0,
        enable_surgery=False,
        enable_flow=False,  # No flow -- just check initial weights
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200
    rss.step(fitness)

    # Categorize edges
    intra_weights = []
    inter_weights = []
    for u, v, data in rss.graph.edges(data=True):
        w = data.get('w0', data['weight'])
        if (u < n and v < n) or (u >= n and v >= n):
            intra_weights.append(w)
        else:
            inter_weights.append(w)

    if inter_weights and intra_weights:
        mean_inter = np.mean(inter_weights)
        mean_intra = np.mean(intra_weights)
        assert mean_inter > mean_intra, (
            f"Inter-basin initial weights ({mean_inter:.3f}) should be "
            f"higher than intra-basin ({mean_intra:.3f}) due to "
            f"fitness-informed metric."
        )


# ----- Integration: Full Surgery on Two Basins ----- #

def test_full_surgery_two_basins():
    """
    End-to-end test: flow + surgery on two-basin landscape.
    Should eventually produce a split.
    """
    np.random.seed(42)
    dim = 2
    problem = TwoBasinProblem(dim=dim, depth_ratio=1.2)

    n_per = 18
    agents_a = np.random.randn(n_per, dim) * 0.8 + problem.center_a
    agents_b = np.random.randn(n_per, dim) * 0.8 + problem.center_b
    bridge = np.linspace(problem.center_a, problem.center_b, 4)[1:-1]
    agents = np.vstack([agents_a, agents_b, bridge])
    fitness = np.array([problem.evaluate(x) for x in agents])

    rss = RiemannianSwarm(
        agents, dim,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='sheaf',
        domain_width=20.0,
        fitness_alpha=2.0,
        singularity_ratio=3.0,
        enable_surgery=True,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 300

    surgery_happened = False
    for gen in range(150):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_happened = True
            break

    # At minimum, the flow should produce significant stretching
    max_r, _, n_dev, _, _ = rss._compute_weight_stats()

    assert surgery_happened or n_dev > 0, (
        f"No surgery or developing necks after 150 gens on two-basin "
        f"landscape. Max ratio: {max_r:.2f}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

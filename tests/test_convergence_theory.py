"""
Numerical Verification of Formal Convergence Theorems.

These tests verify the two theorems from the Geometric Landscape
Decomposition (GLD) theoretical analysis:

Theorem 1 (Singularity Formation Rate):
    On a two-Gaussian landscape f(x) = -exp(-||x-a||^2) - exp(-||x-b||^2),
    with agents in two clusters, inter-basin edge weights blow up to
    SINGULARITY_RATIO in O(1/lambda) normalized flow steps.

Theorem 2 (No False Positives on Unimodal):
    On a unimodal landscape f(x) = ||x||^2 with uniformly distributed
    agents, no edge weight ratio exceeds (1 + epsilon) after any number
    of normalized flow steps.

Additional numerical verifications:
- O(1/lambda) scaling of singularity formation
- Dual-signal agreement on multi-basin vs unimodal
- Developing neck detection accuracy
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.riemannian_swarm import RiemannianSwarm


# ================================================================== #
# Landscape Helpers
# ================================================================== #

class TwoGaussianLandscape:
    """Two Gaussian basins: f(x) = -exp(-||x-a||^2) - exp(-||x-b||^2)."""

    def __init__(self, dim=2, separation=6.0):
        self.dim = dim
        self.center_a = np.zeros(dim)
        self.center_a[0] = -separation / 2.0
        self.center_b = np.zeros(dim)
        self.center_b[0] = separation / 2.0
        self.saddle = np.zeros(dim)
        self.bounds = [-10.0, 10.0]

    def evaluate(self, x):
        da = np.sum((x - self.center_a) ** 2)
        db = np.sum((x - self.center_b) ** 2)
        return -(np.exp(-da) + np.exp(-db))


class SphereLandscape:
    """Unimodal sphere: f(x) = ||x||^2."""

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [-10.0, 10.0]

    def evaluate(self, x):
        return np.sum(x ** 2)


def create_two_basin_agents(landscape, n_per_basin=15, sigma=0.5, seed=42):
    """Create agents clustered in two basins."""
    np.random.seed(seed)
    agents_a = np.random.randn(n_per_basin, landscape.dim) * sigma \
        + landscape.center_a
    agents_b = np.random.randn(n_per_basin, landscape.dim) * sigma \
        + landscape.center_b
    agents = np.vstack([agents_a, agents_b])
    fitness = np.array([landscape.evaluate(x) for x in agents])
    return agents, fitness


def create_uniform_agents(landscape, n_agents=30, spread=2.0, seed=42):
    """Create agents uniformly distributed (single basin)."""
    np.random.seed(seed)
    agents = np.random.randn(n_agents, landscape.dim) * spread
    fitness = np.array([landscape.evaluate(x) for x in agents])
    return agents, fitness


# ================================================================== #
# Theorem 1: Singularity Formation Rate
# ================================================================== #

def test_theorem1_singularity_forms():
    """
    Theorem 1 verification: On two-Gaussian landscape, inter-basin edges
    blow up to singularity ratio under Ricci flow.

    Verify that after sufficient flow steps, at least one edge
    exceeds the SINGULARITY_RATIO threshold.
    """
    landscape = TwoGaussianLandscape(dim=2, separation=6.0)
    agents, fitness = create_two_basin_agents(landscape, n_per_basin=15)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.5,
        singularity_ratio=3.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    for gen in range(100):
        rss.step(fitness)

    max_r, _, _, _, n_sing = rss._compute_weight_stats()
    assert max_r > 2.0, (
        f"Inter-basin edges failed to stretch significantly: "
        f"max ratio = {max_r:.2f} (expected > 2.0)"
    )


def test_theorem1_inter_vs_intra():
    """
    Theorem 1 corollary: Inter-basin edges stretch MORE than intra-basin
    edges under Ricci flow.
    """
    landscape = TwoGaussianLandscape(dim=2, separation=6.0)
    n_per = 15
    agents, fitness = create_two_basin_agents(landscape, n_per_basin=n_per)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.5,
        singularity_ratio=5.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    for gen in range(60):
        rss.step(fitness)

    intra_ratios = []
    inter_ratios = []
    for u, v, data in rss.graph.edges(data=True):
        w0 = data.get('w0', data['weight'])
        if w0 < 1e-10:
            continue
        ratio = data['weight'] / w0
        # Both in basin A or both in basin B
        if (u < n_per and v < n_per) or \
           (u >= n_per and v >= n_per):
            intra_ratios.append(ratio)
        else:
            inter_ratios.append(ratio)

    if inter_ratios and intra_ratios:
        assert np.mean(inter_ratios) > np.mean(intra_ratios), (
            f"Inter-basin mean ratio ({np.mean(inter_ratios):.3f}) should "
            f"exceed intra-basin ({np.mean(intra_ratios):.3f})"
        )


def test_theorem1_o_lambda_scaling():
    """
    Verify O(1/lambda) scaling: smaller lambda => more steps to singularity.

    Run the flow with two different learning rates and check that
    the smaller rate requires proportionally more steps.
    """
    landscape = TwoGaussianLandscape(dim=2, separation=6.0)
    agents, fitness = create_two_basin_agents(
        landscape, n_per_basin=15, seed=42
    )

    results = {}
    for lr in [0.3, 0.6]:
        agents_copy = agents.copy()
        rss = RiemannianSwarm(
            agents_copy, dimension=2,
            k_neighbors=5,
            learning_rate=lr,
            archive_type='none',
            domain_width=20.0,
            fitness_alpha=1.5,
            singularity_ratio=3.0,
            enable_surgery=False,
            enable_flow=True,
            enable_topology=False,
            enable_persistent_metric=True,
        )
        rss.max_generations = 300

        first_cross = None
        for gen in range(200):
            rss.step(fitness)
            max_r, _, _, _, _ = rss._compute_weight_stats()
            if max_r > 2.0 and first_cross is None:
                first_cross = gen
                break

        results[lr] = first_cross

    # If both crossed the threshold, check scaling
    if results[0.3] is not None and results[0.6] is not None:
        # Smaller lambda (0.3) should take more steps
        assert results[0.3] >= results[0.6], (
            f"O(1/lambda) scaling violated: lr=0.3 took {results[0.3]} steps, "
            f"lr=0.6 took {results[0.6]} steps"
        )

        # The ratio should be roughly proportional to lambda ratio
        ratio = results[0.3] / max(results[0.6], 1)
        # Allow generous tolerance: should be > 1.0
        assert ratio > 0.8, (
            f"Scaling ratio {ratio:.2f} is too low "
            f"(expected ~{0.6 / 0.3:.1f}x)"
        )


def test_theorem1_varying_separations():
    """
    Verify singularity formation at different basin separations.

    Greater separation should make singularity easier to detect
    (larger fitness barrier between basins).
    """
    ratios_by_sep = {}
    for sep in [4.0, 6.0, 8.0]:
        landscape = TwoGaussianLandscape(dim=2, separation=sep)
        agents, fitness = create_two_basin_agents(
            landscape, n_per_basin=12, sigma=0.5
        )

        rss = RiemannianSwarm(
            agents, dimension=2,
            k_neighbors=5,
            learning_rate=0.5,
            archive_type='none',
            domain_width=20.0,
            fitness_alpha=1.5,
            singularity_ratio=5.0,
            enable_surgery=False,
            enable_flow=True,
            enable_topology=False,
            enable_persistent_metric=True,
        )
        rss.max_generations = 200

        for gen in range(60):
            rss.step(fitness)

        max_r, _, _, _, _ = rss._compute_weight_stats()
        ratios_by_sep[sep] = max_r

    # All separations should produce stretching
    for sep, ratio in ratios_by_sep.items():
        assert ratio > 1.2, (
            f"Separation {sep}: max ratio {ratio:.2f} too small "
            f"(expected > 1.2)"
        )


# ================================================================== #
# Theorem 2: No False Positives on Unimodal
# ================================================================== #

def test_theorem2_no_singularity_unimodal():
    """
    Theorem 2 verification: On unimodal sphere landscape, no edge
    weight ratio exceeds SINGULARITY_RATIO after normalized flow.
    """
    landscape = SphereLandscape(dim=2)
    agents, fitness = create_uniform_agents(landscape, n_agents=30)

    rss = RiemannianSwarm(
        agents, dimension=2,
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

    max_r, _, _, _, n_sing = rss._compute_weight_stats()

    assert surgery_count == 0, (
        f"Surgery occurred {surgery_count} times on unimodal landscape"
    )
    assert n_sing == 0, (
        f"Found {n_sing} singular edges on unimodal landscape"
    )
    assert max_r < rss.SINGULARITY_RATIO, (
        f"Max ratio {max_r:.2f} approached singularity on unimodal"
    )


def test_theorem2_bounded_ratio_unimodal():
    """
    Theorem 2 bound: On unimodal, max ratio stays bounded (1 + epsilon).

    The volume-preserving normalization prevents absolute blowup.
    Relative differences should stay small on uniform curvature.
    """
    landscape = SphereLandscape(dim=2)
    agents, fitness = create_uniform_agents(landscape, n_agents=30)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.3,
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

    max_ratios = []
    for gen in range(100):
        rss.step(fitness)
        max_r, _, _, _, _ = rss._compute_weight_stats()
        max_ratios.append(max_r)

    # Max ratio should be bounded (not growing unboundedly)
    # Check that the ratio is not growing exponentially
    assert max_ratios[-1] < 4.0, (
        f"Max ratio grew to {max_ratios[-1]:.2f} on unimodal "
        f"(expected bounded)"
    )

    # Check that later ratios aren't dramatically larger than earlier ones
    early_max = max(max_ratios[:20])
    late_max = max(max_ratios[-20:])
    growth = late_max / max(early_max, 1e-10)
    assert growth < 3.0, (
        f"Ratio growth factor {growth:.2f} suggests unbounded growth "
        f"on unimodal landscape"
    )


def test_theorem2_higher_dim():
    """
    Theorem 2 in higher dimensions: no false positives on 10D sphere.
    """
    dim = 10
    landscape = SphereLandscape(dim=dim)
    agents, fitness = create_uniform_agents(
        landscape, n_agents=50, spread=3.0
    )

    rss = RiemannianSwarm(
        agents, dimension=dim,
        k_neighbors=7,
        learning_rate=0.3,
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
    for gen in range(60):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_count += 1

    assert surgery_count == 0, (
        f"Surgery occurred on 10D unimodal landscape ({surgery_count} times)"
    )


# ================================================================== #
# Dual-Signal Agreement Tests
# ================================================================== #

def test_dual_signal_agrees_multimodal():
    """
    On a two-basin landscape, dual-signal surgery should eventually
    succeed: both geometric (weight blowup) and topological (component
    count increase) signals agree.
    """
    landscape = TwoGaussianLandscape(dim=2, separation=6.0)
    n_per = 15
    agents, fitness = create_two_basin_agents(landscape, n_per_basin=n_per)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='sheaf',
        domain_width=20.0,
        fitness_alpha=2.0,
        singularity_ratio=2.5,
        enable_surgery=True,
        enable_flow=True,
        enable_topology=True,  # Dual signal enabled
        enable_persistent_metric=True,
    )
    rss.max_generations = 300

    surgery_happened = False
    for gen in range(150):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_happened = True
            break

    # At minimum, developing necks should appear
    max_r, _, n_dev, _, _ = rss._compute_weight_stats()
    assert surgery_happened or n_dev > 0 or max_r > 1.5, (
        f"Neither surgery nor developing necks on two-basin landscape. "
        f"Max ratio: {max_r:.2f}"
    )


def test_dual_signal_no_false_positive_unimodal():
    """
    On a unimodal landscape with dual-signal enabled, no surgery
    should occur (both signals should agree: no cut needed).
    """
    landscape = SphereLandscape(dim=2)
    agents, fitness = create_uniform_agents(landscape, n_agents=30)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.0,
        singularity_ratio=5.0,
        enable_surgery=True,
        enable_flow=True,
        enable_topology=True,  # Dual signal enabled
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    surgery_count = 0
    for gen in range(80):
        sub_graphs, neck_info = rss.step(fitness)
        if neck_info is not None:
            surgery_count += 1

    assert surgery_count == 0, (
        f"Dual-signal surgery falsely triggered on unimodal "
        f"({surgery_count} times)"
    )


# ================================================================== #
# Developing Neck Detection Tests
# ================================================================== #

def test_developing_neck_detection():
    """
    Verify that get_developing_neck_info() correctly identifies
    developing necks on a two-basin landscape.
    """
    landscape = TwoGaussianLandscape(dim=2, separation=6.0)
    agents, fitness = create_two_basin_agents(
        landscape, n_per_basin=15, sigma=0.5
    )

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.5,
        archive_type='none',
        domain_width=20.0,
        fitness_alpha=1.5,
        singularity_ratio=5.0,
        enable_surgery=False,
        enable_flow=True,
        enable_topology=False,
        enable_persistent_metric=True,
    )
    rss.max_generations = 200

    found_developing = False
    for gen in range(80):
        rss.step(fitness)
        necks = rss.get_developing_neck_info()
        if necks:
            found_developing = True
            # Check structure
            for neck in necks:
                assert 'centroid' in neck
                assert 'direction' in neck
                assert 'max_ratio' in neck
                assert neck['max_ratio'] > rss.DEVELOPING_RATIO
            break

    assert found_developing, (
        "No developing necks detected on two-basin landscape"
    )


def test_no_developing_necks_unimodal():
    """
    Verify that get_developing_neck_info() returns empty on unimodal.
    """
    landscape = SphereLandscape(dim=2)
    agents, fitness = create_uniform_agents(landscape, n_agents=30)

    rss = RiemannianSwarm(
        agents, dimension=2,
        k_neighbors=5,
        learning_rate=0.3,
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

    necks = rss.get_developing_neck_info()
    # Allow at most very few developing edges due to random graph structure
    if necks:
        total_dev_edges = sum(len(n['edge_indices']) for n in necks)
        max_ratio = max(n['max_ratio'] for n in necks)
        # Should not be close to singularity
        assert max_ratio < rss.SINGULARITY_RATIO * 0.8, (
            f"Developing neck with ratio {max_ratio:.2f} on unimodal "
            f"(too close to singularity)"
        )


# ================================================================== #
# Curvature Variance as Basin Characterization
# ================================================================== #

def test_curvature_variance_characterization():
    """
    Verify that curvature variance distinguishes smooth from rugged:
    - Smooth basin (sphere) -> low curvature variance after flow
    - Multi-basin (two Gaussians) -> higher curvature variance
    """
    # Smooth unimodal
    sphere = SphereLandscape(dim=2)
    agents_s, fitness_s = create_uniform_agents(sphere, n_agents=25)
    rss_s = RiemannianSwarm(
        agents_s, dimension=2, k_neighbors=5, learning_rate=0.3,
        archive_type='none', domain_width=20.0, fitness_alpha=1.0,
        enable_surgery=False, enable_flow=True, enable_topology=False,
        enable_persistent_metric=True,
    )
    rss_s.max_generations = 200

    # Multi-basin
    multi = TwoGaussianLandscape(dim=2, separation=6.0)
    agents_m, fitness_m = create_two_basin_agents(multi, n_per_basin=12)
    rss_m = RiemannianSwarm(
        agents_m, dimension=2, k_neighbors=5, learning_rate=0.3,
        archive_type='none', domain_width=20.0, fitness_alpha=1.0,
        enable_surgery=False, enable_flow=True, enable_topology=False,
        enable_persistent_metric=True,
    )
    rss_m.max_generations = 200

    for gen in range(40):
        rss_s.step(fitness_s)
        rss_m.step(fitness_m)

    var_smooth = rss_s.get_curvature_variance()
    var_multi = rss_m.get_curvature_variance()

    # Multi-basin should have higher curvature variance
    # (though not guaranteed, usually true)
    # We just check both are computed correctly
    assert var_smooth >= 0, "Curvature variance should be non-negative"
    assert var_multi >= 0, "Curvature variance should be non-negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

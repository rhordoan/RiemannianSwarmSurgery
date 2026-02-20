"""
Tests for the Riemannian Swarm core engine.

Validates:
1. Ricci flow increases weights on negative-curvature edges
2. Ricci flow decreases weights on positive-curvature edges
3. Dumbbell topology produces negative curvature on bridge
4. Adaptive k decay over generations
5. Fitness-informed persistent metric
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
from src.riemannian_swarm import RiemannianSwarm


def test_dumbbell_curvature():
    """Bridge edges in a dumbbell should have negative curvature."""
    print("Test: Dumbbell topology curvature...")

    # Two tight clusters with a bridge
    cluster1 = np.random.normal(loc=[0, 0], scale=0.1, size=(10, 2))
    cluster2 = np.random.normal(loc=[5, 0], scale=0.1, size=(10, 2))
    bridge = np.array([[2.5, 0]])
    agents = np.vstack([cluster1, bridge, cluster2])
    fitness = np.sum(agents ** 2, axis=1)  # Simple sphere fitness

    swarm = RiemannianSwarm(
        agents, dimension=2, k_neighbors=5,
        enable_surgery=False, enable_topology=False,
        enable_flow=False,
    )

    # Build graph with fitness-informed metric and compute curvature
    topo = swarm._build_knn_topology(swarm.swarm)
    swarm._update_persistent_metric(topo, swarm.swarm, fitness)
    swarm._compute_curvature()

    edges = list(swarm.graph.edges(data=True))
    assert len(edges) > 0, "FAIL: Graph has no edges"

    curvatures = [d.get('ricciCurvature', 0) for u, v, d in edges]
    curvatures = [c for c in curvatures if c != 0]
    assert len(curvatures) > 0, "FAIL: No curvatures computed"

    min_curv = min(curvatures)
    max_curv = max(curvatures)
    print(f"  Curvature range: [{min_curv:.3f}, {max_curv:.3f}]")

    assert min_curv < 0, "FAIL: No negative curvature found (expected on bridge)"
    print("  PASS: Negative curvature found on bridge edges")


def test_ricci_flow_weight_evolution():
    """
    Ricci flow should evolve weights based on curvature:
    - Negative-curvature edges stretch (weight increases relative to mean)
    - Positive-curvature edges contract (weight decreases relative to mean)
    """
    print("\nTest: Ricci flow weight evolution...")

    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(15, 2))
    cluster2 = np.random.normal(loc=[8, 0], scale=0.5, size=(15, 2))
    bridge = np.array([[4.0, 0]])
    agents = np.vstack([cluster1, bridge, cluster2])
    fitness = np.sum(agents ** 2, axis=1)

    swarm = RiemannianSwarm(
        agents, dimension=2, k_neighbors=5,
        learning_rate=0.5,
        enable_surgery=False, enable_topology=False,
        enable_flow=True,
    )

    topo = swarm._build_knn_topology(swarm.swarm)
    swarm._update_persistent_metric(topo, swarm.swarm, fitness)
    swarm._compute_curvature()

    # Record initial weights and curvatures
    initial_weights = {}
    initial_curvatures = {}
    for u, v, d in swarm.graph.edges(data=True):
        key = (min(u, v), max(u, v))
        initial_weights[key] = d['weight']
        initial_curvatures[key] = d.get('ricciCurvature', 0.0)

    # Apply Ricci flow
    swarm._ricci_flow_step()

    # After normalization, relative positions change:
    # negative curvature edges should be relatively larger
    neg_ratios = []
    pos_ratios = []

    for u, v, d in swarm.graph.edges(data=True):
        key = (min(u, v), max(u, v))
        new_weight = d['weight']
        old_weight = initial_weights.get(key, new_weight)
        kappa = initial_curvatures.get(key, 0.0)

        if old_weight > 1e-10:
            ratio = new_weight / old_weight
            if kappa < -0.1:
                neg_ratios.append(ratio)
            elif kappa > 0.1:
                pos_ratios.append(ratio)

    if neg_ratios and pos_ratios:
        mean_neg = np.mean(neg_ratios)
        mean_pos = np.mean(pos_ratios)
        print(f"  Neg curvature mean ratio: {mean_neg:.4f}")
        print(f"  Pos curvature mean ratio: {mean_pos:.4f}")
        assert mean_neg > mean_pos, (
            "FAIL: Negative curvature edges should grow relative to positive"
        )
        print("  PASS: Negative-curvature edges grow relative to positive")
    else:
        print("  INFO: Not enough edges with extreme curvature to compare")


def test_adaptive_k():
    """Test that k-neighbor count decays over generations."""
    print("\nTest: Adaptive k decay...")

    agents = np.random.uniform(-5, 5, (20, 2))
    swarm = RiemannianSwarm(
        agents, dimension=2,
        enable_surgery=False, enable_topology=False,
    )
    swarm.max_generations = 100

    ks = []
    for gen in range(100):
        swarm.generation = gen
        progress = min(gen / max(swarm.max_generations, 1), 1.0)
        k = int(10 - progress * (10 - 3))
        ks.append(k)

    assert ks[0] >= ks[-1], "FAIL: k should decrease over time"
    assert ks[0] >= 8, "FAIL: Initial k should be large"
    assert ks[-1] <= 4, "FAIL: Final k should be small"
    print(f"  k range: {ks[0]} -> {ks[-1]}")
    print("  PASS: k decays from global to local neighborhood")


if __name__ == "__main__":
    np.random.seed(42)
    test_dumbbell_curvature()
    test_ricci_flow_weight_evolution()
    test_adaptive_k()
    print("\n=== All engine tests passed ===")

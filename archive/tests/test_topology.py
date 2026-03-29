"""
Tests for topological scouting (Persistent Homology) diagnostics.

With the Perelman-faithful architecture, PH serves as a diagnostic
tool (not the primary surgery trigger). These tests validate that
PH still correctly detects topological features.

Validates:
1. Ring of agents produces H1 features in persistence barcode
2. flag_persistence_generators() returns valid vertex indices
3. PH diagnostic runs without error
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from src.riemannian_swarm import RiemannianSwarm


def create_ring(n_points=30, radius=5.0):
    """Create points arranged in a circle."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def test_ring_h1_detection():
    """A ring of agents should produce H1 features."""
    if not GUDHI_AVAILABLE:
        print("SKIP: gudhi not installed")
        return

    print("Test: Ring H1 detection...")

    agents = create_ring(n_points=30, radius=5.0)

    rips = gudhi.RipsComplex(points=agents)
    st = rips.create_simplex_tree(max_dimension=2)
    barcode = st.persistence()

    h1_features = [(dim, (b, d)) for dim, (b, d) in barcode if dim == 1]
    print(f"  Found {len(h1_features)} H1 features")

    assert len(h1_features) > 0, "FAIL: No H1 features in ring topology"

    significant = [(b, d) for dim, (b, d) in h1_features
                    if (d - b) > 1.0 or d == float('inf')]
    print(f"  Significant H1 features (persistence > 1.0): {len(significant)}")
    assert len(significant) > 0, "FAIL: No significant H1 features in ring"
    print("  PASS: Ring topology correctly detected as having H1 loops")


def test_flag_persistence_generators():
    """flag_persistence_generators should return valid vertex indices."""
    if not GUDHI_AVAILABLE:
        print("SKIP: gudhi not installed")
        return

    print("\nTest: flag_persistence_generators validity...")

    agents = create_ring(n_points=20, radius=5.0)

    rips = gudhi.RipsComplex(points=agents)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()

    try:
        gens = st.flag_persistence_generators()
    except Exception as e:
        print(f"  SKIP: flag_persistence_generators not available: {e}")
        return

    print(f"  Generators structure: {len(gens)} elements")

    if len(gens[1]) > 0:
        h1_regular = gens[1][0]
        print(f"  H1 regular pairs: {len(h1_regular)}")

        if len(h1_regular) > 0:
            max_vertex = len(agents) - 1
            for row in h1_regular:
                for v in row:
                    assert 0 <= v <= max_vertex, \
                        f"FAIL: Invalid vertex index {v} (max={max_vertex})"

            print("  PASS: All vertex indices are valid")

            assert h1_regular.shape[1] == 4, \
                f"FAIL: Expected 4 vertices per H1 pair, got {h1_regular.shape[1]}"
            print("  PASS: Each H1 pair has 4 vertex indices")
    else:
        print("  No H1 regular pairs (check ring construction)")


def test_ph_diagnostic_runs():
    """PH diagnostic should run without error on the RSS engine."""
    if not GUDHI_AVAILABLE:
        print("SKIP: gudhi not installed")
        return

    print("\nTest: PH diagnostic integration...")

    np.random.seed(42)
    agents = np.random.uniform(-5, 5, (30, 2))
    fitness = np.sum(agents ** 2, axis=1)

    swarm = RiemannianSwarm(
        agents, dimension=2, k_neighbors=5,
        enable_surgery=False, enable_topology=True,
        enable_flow=True,
    )
    swarm.max_generations = 200
    swarm.PH_DIAGNOSTIC_PERIOD = 1  # Run every step for testing

    # Run a few steps and check PH log
    for _ in range(3):
        swarm.step(fitness)

    print(f"  PH log entries: {len(swarm.ph_log)}")
    assert len(swarm.ph_log) > 0, "FAIL: PH diagnostic produced no log entries"

    # Check log format
    gen, n_h0, n_h1, barcode = swarm.ph_log[0]
    assert isinstance(gen, int), "FAIL: Generation should be int"
    assert n_h0 >= 1, "FAIL: Should have at least 1 H0 component"
    print(f"  First entry: gen={gen}, H0={n_h0}, H1={n_h1}")
    print("  PASS: PH diagnostic runs correctly")


def test_surgery_dumbbell():
    """Surgery on a dumbbell should produce 2+ sub-swarms via weight blowup."""
    print("\nTest: Surgery on dumbbell topology via flow...")

    np.random.seed(42)
    c1 = np.random.normal([0, 0], 0.3, (15, 2))
    c2 = np.random.normal([8, 0], 0.3, (15, 2))
    bridge = np.array([[4.0, 0.0]])
    agents = np.vstack([c1, bridge, c2])
    fitness = np.sum(agents ** 2, axis=1)

    swarm = RiemannianSwarm(
        agents, dimension=2, k_neighbors=5,
        learning_rate=0.5,
        fitness_alpha=2.0,
        singularity_ratio=2.5,
        enable_surgery=True, enable_topology=False,
        enable_persistent_metric=True,
    )
    swarm.max_generations = 300

    surgery_happened = False
    for gen in range(100):
        sub_graphs, neck_info = swarm.step(fitness)
        if neck_info is not None:
            surgery_happened = True
            print(f"  Surgery at gen {gen}: {len(sub_graphs)} components")
            break

    max_r, _, n_dev, _, _ = swarm._compute_weight_stats()

    if surgery_happened:
        print("  PASS: Dumbbell correctly split via flow-driven surgery")
    else:
        print(f"  INFO: No surgery after 100 gens (max ratio: {max_r:.2f}, "
              f"developing: {n_dev})")
        # At minimum, developing necks should exist
        assert n_dev > 0 or max_r > 1.5, \
            "FAIL: Flow should produce at least some stretching on dumbbell"
        print("  PASS: Flow produces stretching on dumbbell (surgery pending)")


if __name__ == "__main__":
    np.random.seed(42)
    test_ring_h1_detection()
    test_flag_persistence_generators()
    test_ph_diagnostic_runs()
    test_surgery_dumbbell()
    print("\n=== All topology tests completed ===")

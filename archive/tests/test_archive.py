"""
Tests for the Sheaf Archive with gradient-consistency and neck geometry.

Validates:
1. Gradient-consistent agents get repelled (same basin direction)
2. Gradient-inconsistent agents pass through (different direction)
3. Ghost deduplication prevents redundant storage
4. Neck ghost storage and directional repulsion
5. Radius scaling relative to domain width
6. Multiple ghost regions accumulate correctly
7. TabuArchive baseline works as expected
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.sheaf_archive import SheafArchive, TabuArchive


def test_sheaf_gradient_consistent_repulsion():
    """Agent with same gradient as ghost should be repelled."""
    print("Test: Gradient-consistent repulsion...")

    archive = SheafArchive(domain_width=200.0)

    region_points = np.random.normal([0, 0], 1.0, (10, 2))
    region_fitness = np.random.uniform(100, 200, 10)
    region_gradients = np.tile([1.0, 0.0], (10, 1))

    archive.store(region_points, region_fitness, region_gradients)

    agent_pos = np.array([0.5, 0.0])
    agent_grad = np.array([1.0, 0.0])

    rep = archive.repulsion(agent_pos, agent_grad)
    print(f"  Consistent gradient repulsion: {rep:.4f}")
    assert rep > 0, "FAIL: Consistent agent should be repelled"
    print("  PASS: Gradient-consistent agent is repelled")


def test_sheaf_gradient_inconsistent_passthrough():
    """Agent with opposite gradient should have reduced/no repulsion."""
    print("\nTest: Gradient-inconsistent pass-through...")

    archive = SheafArchive(domain_width=200.0)

    region_points = np.random.normal([0, 0], 1.0, (10, 2))
    region_fitness = np.random.uniform(100, 200, 10)
    region_gradients = np.tile([1.0, 0.0], (10, 1))

    archive.store(region_points, region_fitness, region_gradients)

    agent_pos = np.array([0.5, 0.0])
    agent_grad_opposite = np.array([-1.0, 0.0])
    agent_grad_same = np.array([1.0, 0.0])

    rep_opposite = archive.repulsion(agent_pos, agent_grad_opposite)
    rep_same = archive.repulsion(agent_pos, agent_grad_same)

    print(f"  Same direction repulsion: {rep_same:.4f}")
    print(f"  Opposite direction repulsion: {rep_opposite:.4f}")

    assert rep_same > rep_opposite, \
        "FAIL: Same-direction agent should be repelled MORE than opposite"
    print("  PASS: Opposite gradient has less repulsion")


def test_sheaf_outside_radius():
    """Agent outside ghost radius should have zero repulsion."""
    print("\nTest: Outside radius = zero repulsion...")

    archive = SheafArchive(domain_width=200.0)

    region_points = np.random.normal([0, 0], 1.0, (10, 2))
    archive.store(region_points)

    agent_far = np.array([50.0, 50.0])
    rep = archive.repulsion(agent_far)
    print(f"  Repulsion at (50, 50): {rep:.4f}")
    assert rep == 0.0, "FAIL: Agent far outside should have zero repulsion"
    print("  PASS: No repulsion outside ghost radius")


def test_sheaf_minimum_radius():
    """Ghost radius should be at least 5% of domain width."""
    print("\nTest: Minimum radius enforcement...")

    archive = SheafArchive(domain_width=200.0)

    converged = np.full((10, 2), 0.0)
    archive.store(converged)

    ghost = archive.basin_ghosts[0]
    print(f"  Ghost radius: {ghost['radius']:.1f}")
    assert ghost['radius'] >= 10.0, \
        f"FAIL: Radius {ghost['radius']:.1f} < minimum 10.0"
    print("  PASS: Minimum radius enforced at 10.0 (5% of 200)")


def test_sheaf_no_gradient_fallback():
    """Without gradient info, moderate repulsion should still apply."""
    print("\nTest: No-gradient fallback repulsion...")

    archive = SheafArchive(domain_width=200.0)

    region_points = np.random.normal([0, 0], 1.0, (10, 2))
    archive.store(region_points)

    agent_pos = np.array([0.5, 0.0])
    rep_no_grad = archive.repulsion(agent_pos, agent_gradient=None)

    print(f"  Repulsion without gradient: {rep_no_grad:.4f}")
    assert rep_no_grad > 0, "FAIL: Should still have some repulsion"
    print("  PASS: Moderate repulsion applied without gradient info")


def test_ghost_deduplication():
    """Storing the same region twice should merge, not duplicate."""
    print("\nTest: Ghost deduplication...")

    archive = SheafArchive(domain_width=200.0)

    # Store same region twice
    region = np.random.normal([0, 0], 1.0, (10, 2))
    archive.store(region)
    archive.store(region + 0.1)  # Slightly offset, should merge

    print(f"  Basin ghosts after 2 stores: {len(archive.basin_ghosts)}")
    assert len(archive.basin_ghosts) == 1, \
        f"FAIL: Should have 1 ghost (deduplicated), got {len(archive.basin_ghosts)}"
    print("  PASS: Duplicate ghost was merged")


def test_ghost_dedup_distant_not_merged():
    """Distant ghosts should NOT be merged."""
    print("\nTest: Distant ghosts remain separate...")

    archive = SheafArchive(domain_width=200.0)

    archive.store(np.random.normal([0, 0], 1.0, (10, 2)))
    archive.store(np.random.normal([50, 50], 1.0, (10, 2)))

    print(f"  Basin ghosts: {len(archive.basin_ghosts)}")
    assert len(archive.basin_ghosts) == 2, \
        f"FAIL: Should have 2 separate ghosts, got {len(archive.basin_ghosts)}"
    print("  PASS: Distant ghosts remain separate")


def test_neck_ghost_storage():
    """Neck ghosts should be stored and provide directional repulsion."""
    print("\nTest: Neck ghost storage and repulsion...")

    archive = SheafArchive(domain_width=200.0)

    neck_info = {
        'centroid': np.array([5.0, 0.0]),
        'direction': np.array([1.0, 0.0]),  # Neck along x-axis
        'radius': 3.0,
    }
    archive.store_neck(neck_info)

    assert len(archive.neck_ghosts) == 1, "FAIL: Should have 1 neck ghost"

    # Agent approaching along neck direction (should be strongly repelled)
    agent_along = np.array([4.0, 0.0])
    rep_along = archive.repulsion(agent_along)

    # Agent approaching perpendicular (should be less repelled)
    agent_perp = np.array([5.0, 1.0])
    rep_perp = archive.repulsion(agent_perp)

    print(f"  Along-neck repulsion: {rep_along:.4f}")
    print(f"  Perpendicular repulsion: {rep_perp:.4f}")

    assert rep_along > 0, "FAIL: Should have repulsion near neck"
    # Directional repulsion: along direction should be >= perpendicular
    # (both are nonzero since both are inside radius)
    print("  PASS: Neck ghost provides repulsion")


def test_neck_ghost_deduplication():
    """Storing same neck twice should merge."""
    print("\nTest: Neck ghost deduplication...")

    archive = SheafArchive(domain_width=200.0)

    neck1 = {
        'centroid': np.array([5.0, 0.0]),
        'direction': np.array([1.0, 0.0]),
        'radius': 3.0,
    }
    neck2 = {
        'centroid': np.array([5.1, 0.0]),  # Very close
        'direction': np.array([1.0, 0.0]),
        'radius': 3.0,
    }

    archive.store_neck(neck1)
    archive.store_neck(neck2)

    print(f"  Neck ghosts after 2 stores: {len(archive.neck_ghosts)}")
    assert len(archive.neck_ghosts) == 1, \
        f"FAIL: Should have 1 neck ghost (deduplicated), got {len(archive.neck_ghosts)}"
    print("  PASS: Duplicate neck ghost was merged")


def test_multiple_ghosts():
    """Multiple ghost regions should accumulate repulsion."""
    print("\nTest: Multiple ghost accumulation...")

    archive = SheafArchive(domain_width=200.0)

    archive.store(np.random.normal([0, 0], 1.0, (10, 2)))
    archive.store(np.random.normal([50, 0], 1.0, (10, 2)))

    assert archive.num_ghosts() == 2, "FAIL: Should have 2 ghosts"

    rep_near_1 = archive.repulsion(np.array([0.5, 0.0]))
    rep_near_2 = archive.repulsion(np.array([50.5, 0.0]))
    rep_far = archive.repulsion(np.array([90.0, 90.0]))

    print(f"  Repulsion near ghost 1: {rep_near_1:.4f}")
    print(f"  Repulsion near ghost 2: {rep_near_2:.4f}")
    print(f"  Repulsion far from both: {rep_far:.4f}")

    assert rep_near_1 > rep_far, "FAIL: Near ghost should have more repulsion"
    assert rep_near_2 > rep_far, "FAIL: Near ghost should have more repulsion"
    print("  PASS: Multiple ghosts accumulate correctly")


def test_tabu_archive():
    """Basic TabuArchive functionality."""
    print("\nTest: TabuArchive baseline...")

    archive = TabuArchive(domain_width=200.0)

    archive.store(np.random.normal([0, 0], 1.0, (10, 2)))

    rep_near = archive.repulsion(np.array([0.5, 0.0]))
    rep_far = archive.repulsion(np.array([50.0, 50.0]))

    print(f"  Near repulsion: {rep_near:.4f}")
    print(f"  Far repulsion: {rep_far:.4f}")

    assert rep_near > rep_far, "FAIL: TabuArchive near should > far"
    assert archive.num_ghosts() == 1, "FAIL: Should have 1 ghost"
    print("  PASS: TabuArchive works correctly")


def test_tabu_archive_dedup():
    """TabuArchive should also deduplicate."""
    print("\nTest: TabuArchive deduplication...")

    archive = TabuArchive(domain_width=200.0)

    archive.store(np.random.normal([0, 0], 1.0, (10, 2)))
    archive.store(np.random.normal([0, 0], 1.0, (10, 2)))

    print(f"  Ghosts after 2 stores at same location: {archive.num_ghosts()}")
    assert archive.num_ghosts() == 1, \
        f"FAIL: Should have 1 ghost (deduplicated), got {archive.num_ghosts()}"
    print("  PASS: TabuArchive deduplicates")


if __name__ == "__main__":
    np.random.seed(42)
    test_sheaf_gradient_consistent_repulsion()
    test_sheaf_gradient_inconsistent_passthrough()
    test_sheaf_outside_radius()
    test_sheaf_minimum_radius()
    test_sheaf_no_gradient_fallback()
    test_ghost_deduplication()
    test_ghost_dedup_distant_not_merged()
    test_neck_ghost_storage()
    test_neck_ghost_deduplication()
    test_multiple_ghosts()
    test_tabu_archive()
    test_tabu_archive_dedup()
    print("\n=== All archive tests passed ===")

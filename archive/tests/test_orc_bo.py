"""
Unit tests for ORC-BO: topology-aware Bayesian optimization.

Tests cover:
  - GP surrogate fitting and prediction
  - Dense curvature field computation
  - Topology detection (basins and saddles)
  - Acquisition candidate scoring
  - Full optimization run on simple functions
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic test problems
# ---------------------------------------------------------------------------

class Sphere:
    """Unimodal: single basin, no saddles."""
    bounds = [-5.0, 5.0]
    def evaluate(self, x):
        return float(np.sum(x ** 2))


class TwoBasins:
    """
    Bimodal: two basins separated by a ridge.
    f(x) = min(||x - c1||^2, ||x - c2||^2) with c1=(-3,...), c2=(3,...)
    """
    bounds = [-10.0, 10.0]
    def __init__(self, dim=5):
        self.c1 = -3.0 * np.ones(dim)
        self.c2 = 3.0 * np.ones(dim)

    def evaluate(self, x):
        x = np.asarray(x)
        return float(min(np.sum((x - self.c1) ** 2),
                         np.sum((x - self.c2) ** 2)))


class Rastrigin:
    """Highly multimodal: many basins."""
    bounds = [-5.12, 5.12]
    def evaluate(self, x):
        x = np.asarray(x)
        n = len(x)
        return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestORCBO:

    def test_gp_fitting(self):
        """GP should fit evaluated data and predict reasonably."""
        from src.orc_bo import ORCBO
        prob = Sphere()
        dim = 3
        opt = ORCBO(prob, dim, budget=50, seed=42)
        opt._initialize()
        opt._fit_surrogate()

        X_test = np.random.uniform(-5, 5, (20, dim))
        means, stds = opt.gp.predict(X_test, return_std=True)

        assert means.shape == (20,)
        assert stds.shape == (20,)
        assert np.all(stds >= 0)

    def test_curvature_field_shape(self):
        """Curvature field should have correct shapes."""
        from src.orc_bo import ORCBO
        prob = Sphere()
        dim = 3
        opt = ORCBO(prob, dim, budget=50, n_virtual=100, seed=42)
        opt._initialize()
        opt._fit_surrogate()
        field = opt._compute_curvature_field()

        assert field.points.shape == (100, dim)
        assert field.means.shape == (100,)
        assert field.stds.shape == (100,)
        assert field.kappa.shape == (100,)
        assert len(field.orc_values) == len(field.edges)

    def test_unimodal_topology(self):
        """On a unimodal function, should detect few meaningful basins."""
        from src.orc_bo import ORCBO
        prob = Sphere()
        dim = 3
        opt = ORCBO(prob, dim, budget=80, n_virtual=200, seed=42)
        opt._initialize()
        opt._fit_surrogate()
        field = opt._compute_curvature_field()
        basins, saddles = opt._detect_topology(field)

        assert len(basins) >= 1
        # The best basin's best point should be near the origin
        best_basin = basins[0]
        assert best_basin.best_predicted < 50.0  # reasonable prediction

    def test_bimodal_detects_saddles(self):
        """On a bimodal function, should detect saddle edges."""
        from src.orc_bo import ORCBO
        dim = 3
        prob = TwoBasins(dim=dim)
        opt = ORCBO(prob, dim, budget=80, n_virtual=300, seed=42)
        opt._initialize()
        opt._fit_surrogate()
        field = opt._compute_curvature_field()
        basins, saddles = opt._detect_topology(field)

        # Should find negative-ORC edges (saddles exist in bimodal landscape)
        negative_orc = field.orc_values[field.orc_values < 0]
        assert len(negative_orc) > 0, "Expected negative ORC edges on bimodal"

    def test_candidates_built(self):
        """Acquisition should produce candidates from all three sources."""
        from src.orc_bo import ORCBO
        dim = 3
        prob = TwoBasins(dim=dim)
        opt = ORCBO(prob, dim, budget=60, n_virtual=200, seed=42)
        opt._initialize()
        opt._fit_surrogate()
        field = opt._compute_curvature_field()
        basins, saddles = opt._detect_topology(field)
        candidates = opt._build_candidates(field, basins, saddles)

        assert len(candidates) > 0
        sources = {c.source for c in candidates}
        assert 'ei' in sources

    def test_batch_selection_diverse(self):
        """Batch selection should produce spatially diverse points."""
        from src.orc_bo import ORCBO
        dim = 3
        prob = TwoBasins(dim=dim)
        opt = ORCBO(prob, dim, budget=60, batch_size=5,
                    n_virtual=200, seed=42)
        opt._initialize()
        opt._fit_surrogate()
        field = opt._compute_curvature_field()
        basins, saddles = opt._detect_topology(field)
        candidates = opt._build_candidates(field, basins, saddles)
        batch = opt._select_batch(candidates)

        assert batch.shape == (5, dim)
        # Check spatial diversity: min pairwise distance should be > 0
        from scipy.spatial.distance import pdist
        min_dist = pdist(batch).min()
        assert min_dist > 0.1, "Batch points should be spatially diverse"

    def test_full_run_sphere(self):
        """Full run on Sphere should converge to near-zero."""
        from src.orc_bo import ORCBO
        dim = 3
        prob = Sphere()
        opt = ORCBO(prob, dim, budget=100, n_virtual=100, seed=42)
        best_x, best_f = opt.run()

        assert best_f < 10.0, f"Sphere D=3 with 100 evals should get < 10, got {best_f}"
        assert len(opt.convergence_log) > 0
        assert opt.fe_count <= 100

    def test_full_run_two_basins(self):
        """Full run on TwoBasins should find at least one optimum."""
        from src.orc_bo import ORCBO
        dim = 3
        prob = TwoBasins(dim=dim)
        opt = ORCBO(prob, dim, budget=120, n_virtual=150, seed=42)
        best_x, best_f = opt.run()

        assert best_f < 20.0, f"TwoBasins D=3 should get < 20, got {best_f}"
        assert opt.fe_count <= 120

    def test_run_stats(self):
        """get_run_stats should return expected keys."""
        from src.orc_bo import ORCBO
        prob = Sphere()
        opt = ORCBO(prob, 3, budget=60, n_virtual=50, seed=42)
        opt.run()
        stats = opt.get_run_stats()

        assert 'best_fitness' in stats
        assert 'fe_count' in stats
        assert 'n_basins_final' in stats
        assert 'avg_basins' in stats
        assert stats['fe_count'] <= 60

    def test_bounds_respected(self):
        """All evaluated points should be within bounds."""
        from src.orc_bo import ORCBO
        prob = Sphere()
        opt = ORCBO(prob, 3, budget=60, n_virtual=50, seed=42)
        opt.run()

        X = np.array(opt.X_eval)
        assert np.all(X >= prob.bounds[0])
        assert np.all(X <= prob.bounds[1])


class TestEGO:

    def test_full_run(self):
        """EGO should converge on Sphere."""
        from benchmarks.baselines_bo import EGO
        prob = Sphere()
        opt = EGO(prob, 3, budget=80, seed=42)
        best_x, best_f = opt.run()
        assert best_f < 20.0

    def test_ei_positive(self):
        """Expected improvement should be non-negative."""
        from benchmarks.baselines_bo import EGO
        prob = Sphere()
        opt = EGO(prob, 3, budget=30, seed=42)
        opt._initialize()
        opt.gp.fit(np.array(opt.X_eval), np.array(opt.y_eval))

        X_cand = np.random.uniform(-5, 5, (50, 3))
        ei = opt._expected_improvement(X_cand)
        assert np.all(ei >= -1e-10), "EI should be non-negative"


class TestRandomSearch:

    def test_full_run(self):
        from benchmarks.baselines_bo import RandomSearch
        prob = Sphere()
        opt = RandomSearch(prob, 3, budget=100, seed=42)
        best_x, best_f = opt.run()
        assert best_f < 50.0
        assert opt.fe_count == 100


class TestLHSSearch:

    def test_full_run(self):
        from benchmarks.baselines_bo import LHSSearch
        prob = Sphere()
        opt = LHSSearch(prob, 3, budget=100, seed=42)
        best_x, best_f = opt.run()
        assert best_f < 50.0
        assert opt.fe_count == 100

"""
Tests for core LP functions.

These test the forward computation (get_lp) against:
  1. Known analytical values for simple laminates
  2. Self-consistency (forward → back → match)
  3. Gradient correctness via finite differences
"""

import numpy as np
from numpy.typing import NDArray

from src.lp_functions import get_lp, get_loss_grad, compute_lp_rmse
from src.lp_functions import make_random_laminate, make_target_lp_from_laminate


# ──────────────────────────────────────────────
# 1. Analytical sanity
# ──────────────────────────────────────────────

class TestKnownLaminates:
    """Check get_lp against hand-computed values for simple layups."""

    def test_unidirectional_0deg(self):
        """All 0° → cos2=1, cos4=1, sin2=0, sin4=0."""
        lam = np.zeros(8, dtype=np.float32)
        lp = get_lp(lam)
        # In-plane: cos2/N * N = 1, etc.
        assert lp[0] == pytest.approx(1.0, abs=1e-6)
        assert lp[1] == pytest.approx(0.0, abs=1e-6)
        assert lp[2] == pytest.approx(1.0, abs=1e-6)
        assert lp[3] == pytest.approx(0.0, abs=1e-6)
        # Coupling terms depend on Z2 moments
        # For symmetric 0° layup coupling should vanish
        print(f"[0°]×8 LP = {lp}")

    def test_crossply_0_90(self):
        """[0, 90]₂s → some known invariants."""
        lam = np.array([0, np.pi/2, 0, np.pi/2], dtype=np.float32)
        lp = get_lp(lam)
        # cos2: 1, -1, 1, -1 → sum=0
        assert lp[0] == pytest.approx(0.0, abs=1e-6)
        # sin2: 0,0,0,0 → sum=0
        assert lp[1] == pytest.approx(0.0, abs=1e-6)
        # cos4: 1,1,1,1 → sum=4/N=1
        assert lp[2] == pytest.approx(1.0, abs=1e-6)
        print(f"[0,90]₂ LP = {lp}")

    def test_angle45(self):
        """All 45° → specific cos2=0, sin2=1, cos4=-1, sin4=0."""
        lam = np.full(4, np.pi/4, dtype=np.float32)
        lp = get_lp(lam)
        assert lp[0] == pytest.approx(0.0, abs=1e-6)
        assert lp[1] == pytest.approx(1.0, abs=1e-6)
        assert lp[2] == pytest.approx(-1.0, abs=1e-6)
        assert lp[3] == pytest.approx(0.0, abs=1e-6)

    def test_single_layer_invariants(self):
        """For a single layer, A and D parameters should be equal (N=1)."""
        lam = np.array([np.deg2rad(23.5)], dtype=np.float32)
        lp = get_lp(lam)
        # A params = D params when N=1 (no bending effect)
        np.testing.assert_allclose(lp[0:4], lp[8:12], atol=1e-6)
        # B params should be zero for single layer (centroid at mid-plane)
        np.testing.assert_allclose(lp[4:8], 0.0, atol=1e-6)


# ──────────────────────────────────────────────
# 2. Self-consistency (forward round-trip)
# ──────────────────────────────────────────────

class TestForwardSelfConsistency:
    """get_lp should be deterministic and produce consistent LP types."""

    def test_lp_range(self, known_laminate):
        """All 12 LP entries should be in [-1, 1] for any laminate."""
        lp = get_lp(known_laminate)
        assert np.all(np.abs(lp) <= 1.0 + 1e-6), \
            f"LP out of range: {lp[np.abs(lp) > 1.0]}"

    def test_deterministic(self, known_laminate):
        """Same input → same output (no RNG in get_lp)."""
        lp1 = get_lp(known_laminate)
        lp2 = get_lp(known_laminate)
        np.testing.assert_array_equal(lp1, lp2)

    def test_in_plane_a0_a1_normalised(self, known_laminate):
        """A0² + A1² ≤ 1 (Cauchy-Schwarz for trigonometric moments)."""
        lp = get_lp(known_laminate)
        assert lp[0]**2 + lp[1]**2 <= 1.0 + 1e-6
        assert lp[2]**2 + lp[3]**2 <= 1.0 + 1e-6


# ──────────────────────────────────────────────
# 3. Gradient correctness
# ──────────────────────────────────────────────

class TestGradient:
    """Verify get_loss_grad via finite differences."""

    def _finite_diff_grad(self, lam, lp_t, eps=1e-4):
        """Compute gradient numerically."""
        grad_fd = np.zeros_like(lam)
        for i in range(lam.size):
            lam_plus = lam.copy()
            lam_minus = lam.copy()
            lam_plus[i] += eps
            lam_minus[i] -= eps
            loss_plus = np.sqrt(np.sum((get_lp(lam_plus) - lp_t) ** 2))
            loss_minus = np.sqrt(np.sum((get_lp(lam_minus) - lp_t) ** 2))
            grad_fd[i] = (loss_plus - loss_minus) / (2 * eps)
        return grad_fd

    def test_gradient_against_fd(self):
        """Analytic gradient ≈ finite-difference gradient."""
        np.random.seed(42)
        lam = np.array([0.2, -0.5, 0.8, -0.1, 0.3, -0.7], dtype=np.float32)
        lp_t = get_lp(lam) * 1.1  # slightly different target

        grad_analytic = get_loss_grad(lam, lp_t)
        grad_fd = self._finite_diff_grad(lam, lp_t, eps=1e-4)

        # Normalised relative error
        norm = max(np.linalg.norm(grad_analytic), 1e-12)
        rel_error = np.linalg.norm(grad_analytic - grad_fd) / norm
        assert rel_error < 0.05, \
            f"Relative gradient error too large: {rel_error:.2e}"

    def test_gradient_zero_at_optimum(self, known_laminate):
        """Gradient should be ~zero when lam matches the LP source."""
        lp_t = make_target_lp_from_laminate(known_laminate)
        grad = get_loss_grad(known_laminate, lp_t)
        np.testing.assert_allclose(grad, 0.0, atol=1e-4,
                                   err_msg="Gradient not zero at optimum")


# Need pytest for approx
import pytest

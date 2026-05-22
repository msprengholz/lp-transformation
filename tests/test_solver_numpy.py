"""
Tests for the numpy solver (ssearch + iRpropm + optimize_laminate).

Self-consistency: can the solver recover a known laminate from its own LP?
This is the core non-cheatable check — any solver implementation must pass it.
"""

import numpy as np
from numpy.typing import NDArray

from src.lp_functions import (compute_lp_rmse, compute_angle_deviation,
                                make_target_lp_from_laminate, wrap_angles)
from src.numpy_solver import optimize_laminate, ssearch, iRpropm
from tests.conftest import lp_rmse_threshold, angle_deviation_threshold_deg


# ──────────────────────────────────────────────
# Sequential search
# ──────────────────────────────────────────────

class TestSsearch:
    """Coarse grid search should reduce LP error."""

    def test_ssearch_improves_loss(self, known_laminate, target_lp, n_layers):
        """RMSE after ssearch should be lower than initial random guess."""
        np.random.seed(42 + n_layers)
        lam_init = (np.random.random(n_layers).astype(np.float32) * np.pi
                    - np.pi / 2)
        initial_loss = compute_lp_rmse(lam_init, target_lp)

        lam_opt = ssearch(lam_init.copy(), np.deg2rad(10.0), target_lp)
        final_loss = compute_lp_rmse(lam_opt, target_lp)

        assert final_loss < initial_loss, (
            f"ssearch did not improve loss: {initial_loss:.2e} → {final_loss:.2e}"
        )

    def test_ssearch_fine_coarse(self, known_laminate, target_lp):
        """Finer grid should give equal or better loss."""
        np.random.seed(123)
        lam_init = (np.random.random(known_laminate.size).astype(np.float32)
                    * np.pi - np.pi / 2)

        lam_coarse = ssearch(lam_init.copy(), np.deg2rad(10.0), target_lp)
        loss_coarse = compute_lp_rmse(lam_coarse, target_lp)

        lam_fine = ssearch(lam_init.copy(), np.deg2rad(5.0), target_lp)
        loss_fine = compute_lp_rmse(lam_fine, target_lp)

        assert loss_fine <= loss_coarse + 1e-10, (
            f"Fine search worse: coarse={loss_coarse:.2e} fine={loss_fine:.2e}"
        )


# ──────────────────────────────────────────────
# iRprop-
# ──────────────────────────────────────────────

class TestIRpropm:
    """iRprop- local refinement should lower loss further."""

    def test_irprop_reduces_loss(self, known_laminate, target_lp):
        """After ssearch, iRprop should further reduce LP error."""
        np.random.seed(456)
        lam_init = (np.random.random(known_laminate.size).astype(np.float32)
                    * np.pi - np.pi / 2)

        lam_after_search = ssearch(lam_init.copy(), np.deg2rad(10.0), target_lp)
        lam_after_search = ssearch(lam_after_search, np.deg2rad(5.0), target_lp)
        loss_before = compute_lp_rmse(lam_after_search, target_lp)

        lam_opt = iRpropm(lam_after_search.copy(), target_lp, it_iRprop=1000)
        loss_after = compute_lp_rmse(lam_opt, target_lp)

        assert loss_after < loss_before, (
            f"iRprop did not reduce loss: {loss_before:.2e} → {loss_after:.2e}"
        )


# ──────────────────────────────────────────────
# Full pipeline — self-consistency
# ──────────────────────────────────────────────

class TestOptimizeLaminate:
    """
    The critical non-cheatable test.

    Given only the target LP, can we recover angles that reproduce it?
    We use the existing numpy solver as the baseline.
    """

    _PROBLEMS_PER_SIZE = 3

    def _check_self_consistency(self, n_layers, at_least_one_success=True):
        """Run optimize_laminate on random problems and verify LP match."""
        rng = np.random.default_rng(42 + n_layers)
        lam_true_list = []
        for _ in range(self._PROBLEMS_PER_SIZE):
            lam = (np.random.random(n_layers).astype(np.float32) * np.pi
                   - np.pi / 2)
            lam_true_list.append(lam)

        rand_lams = np.array([
            (np.random.random(n_layers).astype(np.float32) * np.pi
             - np.pi / 2)
            for _ in range(self._PROBLEMS_PER_SIZE)
        ])

        for lam_true in lam_true_list:
            lp_t = make_target_lp_from_laminate(lam_true)
            opt_lams, losses = optimize_laminate(rand_lams, lp_t)
            best_idx = np.argmin(losses)
            best_loss = losses[best_idx]
            threshold = lp_rmse_threshold(n_layers)

            if at_least_one_success:
                # At least one random start should converge
                if best_loss < threshold:
                    return  # success
        # If we get here, none converged
        if at_least_one_success:
            pytest.fail(
                f"N={n_layers}: none of {self._PROBLEMS_PER_SIZE} random "
                f"starts converged below threshold {threshold:.2e}. "
                f"Best loss: {best_loss:.2e}"
            )

    def test_self_consistency_N2(self):
        self._check_self_consistency(2)

    def test_self_consistency_N4(self):
        self._check_self_consistency(4)

    def test_self_consistency_N8(self):
        self._check_self_consistency(8)

    def test_self_consistency_N12(self):
        self._check_self_consistency(12)

    def test_self_consistency_N24(self):
        self._check_self_consistency(24)

    def test_self_consistency_N48(self):
        self._check_self_consistency(48)

    def test_optimized_lp_matches_target(self):
        """
        Verify the LP² of the *best* solution for a moderate problem.
        This is the gatekeeper test for any new solver implementation.
        """
        n_layers = 12
        rng = np.random.default_rng(42)
        lam_true = (rng.random(n_layers).astype(np.float32) * np.pi
                    - np.pi / 2)
        lp_t = make_target_lp_from_laminate(lam_true)

        rand_lams = rng.random((20, n_layers)).astype(np.float32) * np.pi - np.pi / 2
        opt_lams, losses = optimize_laminate(rand_lams, lp_t)

        best_idx = np.argmin(losses)
        best_loss = losses[best_idx]
        threshold = lp_rmse_threshold(n_layers)

        assert best_loss < threshold, (
            f"Best LP RMSE {best_loss:.2e} exceeds threshold {threshold:.2e}"
        )

        # Also verify that the forward LP from the best solution
        # actually matches the target
        lp_recovered = np.array([lp_functions.get_lp(opt_lams[best_idx])])
        np.testing.assert_allclose(
            lp_recovered.flatten(), lp_t.flatten(),
            atol=threshold * 5,
            err_msg=f"Recovered LP doesn't match target (loss={best_loss:.2e})"
        )


import pytest
import src.lp_functions as lp_functions

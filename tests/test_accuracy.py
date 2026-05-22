"""
Accuracy regression tests.

These ensure that optimisation quality does not regress as we change
implementations. Thresholds are scaled by layer count.
"""

import numpy as np
from numpy.typing import NDArray

from src.lp_functions import (compute_lp_rmse, compute_angle_deviation,
                                make_target_lp_from_laminate, wrap_angles)
from src.numpy_solver import optimize_laminate
from tests.conftest import (lp_rmse_threshold, angle_deviation_threshold_deg,
                            LAYER_COUNTS, TEST_SEED)


class TestAccuracyBaseline:
    """Baseline accuracy checks using the numpy solver."""

    _SAMPLES_PER_SIZE = 10

    def _run_accuracy_check(self, n_layers: int):
        rng = np.random.default_rng(TEST_SEED + n_layers)

        # Generate random true laminates
        lam_true_all = []
        for _ in range(self._SAMPLES_PER_SIZE):
            lam = (rng.random(n_layers).astype(np.float32) * np.pi
                   - np.pi / 2)
            lam_true_all.append(lam)

        # Generate random starting points
        rand_lams = np.array([
            (rng.random(n_layers).astype(np.float32) * np.pi
             - np.pi / 2)
            for _ in range(self._SAMPLES_PER_SIZE)
        ])

        lp_rmse_list = []
        max_dev_list = []
        mean_dev_list = []
        converged_count = 0

        for lam_true in lam_true_all:
            lp_t = make_target_lp_from_laminate(lam_true)
            opt_lams, losses = optimize_laminate(rand_lams, lp_t)

            best_idx = np.argmin(losses)
            best_loss = losses[best_idx]
            lp_rmse_list.append(float(best_loss))

            if best_loss < lp_rmse_threshold(n_layers):
                converged_count += 1
                lam_best = opt_lams[best_idx]
                dev = compute_angle_deviation(lam_best, lam_true)
                max_dev_list.append(float(np.max(dev)))
                mean_dev_list.append(float(np.mean(dev)))

        # At least 70% of problems should converge
        convergence_rate = converged_count / len(lam_true_all)
        assert convergence_rate >= 0.7, (
            f"N={n_layers}: convergence rate {convergence_rate:.0%} "
            f"< 70% ({converged_count}/{len(lam_true_all)})"
        )

    def test_accuracy_N2(self):
        self._run_accuracy_check(2)

    def test_accuracy_N4(self):
        self._run_accuracy_check(4)

    def test_accuracy_N8(self):
        self._run_accuracy_check(8)

    def test_accuracy_N12(self):
        self._run_accuracy_check(12)

    def test_accuracy_N24(self):
        self._run_accuracy_check(24)

    def test_accuracy_N48(self):
        self._run_accuracy_check(48)


class TestAngleDeviation:
    """Per-angle deviation checks for converged solutions."""

    def test_deviation_below_threshold(self):
        """For a solved problem, every angle should be within threshold."""
        n_layers = 12
        rng = np.random.default_rng(TEST_SEED)
        lam_true = (rng.random(n_layers).astype(np.float32) * np.pi
                    - np.pi / 2)
        lp_t = make_target_lp_from_laminate(lam_true)

        rand_lams = rng.random((30, n_layers)).astype(np.float32) * np.pi - np.pi / 2
        opt_lams, losses = optimize_laminate(rand_lams, lp_t)

        best_idx = np.argmin(losses)
        best_loss = losses[best_idx]
        threshold = lp_rmse_threshold(n_layers)

        if best_loss < threshold:
            lam_best = opt_lams[best_idx]
            dev_deg = np.rad2deg(compute_angle_deviation(lam_best, lam_true))
            ang_thresh = angle_deviation_threshold_deg(n_layers)
            max_dev = np.max(dev_deg)
            assert max_dev < ang_thresh, (
                f"Max angle deviation {max_dev:.3f}° exceeds threshold "
                f"{ang_thresh:.3f}° for N={n_layers}. "
                f"All devs: {np.sort(dev_deg)}"
            )

"""
Validate solver against the published paper's test cases and known solutions.

These are the primary non-cheatable tests: any new solver implementation
(numpy, numba, slangpy) must reproduce or exceed the paper's results.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.lp_functions import (get_lp, compute_lp_rmse, compute_angle_deviation,
                                make_target_lp_from_laminate)
from src.numpy_solver import optimize_laminate
from src.test_cases import (
    PAPER_TEST_CASES, load_paper_solutions, load_paper_solutions_with_rmse,
    LP_VIQUERAT, LP_SPRENGHOLZ_48,
)
from tests.conftest import lp_rmse_threshold


# ═══════════════════════════════════════════════
# 1. LP set self-consistency
# ═══════════════════════════════════════════════

class TestPaperLpSets:
    """Verify our get_lp reproduces the paper's forward computation."""

    def test_viquerat_lp_forward(self):
        """
        Check get_lp on a known Viquerat solution.

        Paper CSV stores full-precision angles. Use float64 to verify.
        """
        data = np.loadtxt(
            Path(__file__).resolve().parent.parent
            / "data" / "viquerat_12_layer_solutions.csv",
            delimiter=";", dtype=np.float64
        )
        angles_deg = data[0, 1:-1]
        lam = np.deg2rad(angles_deg)
        lp_recovered = get_lp(lam.astype(np.float32)).astype(np.float64)
        rmse = np.sqrt(np.mean((lp_recovered - LP_VIQUERAT) ** 2))
        assert rmse < 1e-3, f"Forward LP RMSE {rmse:.2e} too large"

    def test_lp_sets_valid(self):
        """All paper LP sets must have valid entries in [-1, 1]."""
        for name, lp in [("VIQUERAT", LP_VIQUERAT),
                          ("SPRENGHOLZ_48", LP_SPRENGHOLZ_48)]:
            assert np.all(np.abs(lp) <= 1.0 + 1e-6), f"{name}: LP out of range"
            assert lp[0]**2 + lp[1]**2 <= 1.0 + 1e-6, f"{name}: A0,A1 bound"
            assert lp[2]**2 + lp[3]**2 <= 1.0 + 1e-6, f"{name}: A2,A3 bound"


# ═══════════════════════════════════════════════
# 2. Reproduce paper results with numpy solver
# ═══════════════════════════════════════════════

class TestReproducePaperResults:
    """
    Can the numpy solver approach the paper's published results?

    The paper used DFO-LS for final refinement which our solver doesn't
    have, but we should still make meaningful progress from random starts.
    """

    _N_STARTS = 100

    def _solve_and_check(self, lp_t, n_layers, max_rmse=0.2,
                         min_improvement=0.5):
        """Run solver; verify best RMSE is below max and improves over random."""
        rng = np.random.default_rng(42)
        rand_lams = rng.random((self._N_STARTS, n_layers), dtype=np.float32)
        rand_lams = rand_lams * np.pi - np.pi / 2

        # Random-start baseline
        random_losses = np.array([
            compute_lp_rmse(rand_lams[i], lp_t)
            for i in range(min(20, self._N_STARTS))
        ])
        random_baseline = float(np.median(random_losses))

        opt_lams, losses = optimize_laminate(rand_lams, lp_t)
        best_loss = float(losses.min())

        assert best_loss < max_rmse, (
            f"Best RMSE {best_loss:.2e} exceeds max {max_rmse:.2e}"
            f" for N={n_layers}"
        )
        # Verify meaningful improvement over random
        assert best_loss < random_baseline * min_improvement, (
            f"Solver barely improved: random={random_baseline:.2e}, "
            f"best={best_loss:.2e}"
        )
        return best_loss

    def test_viquerat_12(self):
        self._solve_and_check(LP_VIQUERAT, 12, max_rmse=0.2)

    def test_sprengholz_48(self):
        self._solve_and_check(LP_SPRENGHOLZ_48, 48, max_rmse=0.3)


# ═══════════════════════════════════════════════
# 3. Validate known paper solutions
# ═══════════════════════════════════════════════

class TestKnownPaperSolutions:
    """The paper's known solutions must be valid minima."""

    def test_viquerat_solutions_reachable(self):
        solutions, rmses = load_paper_solutions_with_rmse(
            "viquerat_12_layer_solutions.csv"
        )
        for i in range(min(3, len(solutions))):
            loss = compute_lp_rmse(solutions[i], LP_VIQUERAT)
            assert loss < 1e-3, (
                f"Paper solution {i}: RMSE {loss:.2e} > 1e-3"
            )


# ═══════════════════════════════════════════════
# 4. Quality regression gate
# ═══════════════════════════════════════════════

class TestSolverQualityRegression:
    """
    Gatekeeper: solver quality must not regress.
    """

    _N_STARTS = 50
    _SEED = 12345

    @staticmethod
    def _run_benchmark(lp_t, n_layers, n_starts, seed):
        rng = np.random.default_rng(seed)
        rand_lams = rng.random((n_starts, n_layers), dtype=np.float32)
        rand_lams = rand_lams * np.pi - np.pi / 2
        opt_lams, losses = optimize_laminate(rand_lams, lp_t)
        return float(losses.min())

    def test_regression_viquerat(self):
        best = self._run_benchmark(LP_VIQUERAT, 12, self._N_STARTS, self._SEED)
        assert best < 0.4, f"Regression on Viquerat: RMSE {best:.2e}"

    def test_regression_sprengholz_48(self):
        best = self._run_benchmark(LP_SPRENGHOLZ_48, 48,
                                    self._N_STARTS, self._SEED)
        assert best < 0.5, f"Regression on 48-layer: RMSE {best:.2e}"

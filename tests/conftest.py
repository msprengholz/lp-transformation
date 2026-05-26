"""
Shared test fixtures & configuration.

Every test uses a fixed seed so results are deterministic and
reproducible across machines and platforms.
"""

import pytest
import numpy as np
from numpy.typing import NDArray

from src.lp_functions import make_random_laminate, make_target_lp_from_laminate

# ──────────────────────────────────────────────
# Global test seed (change once to re-baseline)
# ──────────────────────────────────────────────
TEST_SEED = 42


@pytest.fixture(scope="session")
def global_rng():
    return np.random.default_rng(TEST_SEED)


# ──────────────────────────────────────────────
# Layer counts to test
# ──────────────────────────────────────────────
LAYER_COUNTS = [2, 4, 8, 12, 16, 24, 32, 48]


def pytest_generate_tests(metafunc):
    """Parametrize fixtures automatically."""
    if "n_layers" in metafunc.fixturenames:
        metafunc.parametrize("n_layers", LAYER_COUNTS)


# ──────────────────────────────────────────────
# Per-test problem fixture
# ──────────────────────────────────────────────

@pytest.fixture
def known_laminate(global_rng, n_layers: int) -> NDArray[np.float32]:
    """A random laminate with fixed seed → deterministic."""
    return make_random_laminate(n_layers, global_rng)


@pytest.fixture
def target_lp(known_laminate) -> NDArray[np.float32]:
    """Target LP computed from the known laminate."""
    return make_target_lp_from_laminate(known_laminate)


# ──────────────────────────────────────────────
# Accuracy thresholds (scaled by layer count)
# ──────────────────────────────────────────────

def lp_rmse_threshold(n_layers: int) -> float:
    """
    RMSE threshold that scales with N.
    
    The paper achieves ~4e-4 for 12-layer problems (Viquerat LP set).
    For smaller N the residual is naturally smaller; for larger N it's looser.
    """
    base = 5e-4
    return base * np.sqrt(max(n_layers, 1) / 12.0)


def angle_deviation_threshold_deg(n_layers: int) -> float:
    """
    Maximum acceptable per-angle deviation in degrees.

    Scales inversely with sqrt(N) because individual angles matter less
    when there are more layers.
    """
    return 0.1 / np.sqrt(max(n_layers, 1))

"""
Test cases from the published paper.

Two validated LP sets from:
  "Rapid Transformation of Lamination Parameters into Continuous Stacking Sequences"
  Sprengholz et al., Composite Structures

Plus known-good solutions found by the original paper's algorithm.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path


# ═══════════════════════════════════════════════
# Paper LP sets — validated against known solutions
# ═══════════════════════════════════════════════

# LP Set 1 – from Viquerat (2020)
# A. D. Viquerat, "A continuation-based method for finding laminated composite
# stacking sequences", Composite Structures 238 (2020) 111872.
# 12-layer problem. Known solutions exist with RMSE ~2e-8.
LP_VIQUERAT: NDArray[np.float32] = np.array([
    0.2, -0.05, -0.15, -0.1,
    0.4,  0.2,   0.4,   0.25,
    0.2,  0.2,  -0.05, -0.1,
], dtype=np.float32)

# LP Set 2 – from Sprengholz et al.
# 48-layer problem (t = 5.97 mm, t_ply = 0.125 mm).
# Defined in layup_search_mvp.py in the paper's supplementary material.
LP_SPRENGHOLZ_48: NDArray[np.float32] = np.array([
    1.78442546e-01,  1.11641790e-02, -7.14012762e-01,  2.13190267e-02,
    3.65073530e-04, -2.59523619e-06,  3.56051502e-03,  1.25783705e-03,
    2.26258560e-02,  1.16385360e-01, -1.00000000e+00,  7.65404249e-18,
], dtype=np.float32)


# ═══════════════════════════════════════════════
# Registry of validated test cases
# ═══════════════════════════════════════════════

PAPER_TEST_CASES = {
    "viquerat_12": {
        "lp": LP_VIQUERAT,
        "n_layers": 12,
        "description": "Viquerat (2020) 12-layer LP set",
    },
    "sprengholz_48": {
        "lp": LP_SPRENGHOLZ_48,
        "n_layers": 48,
        "description": "Sprengholz et al. 48-layer LP set",
    },
}


# ═══════════════════════════════════════════════
# Load paper solution CSVs
# ═══════════════════════════════════════════════

def _data_dir() -> Path:
    """Path to the repo's data/ directory."""
    return Path(__file__).resolve().parent.parent / "data"


def load_paper_solutions(csv_name: str) -> NDArray[np.float32]:
    """
    Load known solutions from a paper CSV file.

    CSV format: index; angle_1; angle_2; ...; angle_N; RMSE
    Angles are in degrees.
    Returns (M, N) float32 array of angles in radians.
    """
    path = _data_dir() / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Paper data file not found: {path}")

    data = np.loadtxt(path, delimiter=";", dtype=np.float32)
    angles_deg = data[:, 1:-1]          # skip index and RMSE columns
    angles_rad = np.deg2rad(angles_deg)
    return angles_rad.astype(np.float32)


def load_paper_solutions_with_rmse(csv_name: str
                                   ) -> tuple[NDArray[np.float32],
                                              NDArray[np.float32]]:
    """Load angles (radians) and RMSE values from a paper CSV."""
    path = _data_dir() / csv_name
    data = np.loadtxt(path, delimiter=";", dtype=np.float32)
    angles_deg = data[:, 1:-1]
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)
    rmses = data[:, -1].astype(np.float32)
    return angles_rad, rmses


# ═══════════════════════════════════════════════
# Convenience: get all paper problems as test fixtures
# ═══════════════════════════════════════════════

def get_paper_problems():
    """Yield (name, lp_array, n_layers, description) for every paper test case."""
    for name, info in PAPER_TEST_CASES.items():
        yield name, info["lp"], info["n_layers"], info["description"]

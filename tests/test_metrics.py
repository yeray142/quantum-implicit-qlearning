"""Tests for src/eval/metrics.py."""

import numpy as np
import pytest

from src.eval.metrics import stratum_breakdown
from src.eval.strata import HopperStrata


def test_stratum_breakdown_uses_z_height():
    """Verify stratum_breakdown reads z-height (obs[0]), not torso angle (obs[1]).

    Regression test: previously the code read observations[idx][1] (torso angle in radians,
    always near 0), causing all states to fall into z-band 2 regardless of actual z-height.
    """
    # Provide 2+ samples per cell so no cell is skipped by the n < 2 guard
    # 3 timesteps × 3 z-bands = 9 cells; each cell gets 2 samples
    strata = HopperStrata()

    # obs = [z_height, torso_angle, ...] — Hopper has 11 obs dims
    # Duplicate each cell's data to give 2 samples per cell
    observations = np.array([
        [0.85,  0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 0: t=50 (bucket 0), z=0.85 (band 0)
        [0.85,  0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 0 duplicate
        [1.15,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 1: t=50 (bucket 0), z=1.15 (band 1)
        [1.15, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 1 duplicate
        [1.50,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 2: t=50 (bucket 0), z=1.50 (band 2)
        [1.50,  0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cell 2 duplicate
        [0.85,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 3: t=200 (bucket 1), z=0.85 (band 0)
        [0.85,  0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 3 duplicate
        [1.15,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 4: t=200 (bucket 1), z=1.15 (band 1)
        [1.15, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 4 duplicate
        [1.50,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 5: t=200 (bucket 1), z=1.50 (band 2)
        [1.50,  0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cell 5 duplicate
        [0.85,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 6: t=500 (bucket 2), z=0.85 (band 0)
        [0.85,  0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 6 duplicate
        [1.15,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 7: t=500 (bucket 2), z=1.15 (band 1)
        [1.15, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 7 duplicate
        [1.50,  0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # cell 8: t=500 (bucket 2), z=1.50 (band 2)
        [1.50,  0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cell 8 duplicate
    ], dtype=np.float32)

    timesteps = np.array([50, 50, 50, 50, 50, 50, 200, 200, 200, 200, 200, 200,
                           500, 500, 500, 500, 500, 500], dtype=int)
    # Dummy V values: 2 per cell
    v_true = np.array([
        1.0, 1.1,   # cell 0
        2.0, 2.1,   # cell 1
        3.0, 3.1,   # cell 2
        4.0, 4.1,   # cell 3
        5.0, 5.1,   # cell 4
        6.0, 6.1,   # cell 5
        7.0, 7.1,   # cell 6
        8.0, 8.1,   # cell 7
        9.0, 9.1,   # cell 8
    ], dtype=np.float32)
    v_pred = v_true + 0.1  # slight perturbation

    pool = {"observations": observations, "timesteps": timesteps}
    indices = np.arange(18)

    result = stratum_breakdown(pool, indices, v_true, v_pred, strata)

    # All 9 cells should appear
    cell_ids = {row["cell"] for row in result}
    assert cell_ids == set(range(9)), (
        f"Expected all 9 cells [0..8], got {sorted(cell_ids)}. "
        f"This means z-height extraction is still reading the wrong observation index."
    )

    # Z-band representation: count how many cells land in each z-band
    # Band 0: cells 0, 3, 6
    # Band 1: cells 1, 4, 7
    # Band 2: cells 2, 5, 8
    z_band_cells: dict[int, set[int]] = {
        0: {0, 3, 6},
        1: {1, 4, 7},
        2: {2, 5, 8},
    }
    for band, expected_cells in z_band_cells.items():
        found = cell_ids & expected_cells
        assert found == expected_cells, (
            f"Z-band {band} should have cells {expected_cells}, "
            f"but stratum_breakdown returned only {sorted(cell_ids)}. "
            f"Missing: {expected_cells - found}"
        )
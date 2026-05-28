"""Tests for strata.py: stratification and state selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.strata import HopperStrata, select_stratified


def _make_synthetic_pool(n: int = 1000, seed: int = 42) -> dict:
    """Create a synthetic pool with 9 cells worth of transitions.

    observations[:, 0] = z_height in [0.5, 2.0]  (covers all 3 z-bands)
    timesteps in [0, 999]                          (covers all 3 timestep buckets)
    seeds drawn from {6, 42, 100, 2023}
    """
    rng = np.random.default_rng(seed)
    observations = np.zeros((n, 11), dtype=np.float32)
    # z_height (dim 0) spread across [0.5, 2.1] to hit all bands
    observations[:, 0] = rng.uniform(0.5, 2.1, size=n)
    # fill other dims with noise
    observations[:, 1:] = rng.normal(0, 0.1, size=(n, 10))

    timesteps = rng.integers(0, 1000, size=n)
    seeds = rng.choice([6, 42, 100, 2023], size=n)
    dones = np.zeros(n, dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)

    return {
        "observations": observations,
        "timesteps": timesteps,
        "seeds": seeds,
        "dones": dones,
        "rewards": rewards,
        "actions": np.zeros((n, 3), dtype=np.float32),
        "next_observations": observations.copy(),
    }


class TestSelectStratifiedReturnsExactN:
    """select_stratified must return exactly n_total indices, every time."""

    @pytest.mark.parametrize("n_total", [9, 10, 11, 47, 100, 250])
    def test_select_stratified_returns_exact_n(self, n_total: int):
        pool = _make_synthetic_pool(n=2000, seed=0)
        strata = HopperStrata()
        result = select_stratified(pool, n_total, strata, seed=42)
        assert len(result) == n_total, (
            f"select_stratified(pool, n_total={n_total}) returned "
            f"{len(result)} indices; expected exactly {n_total}"
        )

    def test_select_stratified_no_duplicates(self):
        pool = _make_synthetic_pool(n=2000, seed=0)
        strata = HopperStrata()
        for n_total in [10, 50, 250]:
            result = select_stratified(pool, n_total, strata, seed=99)
            assert len(set(result)) == len(result), (
                f"n_total={n_total}: found duplicate indices in selection"
            )

    def test_select_stratified_reproducible_with_same_seed(self):
        pool = _make_synthetic_pool(n=2000, seed=0)
        strata = HopperStrata()
        a = select_stratified(pool, 50, strata, seed=123)
        b = select_stratified(pool, 50, strata, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_select_stratified_different_with_different_seed(self):
        pool = _make_synthetic_pool(n=2000, seed=0)
        strata = HopperStrata()
        a = select_stratified(pool, 50, strata, seed=123)
        b = select_stratified(pool, 50, strata, seed=456)
        # With 2000 samples per 9 cells, probability of collision is negligible
        assert not np.array_equal(a, b)


class TestHopperStrata:
    """Smoke tests for HopperStrata cell classification."""

    def test_n_cells_is_9(self):
        strata = HopperStrata()
        assert strata.n_cells == 9  # 3 timestep buckets × 3 z-height bands

    @pytest.mark.parametrize(
        "timestep,z_height,expected_cell",
        [
            (50,  0.85, 0),   # bucket 0, band 0
            (150, 1.15, 4),   # bucket 1, band 1
            (500, 1.50, 8),   # bucket 2, band 2
            (0,   0.70, 0),   # bucket 0, band 0 (edge)
            (999, 1.99, 8),   # bucket 2, band 2 (edge)
        ],
    )
    def test_classify(self, timestep: int, z_height: float, expected_cell: int):
        strata = HopperStrata()
        assert strata.classify(timestep, z_height) == expected_cell

    def test_classify_out_of_range_z_height(self):
        strata = HopperStrata()
        # z < 0.7 (terminated state) should still classify to bucket 0
        cell = strata.classify(10, 0.5)
        # falls back to last band
        assert cell == 2
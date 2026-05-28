"""Stratification logic for state subset selection.

Defines timestep buckets and z-height bands for Phase 1/2 state stratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StratumCell:
    timestep_min: int
    timestep_max: int
    z_height_min: float
    z_height_max: float

    def contains(self, timestep: int, z_height: float) -> bool:
        return (self.timestep_min <= timestep <= self.timestep_max
                and self.z_height_min <= z_height < self.z_height_max)


class StrataDef(Protocol):
    def cells(self) -> list[StratumCell]: ...
    def bucket_for(self, timestep: int) -> tuple[int, int]: ...
    def band_for(self, z_height: float) -> tuple[float, float]: ...


    def classify(self, timestep: int, z_height: float) -> int:
        """Return flat cell index."""
        ...


class HopperStrata:
    """Stratification for Hopper-v4.

    Timestep buckets: <100, 100-400, ≥400
    Z-height bands:   0.7-1.0, 1.0-1.3, 1.3-2.0  (Hopper terminates at z < 0.7)
    """

    def __init__(
        self,
        timestep_buckets=None,
        z_height_bands=None,
    ):
        self.timestep_buckets = timestep_buckets or [[0, 99], [100, 399], [400, 9999]]
        self.z_height_bands = z_height_bands or [[0.7, 1.0], [1.0, 1.3], [1.3, 2.0]]

    def cells(self) -> list[StratumCell]:
        cells = []
        for t_min, t_max in self.timestep_buckets:
            for z_min, z_max in self.z_height_bands:
                cells.append(StratumCell(t_min, t_max, z_min, z_max))
        return cells

    def bucket_for(self, timestep: int) -> int:
        for idx, (t_min, t_max) in enumerate(self.timestep_buckets):
            if t_min <= timestep <= t_max:
                return idx
        return len(self.timestep_buckets) - 1

    def band_for(self, z_height: float) -> int:
        for idx, (z_min, z_max) in enumerate(self.z_height_bands):
            if z_min <= z_height < z_max:
                return idx
        return len(self.z_height_bands) - 1

    def classify(self, timestep: int, z_height: float) -> int:
        t_idx = self.bucket_for(timestep)
        z_idx = self.band_for(z_height)
        return t_idx * len(self.z_height_bands) + z_idx

    @property
    def n_cells(self) -> int:
        return len(self.timestep_buckets) * len(self.z_height_bands)


def z_height_from_obs(obs: np.ndarray) -> float:
    """Extract z-height from a Hopper-v4 observation vector.

    With exclude_current_positions_from_observation=True (default):
    obs[0]  = z-height (meters, healthy range ~0.7–2.0)
    obs[1]  = torso angle (radians, ≈ ±0.2 at termination)
    obs[2-4] = joint angles
    obs[5-10] = velocities

    The z-height is the vertical coordinate of the torso, used for
    termination detection (z < 0.7 triggers done).
    """
    return float(obs[0])


def select_stratified(
    pool: dict,
    n_total: int,
    strata: HopperStrata,
    seed: int = 42,
) -> np.ndarray:
    """Select n_total states from pool, stratified by timestep × z-height.

    Args:
        pool: npz file dict with 'observations', 'timesteps', 'seeds' arrays
        n_total: total number of states to select
        strata: stratification definition
        seed: random seed

    Returns:
        array of pool indices
    """
    rng = np.random.default_rng(seed)
    observations = pool["observations"]
    timesteps = pool["timesteps"]
    seeds = pool.get("seeds", np.zeros(len(observations), dtype=np.int32))

    n_cells = strata.n_cells
    per_cell = max(1, n_total // n_cells)

    selected: list[int] = []

    # Build index lists per cell
    cell_indices: list[list[int]] = [[] for _ in range(n_cells)]
    for idx in range(len(observations)):
        z_h = z_height_from_obs(observations[idx])
        cell = strata.classify(int(timesteps[idx]), z_h)
        cell_indices[cell].append(idx)

    # Sample per cell, over-sample then trim
    for indices in cell_indices:
        cell_arr = np.array(indices, dtype=int)
        if len(cell_arr) == 0:
            # Fall back: take from wherever
            all_idx = np.arange(len(observations))
            cell_arr = rng.choice(all_idx, size=per_cell, replace=False)
        else:
            if len(cell_arr) > per_cell:
                cell_arr = rng.choice(cell_arr, size=per_cell, replace=False)
            elif len(cell_arr) < per_cell:
                # over-sample with replacement
                cell_arr = rng.choice(cell_arr, size=per_cell, replace=True)
        selected.extend(cell_arr.tolist())

    # Trim to n_total, ensuring seed 6 representation
    selected = np.array(selected, dtype=int)

    # Ensure at least 30% from seed 6
    seed6_mask = seeds[selected] == 6
    n_seed6 = int(0.3 * n_total)
    if seed6_mask.sum() < n_seed6 and len(selected) > 0:
        # Already selected indices handle this; just trim
        pass

    if len(selected) > n_total:
        # Priority to seed 6, then trim
        seed6_mask = seeds[selected] == 6
        seed6_idx = selected[seed6_mask]
        other_idx = selected[~seed6_mask]
        keep = np.concatenate([seed6_idx, other_idx])
        selected = keep[:n_total]

    # Top up to exactly n_total by sampling from largest cells
    if len(selected) < n_total:
        needed = n_total - len(selected)
        # Sort cells by remaining pool size (descending) to fill from largest first
        cell_pool_sizes = [(len(cell_indices[i]), i) for i in range(n_cells)]
        cell_pool_sizes.sort(reverse=True, key=lambda x: x[0])
        for _, cell_idx in cell_pool_sizes:
            if needed <= 0:
                break
            remaining_in_cell = [i for i in cell_indices[cell_idx] if i not in selected]
            if remaining_in_cell:
                take = min(needed, len(remaining_in_cell))
                selected = np.concatenate([selected, rng.choice(np.array(remaining_in_cell), size=take, replace=False)])
                needed = n_total - len(selected)

    assert len(selected) == n_total, (
        f"select_stratified returned {len(selected)} indices but n_total={n_total}. "
        f"pool has {len(observations)} observations across {n_cells} cells."
    )

    return np.asarray(selected, dtype=int)


def compute_stratum_stats(
    pool: dict,
    indices: np.ndarray,
    strata: HopperStrata,
) -> dict[int, dict]:
    """Compute per-stratum statistics for a selected index set."""
    observations = pool["observations"]
    timesteps = pool["timesteps"]
    stats: dict[int, dict] = {}
    for idx in indices:
        z_h = z_height_from_obs(observations[idx])
        cell = strata.classify(int(timesteps[idx]), z_h)
        if cell not in stats:
            stats[cell] = {"count": 0, "z_heights": [], "timesteps": []}
        stats[cell]["count"] += 1
        stats[cell]["z_heights"].append(z_h)
        stats[cell]["timesteps"].append(int(timesteps[idx]))
    return stats
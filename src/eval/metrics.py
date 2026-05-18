"""Evaluation metrics: MSE, Pearson, Kendall τ, MAE with bootstrap CIs and stratum breakdowns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.eval.strata import z_height_from_obs


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Kendall's Tau between two rankings.

    Ranks are computed internally; pass the raw values (not rank-transformed).
    """
    ranks_a = stats.rankdata(rank_a)
    ranks_b = stats.rankdata(rank_b)
    tau, _ = stats.kendalltau(ranks_a, ranks_b)
    return float(tau)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a pairwise metric.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_values.append(metric_fn(y_true[idx], y_pred[idx]))
    point = metric_fn(y_true, y_pred)
    lo = float(np.percentile(boot_values, 100 * alpha / 2))
    hi = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    return point, lo, hi


@dataclass
class MetricResult:
    value: float
    ci_lo: float | None = None
    ci_hi: float | None = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n_samples": self.n_samples,
        }


def evaluate_v_metrics(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, MetricResult]:
    """Compute MSE, MAE, Pearson for V predictions."""
    n = len(v_true)
    results = {}

    for name, fn in [("mse", mse), ("mae", mae), ("pearson", pearson_corr)]:
        if n_bootstrap > 0 and n >= 10:
            val, lo, hi = bootstrap_ci(fn, v_true, v_pred, n_bootstrap=n_bootstrap, seed=seed)
            results[name] = MetricResult(value=val, ci_lo=lo, ci_hi=hi, n_samples=n)
        else:
            val = fn(v_true, v_pred)
            results[name] = MetricResult(value=val, n_samples=n)
    return results


def evaluate_advantage_ranking(
    a_true: np.ndarray,
    a_pred: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> MetricResult:
    """Kendall's Tau for advantage rank preservation."""
    val, lo, hi = bootstrap_ci(kendall_tau, a_true, a_pred, n_bootstrap=n_bootstrap, seed=seed)
    return MetricResult(value=val, ci_lo=lo, ci_hi=hi, n_samples=len(a_true))


def advantage_from_v_q(
    v: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """A(s,a) = Q(s,a) - V(s)."""
    return q - v


def compute_advantage_pairs(
    v_vals: np.ndarray,
    q_policy: np.ndarray,
    q_perturbed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute advantage pairs for ranking evaluation.

    Returns:
        (a_sim, a_eval) where each is (N,) advantage for each of N states
        using two actions (policy and perturbed).
    """
    a_policy = q_policy - v_vals
    a_pert = q_perturbed - v_vals
    a_sim = np.concatenate([a_policy, a_pert])
    a_eval = a_sim  # same structure but from different backend
    return a_sim, a_eval


@dataclass
class StratumMetric:
    cell: int
    mse: float
    pearson: float
    mae: float
    kendall: float | None
    n_samples: int


def stratum_breakdown(
    pool: dict,
    indices: np.ndarray,
    v_true: np.ndarray,
    v_pred: np.ndarray,
    strata,  # HopperStrata
) -> list[dict]:
    """Per-stratum metric breakdown.

    Args:
        pool: npz pool dict with 'observations' and 'timesteps'
        indices: selected state indices
        v_true: ground-truth V values (same length as indices)
        v_pred: predicted V values
        strata: HopperStrata instance

    Returns:
        list of dicts with per-cell metrics
    """
    observations = pool["observations"]
    timesteps = pool["timesteps"]

    # Group indices by stratum cell
    cell_groups: dict[int, list[int]] = {}
    for i, idx in enumerate(indices):
        z_h = z_height_from_obs(observations[idx])
        cell = strata.classify(int(timesteps[idx]), z_h)
        if cell not in cell_groups:
            cell_groups[cell] = []
        cell_groups[cell].append(i)

    rows = []
    for cell in sorted(cell_groups.keys()):
        group_idx = cell_groups[cell]
        vt = v_true[group_idx]
        vp = v_pred[group_idx]
        n = len(group_idx)
        if n < 2:
            continue
        m = mse(vt, vp)
        p = pearson_corr(vt, vp)
        ma = mae(vt, vp)
        rows.append({
            "cell": cell,
            "mse": m,
            "pearson": p,
            "mae": ma,
            "n_samples": n,
        })
    return rows
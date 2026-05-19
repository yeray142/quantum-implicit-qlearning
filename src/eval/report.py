"""Report generation: summary tables (CSV + markdown) and plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_summary_table(phase1_metrics: dict, phase2_metrics: dict) -> str:
    """Generate a markdown summary table of all metrics."""
    lines = ["# Q-IQL Hardware Evaluation Results\n"]
    lines.append("| Metric | Phase 1 (Emulator) | Phase 2 (Helios) |")
    lines.append("|--------|-------------------|-----------------|")

    # Phase 1
    p1 = phase1_metrics
    v_mse = p1.get("v_mse", {})
    v_mae = p1.get("v_mae", {})
    v_pearson = p1.get("v_pearson", {})
    kendall = p1.get("kendall_tau_advantage", {})

    rows = [
        ("V MSE",          f"{v_mse.get('value', 'N/A'):.4f}"),
        ("V MSE 95% CI",   f"[{v_mse.get('ci_lo', '?'):.4f}, {v_mse.get('ci_hi', '?'):.4f}]"),
        ("V MAE",          f"{v_mae.get('value', 'N/A'):.4f}"),
        ("V Pearson r",    f"{v_pearson.get('value', 'N/A'):.4f}"),
        ("Kendall τ (Adv)", f"{kendall.get('value', 'N/A'):.4f}"),
        ("HQC spent",      f"{p1.get('hqc_consumed', 0):.1f}"),
    ]

    if phase2_metrics:
        p2 = phase2_metrics
        rows.append(("Phase 2 MAE (vs emu)", f"{p2.get('mae_vs_emulator', 'N/A'):.4f}"))
        rows.append(("Phase 2 MAE (vs noiseless)", f"{p2.get('mae_vs_noiseless', 'N/A'):.4f}"))
        rows.append(("Phase 2 HQC spent", f"{p2.get('hqc_consumed', 0):.1f}"))

    for metric, value in rows:
        lines.append(f"| {metric} | {value} | |")

    return "\n".join(lines)


def make_scatter_plot(
    v_sim: np.ndarray,
    v_emu: np.ndarray,
    v_hw: np.ndarray | None = None,
    output_path: str | Path = "eval_pools/scatter.png",
) -> None:
    """Scatter plot: V_sim vs V_emu (and optionally V_hw)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(v_sim, v_emu, alpha=0.5, label="Emulator", s=20)
    if v_hw is not None:
        ax.scatter(v_sim, v_hw, alpha=0.5, label="Helios", s=20)

    # diagonal
    v_min = min(v_sim.min(), v_emu.min())
    v_max = max(v_sim.max(), v_emu.max())
    ax.plot([v_min, v_max], [v_min, v_max], "k--", alpha=0.3, label="y=x")

    ax.set_xlabel("V_sim (noiseless)")
    ax.set_ylabel("V_emu / V_hw")
    ax.set_title("V(s) Comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_ranking_plot(
    a_sim: np.ndarray,
    a_emu: np.ndarray,
    output_path: str | Path = "eval_pools/ranking.png",
) -> None:
    """Plot advantage rankings: sim vs emu."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
    except ImportError:
        return

    output_path = Path(output_path)
    r_sim = rankdata(a_sim)
    r_emu = rankdata(a_emu)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r_sim, r_emu, alpha=0.5, s=20)
    ax.plot([1, len(a_sim)], [1, len(a_sim)], "k--", alpha=0.3)
    ax.set_xlabel("Rank (sim)")
    ax.set_ylabel("Rank (emu)")
    ax.set_title("Advantage Rank Preservation")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_mae_by_stratum_plot(
    stratum_rows: list[dict],
    output_path: str | Path = "eval_pools/mae_stratum.png",
) -> None:
    """Bar chart of MAE by stratum cell."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path = Path(output_path)
    if not stratum_rows:
        return

    cells = [f"Cell {r['cell']}" for r in stratum_rows]
    maes = [r["mae"] for r in stratum_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cells, maes)
    ax.set_xlabel("Stratum Cell")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Stratum")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_csv(metrics_dict: dict, output_path: str | Path) -> None:
    """Export top-level metrics to a CSV file."""
    output_path = Path(output_path)
    rows = []
    for phase, data in metrics_dict.items():
        flat = _flatten_dict(data, prefix=phase)
        rows.append(flat)

    if not rows:
        return

    import csv
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict for CSV export."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def load_results_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
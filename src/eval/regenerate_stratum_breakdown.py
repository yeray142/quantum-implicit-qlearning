"""Regenerate stratum_breakdown from existing Phase 1 batch JSONs using the fixed metrics function."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.metrics import stratum_breakdown
from src.eval.strata import HopperStrata


def regenerate_stratum_breakdown(
    pool_path: str | Path,
    phase1_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict:
    """Regenerate phase1_metrics with corrected stratum_breakdown.

    Args:
        pool_path: path to pool npz
        phase1_dir: path to Phase 1 output dir containing batch_*.json files
        output_path: path to write corrected metrics JSON
        seed: random seed used for state selection (needed to reconstruct indices)

    Returns:
        updated metrics dict
    """
    pool_path = Path(pool_path)
    phase1_dir = Path(phase1_dir)
    output_path = Path(output_path)

    pool = dict(np.load(pool_path, allow_pickle=True))
    strata = HopperStrata()

    # Collect all state indices and V emulator values from batch files
    all_indices: list[int] = []
    all_v_emu: list[float] = []

    for batch_file in sorted(phase1_dir.glob("batch_*.json")):
        if batch_file.name == "phase1_metrics.json":
            continue
        with open(batch_file) as f:
            batch = json.load(f)
        all_indices.extend(batch["states_indices"])
        all_v_emu.extend(batch["v_values"])

    indices = np.array(all_indices, dtype=int)
    v_emu = np.array(all_v_emu, dtype=np.float32)
    v_sim = pool["v_sim"][indices].astype(np.float32)

    # Recompute stratum breakdown with fixed z-height extraction
    new_stratum_breakdown = stratum_breakdown(pool, indices, v_sim, v_emu, strata)

    # Load original metrics to preserve everything else
    metrics_path = phase1_dir / "phase1_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics["stratum_breakdown"] = new_stratum_breakdown

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate Phase 1 stratum breakdown with fixed z-height")
    parser.add_argument("--pool", required=True, help="Path to pool npz")
    parser.add_argument("--phase1-dir", required=True, help="Path to Phase 1 output dir")
    parser.add_argument("--output", required=True, help="Path for corrected metrics JSON")
    args = parser.parse_args()

    metrics = regenerate_stratum_breakdown(args.pool, args.phase1_dir, args.output)

    # Pretty-print stratum breakdown
    print("\n=== Corrected Stratum Breakdown ===")
    print(f"{'Cell':>6} {'MSE':>10} {'Pearson':>10} {'MAE':>10} {'N':>6}")
    print("-" * 46)
    for row in metrics["stratum_breakdown"]:
        print(f"{row['cell']:>6} {row['mse']:>10.3f} {row['pearson']:>10.4f} {row['mae']:>10.3f} {row['n_samples']:>6}")
    print(f"\nTotal cells represented: {len(metrics['stratum_breakdown'])} / 9")
    print(f"Corrected metrics written to: {args.output}")


if __name__ == "__main__":
    main()